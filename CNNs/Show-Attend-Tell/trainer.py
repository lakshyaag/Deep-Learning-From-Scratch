import wandb
import numpy as np
import torch
import torch.nn as nn
from config import Config
from data import (
    CustomCollate,
    Flickr8kDataset,
    get_vocab,
    load_captions,
)
from models import ImageCaptioningModel
from nltk.translate.bleu_score import corpus_bleu
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import get_split_ids, get_transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wandb.require("core")


def load_data():
    captions_dict = load_captions()
    tokenizer = get_vocab(captions_dict)

    cfg = Config(VOCAB_SIZE=len(tokenizer))

    train_ids = get_split_ids("train")
    val_ids = get_split_ids("dev")
    test_ids = get_split_ids("test")

    # Create datasets
    train_dataset = Flickr8kDataset(
        tokenizer,
        train_ids,
        captions_dict,
        split="train",
        transform=get_transforms("train"),
    )
    val_dataset = Flickr8kDataset(
        tokenizer, val_ids, captions_dict, split="val", transform=get_transforms("val")
    )
    test_dataset = Flickr8kDataset(
        tokenizer,
        test_ids,
        captions_dict,
        split="test",
        transform=get_transforms("test"),
    )

    pad_idx = tokenizer.stoi["<PAD>"]

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        collate_fn=CustomCollate(pad_idx=pad_idx),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        collate_fn=CustomCollate(pad_idx=pad_idx, split="val"),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # Batch size of 1 to prevent batched beam search errors
        shuffle=False,
        collate_fn=CustomCollate(pad_idx=pad_idx, split="test"),
    )

    return (train_loader, val_loader, test_loader), tokenizer, cfg


def train_step(loader, model, loss_fn, optimizer, epoch):
    model.train()

    losses = []
    for idx, (imgs, captions, caplens) in enumerate(
        tqdm(loader, desc=f"Epoch {epoch}")
    ):
        imgs = imgs.to(device)
        captions = captions.to(device)
        caplens = caplens.to(device)

        (
            image_features,
            predictions,
            caption_tokens,
            decode_lengths,
            alphas,
            sort_index,
        ) = model(imgs, captions, caplens)

        targets = caption_tokens[:, 1:]

        predictions = pack_padded_sequence(
            predictions, decode_lengths, batch_first=True
        )[0]
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)[0]

        loss = loss_fn(predictions, targets)

        # Doubly stochastic attention regularization
        # if alphas is not None:
        #     loss += (1 - alphas.sum(dim=1) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()

        clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        losses.append(loss.item())

    wandb.log({"train_loss": np.mean(losses)}, step=epoch)
    return np.mean(losses)


def val_step(loader, model, loss_fn, epoch, tokenizer):
    model.eval()

    losses = []
    top5accs = []

    references = list()
    hypotheses = list()

    with torch.no_grad():
        for idx, (imgs, captions, caplens, allcaps) in enumerate(
            tqdm(loader, desc=f"Epoch {epoch}")
        ):
            imgs = imgs.to(device)
            captions = captions.to(device)
            caplens = caplens.to(device)

            (
                image_features,
                predictions,
                caption_tokens,
                decode_lengths,
                alphas,
                sort_index,
            ) = model(imgs, captions, caplens)

            targets = caption_tokens[:, 1:]

            predictions_copy = predictions.clone()

            predictions = pack_padded_sequence(
                predictions, decode_lengths, batch_first=True
            )[0]
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)[0]

            # Calculate loss
            loss = loss_fn(predictions, targets)

            # Doubly stochastic attention regularization
            # if alphas is not None:
            #     loss += (1 - alphas.sum(dim=1) ** 2).mean()

            # Accuracy@5
            _, ind = predictions.topk(k=5, dim=1, largest=True, sorted=True)
            correct = ind.eq(targets.view(-1, 1).expand_as(ind))
            correct_total = correct.view(-1).float().sum()
            accuracy = correct_total.item() * (100.0 / targets.size(0))

            losses.append(loss.item())
            top5accs.append(accuracy)

            # BLEU evaluation
            # Reference: [[ref1a, ref1b, ref1c], [ref2a, ref2b, ref2c], ...]
            allcaps = [allcaps[i] for i in sort_index]
            for j in range(len(allcaps)):
                img_caps = allcaps[j]

                img_captions = list(
                    tokenizer.tokenize(img_caps[j]) for j in range(len(img_caps))
                )
                references.append(img_captions)

            # Hypothesis: [hyp1, hyp2, ...]
            _, preds = predictions_copy.max(dim=2)
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(
                    [tokenizer.itos[word] for word in preds[j][: decode_lengths[j]]]
                )

            preds = temp_preds
            hypotheses.extend(preds)

        bleu_4 = corpus_bleu(references, hypotheses)

        wandb.log(
            {
                "val_loss": np.mean(losses),
                "val_top5acc": np.mean(top5accs),
                "val_bleu4": bleu_4,
            }
        )
        return np.mean(losses), np.mean(top5accs), bleu_4


def test_step(loader, model, tokenizer, beam_size=5, max_caption_length=50):
    model.eval()

    references = list()
    hypotheses = list()

    with torch.no_grad():
        for idx, (imgs, captions, caplens, allcaps) in enumerate(
            tqdm(loader, desc="Testing")
        ):
            imgs = imgs.to(device)
            captions = captions.to(device)
            caplens = caplens.to(device)

            final_seqs, _ = model.generate_caption_beam_search(
                imgs, beam_size=beam_size, max_caption_length=max_caption_length
            )

            # BLEU evaluation
            # Reference: [[ref1a, ref1b, ref1c], [ref2a, ref2b, ref2c], ...]
            for j in range(len(allcaps)):
                img_caps = allcaps[j]

                img_captions = list(
                    tokenizer.tokenize(img_caps[j]) for j in range(len(img_caps))
                )
                references.append(img_captions)

            # Hypothesis: [hyp1, hyp2, ...]
            hypotheses.append(
                [
                    tokenizer.itos[word]
                    for word in final_seqs
                    if word
                    not in {
                        tokenizer.stoi["<SOS>"],
                        tokenizer.stoi["<EOS>"],
                        tokenizer.stoi["<PAD>"],
                    }
                ]
            )

        bleu_4 = corpus_bleu(references, hypotheses)

    wandb.log({"test_bleu4": bleu_4})
    return bleu_4


def main():
    (train_loader, val_loader, test_loader), tokenizer, cfg = load_data()

    model = ImageCaptioningModel(
        tokenizer=tokenizer,
        encoded_image_size=cfg.ENCODED_IMAGE_SIZE,
        embed_dim=cfg.EMBED_DIM,
        decoder_dim=cfg.DECODER_DIM,
        attention_dim=cfg.ATTENTION_DIM,
        vocab_size=cfg.VOCAB_SIZE,
        encoder_dim=cfg.ENCODER_DIM,
        debug=False,
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        params=[p for p in model.parameters() if p.requires_grad],
        lr=cfg.LEARNING_RATE,
    )

    wandb.init(
        project=cfg.EXPERIMENT_NAME,
        config={
            **cfg.__dict__,
            "Optimizer": optimizer.__class__.__name__,
            "Loss Function": criterion.__class__.__name__,
            "ENCODER_MODEL": "resnet101",
            "DECODER_MODEL": "LSTM",
        },
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters in the model: {total_params}")

    best_val_bleu4 = 0.0

    for epoch in tqdm(range(cfg.N_EPOCHS), desc="Epochs"):
        train_loss = train_step(train_loader, model, criterion, optimizer, epoch)
        print(f"Epoch: {epoch} | Train Loss: {train_loss:.4f}")

        val_loss, val_top5acc, val_bleu = val_step(
            val_loader, model, criterion, epoch, tokenizer
        )
        print(
            f"Epoch: {epoch} | Val Loss: {val_loss:.4f} | Val Top-5 Acc: {val_top5acc:.2f} | Val BLEU-4: {val_bleu:.4f}"
        )

        if val_bleu > best_val_bleu4:
            best_val_bleu4 = val_bleu
            torch.save(model.state_dict(), "data/models/best_model.pth")

    torch.save(model.state_dict(), "data/models/final_model.pth")
    wandb.log_model("./data/models/final_model.pth", "show-attend-tell")

    test_bleu4 = test_step(
        test_loader,
        model,
        tokenizer,
        beam_size=cfg.BEAM_SIZE,
        max_caption_length=cfg.MAX_CAPTION_LENGTH,
    )

    print(f"Test-set BLEU-4 score: {test_bleu4:.4f}")


if __name__ == "__main__":
    main()
