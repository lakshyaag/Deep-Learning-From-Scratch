import itertools

import torch
from config import Config
from data import ImageTextDataset, get_split_ids, load_captions
from models import CLIP
from rich import print
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformers import AutoTokenizer

cfg = Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data(caption_file_path: str):
    """
    Load the training and validation datasets.

    Args:
    - caption_file_path (str): Path to the caption file.

    Returns:
    - train_dataset (ImageTextDataset): Training dataset.
    - val_dataset (ImageTextDataset): Validation dataset.
    """
    captions_dict = load_captions(caption_file_path)

    train_ids = get_split_ids("train")
    val_ids = get_split_ids("dev")

    tokenizer = AutoTokenizer.from_pretrained(cfg.TEXT_MODEL)

    train_dataset = ImageTextDataset(
        tokenizer=tokenizer,
        captions_dict=captions_dict,
        target_size=cfg.IMAGE_SHAPE[1:],
        split_ids=train_ids,
        max_length=cfg.TOKENIZER_MAX_LEN,
    )
    val_dataset = ImageTextDataset(
        tokenizer=tokenizer,
        captions_dict=captions_dict,
        target_size=cfg.IMAGE_SHAPE[1:],
        split_ids=val_ids,
        max_length=cfg.TOKENIZER_MAX_LEN,
    )

    return (train_dataset, val_dataset)


def train_epoch(
    loader: DataLoader,
    model: CLIP,
    optimizer: torch.optim.Optimizer,
    epoch: int,
) -> float:
    """
    Train the model for one epoch.

    Args:
    - loader (DataLoader): Training data loader.
    - model (CLIP): Model to train.
    - optimizer (torch.optim.Optimizer): Optimizer for the model.
    - epoch (int): Current epoch number.
    """
    model.train()
    train_loss = 0.0
    for idx, item in enumerate(tqdm(loader, desc=f"Training Epoch {epoch}")):
        image = item["image"].to(device)
        input_ids = item["input_ids"].to(device)
        attn_mask = item["attention_mask"].to(device)

        loss = model(image, input_ids, attn_mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * image.size(0)

    train_loss /= len(loader.dataset)

    return train_loss


def valid_epoch(
    loader: DataLoader,
    model: CLIP,
    epoch: int,
):
    """
    Validate the model for one epoch.

    Args:
    - loader (DataLoader): Validation data loader.
    - model (CLIP): Model to validate.
    - epoch (int): Current epoch number.
    """

    model.eval()
    val_loss = 0.0
    for idx, item in enumerate(tqdm(loader, desc=f"Validation Epoch {epoch}")):
        image = item["image"].to(device)
        input_ids = item["input_ids"].to(device)
        attn_mask = item["attention_mask"].to(device)

        with torch.no_grad():
            loss = model(image, input_ids, attn_mask)

        val_loss += loss.item() * image.size(0)

    val_loss /= len(loader.dataset)

    return val_loss


def main():
    (train_dataset, val_dataset) = load_data("Flickr8k.token.txt")

    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False)

    model = CLIP(
        text_model=cfg.TEXT_MODEL,
        d_img_model=cfg.D_IMG_MODEL,
        d_text_model=cfg.D_TEXT_MODEL,
        d_embd=cfg.D_EMBD,
        temp=cfg.INITIAL_TEMPERATURE,
        dropout=cfg.DROPOUT,
        trainable=cfg.TRAINABLE,
        debug=False,
    ).to(device)

    params = [
        {"params": model.image_encoder.parameters(), "lr": cfg.IMAGE_LEARNING_RATE},
        {"params": model.text_encoder.parameters(), "lr": cfg.TEXT_LEARNING_RATE},
        {
            "params": itertools.chain(
                model.image_projection.parameters(), model.text_projection.parameters()
            ),
            "lr": cfg.LEARNING_RATE,
            "weight_decay": cfg.WEIGHT_DECAY,
        },
    ]

    optimizer = torch.optim.AdamW(params, weight_decay=0.0)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total Parameters: {total_params / 1e6:.2f}M")

    best_val_loss = float("inf")

    print(f"Starting model training with config: {cfg}")

    for epoch in range(cfg.N_EPOCHS):
        train_loss = train_epoch(train_loader, model, optimizer, epoch)
        val_loss = valid_epoch(val_loader, model, epoch)

        print(
            f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                model.state_dict(),
                f"./data/models/best_{cfg.EXPERIMENT_NAME}.pt",
            )
            print(f"Best model saved at epoch {epoch}")

    # Save the final model
    torch.save(
        model.state_dict(),
        f"./data/models/final_{cfg.EXPERIMENT_NAME}.pt",
    )

    print(
        f'Model training complete after {cfg.N_EPOCHS} epochs. Model saved as "final_{cfg.EXPERIMENT_NAME}.pt"'
    )


if __name__ == "__main__":
    main()
