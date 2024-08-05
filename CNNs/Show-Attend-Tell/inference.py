import argparse

import torch
from config import Config
from data import get_vocab, load_captions
from models import ImageCaptioningModel
from PIL import Image
from utils import get_transforms, visualize_model_attention

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_path, tokenizer, cfg):
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

    model.load_state_dict(torch.load(model_path, map_location=device))

    model.eval()
    return model


def preprocess_image(image_path):
    transform = get_transforms("test")
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)
    return image.to(device)


def generate_caption(image_path, model, tokenizer, beam_size=5, max_caption_length=50):
    image = preprocess_image(image_path)
    gen_caption, gen_alphas = model.generate_caption_beam_search(
        image, beam_size=beam_size, max_caption_length=max_caption_length
    )
    caption = " ".join([tokenizer.itos[i] for i in gen_caption])
    print(f"Generated caption: {caption}")
    visualize_model_attention(image.squeeze(0), gen_caption, gen_alphas, tokenizer)
    return caption


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Captioning Inference")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the model file"
    )
    parser.add_argument(
        "--image_path", type=str, required=True, help="Path to the image file"
    )
    parser.add_argument(
        "--beam_size", type=int, default=5, help="Beam size for beam search"
    )
    parser.add_argument(
        "--max_caption_length", type=int, default=50, help="Maximum caption length"
    )
    args = parser.parse_args()

    captions_dict = load_captions()
    tokenizer = get_vocab(captions_dict)
    cfg = Config(VOCAB_SIZE=len(tokenizer))

    model = load_model(args.model_path, tokenizer, cfg)
    generate_caption(
        args.image_path, model, tokenizer, args.beam_size, args.max_caption_length
    )
