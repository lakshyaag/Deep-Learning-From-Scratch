import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import typer
from config import Config
from data import ImageTextDataset, get_split_ids, load_captions
from models import CLIP
from rich import print, prompt
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm
from utils import reverse_transform

from transformers import AutoTokenizer

cfg = Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_image_embeddings(model: CLIP, loader: DataLoader) -> torch.Tensor:
    embeddings = []
    with torch.no_grad():
        for idx, item in enumerate(tqdm(loader, desc="Embeddings")):
            image = item["image"].to(device)
            image_features = model.image_encoder(image)
            image_embeddings = model.image_projection(image_features)
            embeddings.append(image_embeddings)

    return torch.cat(embeddings)


def find_matches(
    tokenizer: AutoTokenizer,
    model: CLIP,
    image_embeddings: torch.Tensor,
    query: str,
    dataset: ImageTextDataset,
    n: int = 8,
) -> list[torch.Tensor]:
    encoded_query = tokenizer([query])
    item = {key: torch.tensor(value).to(device) for key, value in encoded_query.items()}

    with torch.no_grad():
        text_features = model.text_encoder(item["input_ids"])
        text_embeddings = model.text_projection(text_features)

    image_embeddings_normalized = F.normalize(image_embeddings, p=2, dim=-1)
    text_embeddings_normalized = F.normalize(text_embeddings, p=2, dim=-1)

    dot_product = text_embeddings_normalized @ image_embeddings_normalized.T
    _, topk_indices = torch.topk(dot_product.squeeze(0), k=n, dim=-1)
    matched_imgs = [dataset[idx]["image"] for idx in topk_indices if idx < len(dataset)]

    return matched_imgs


def main(weights: str = "final_clip-openai.pt", n: int = 8):
    """
    Find images that match the specified query using the specified CLIP model.

    Args:

    - weights (str): Path to the model weights.

    - n (int): Number of images to display
    """
    captions_dict = load_captions("Flickr8k.token.txt")
    tokenizer = AutoTokenizer.from_pretrained(cfg.TEXT_MODEL)

    test_ids = get_split_ids("test")
    test_dataset = ImageTextDataset(
        tokenizer=tokenizer,
        captions_dict=captions_dict,
        target_size=cfg.IMAGE_SHAPE[1:],
        split_ids=test_ids,
        max_length=cfg.TOKENIZER_MAX_LEN,
    )
    test_loader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False)

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

    model.load_state_dict(torch.load(f"./data/models/{weights}", map_location=device))

    image_embeddings = get_image_embeddings(model, test_loader)

    while True:
        query = prompt.Prompt.ask("Enter a query to find matching images")
        print(f"[bold green]Query:[/bold green] {query}")
        matched_imgs = find_matches(
            tokenizer, model, image_embeddings, query, test_dataset, n=n
        )

        plt.figure(figsize=(10, 6))
        plt.imshow(
            reverse_transform(make_grid(matched_imgs, nrow=4, padding=10, pad_value=1))
        )
        plt.title(f"Query: {query}")
        plt.axis("off")
        plt.show()

        if not prompt.Confirm.ask("Do you want to enter another query?"):
            break


if __name__ == "__main__":
    typer.run(main)
