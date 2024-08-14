import random

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2

from transformers import AutoTokenizer


def load_captions(file_name: str) -> dict:
    """
    Load captions from the specified file path.

    Args:
    - file_name (str): Path to the file containing the captions.

    Returns:
    - dict: Dictionary containing the captions.
    """
    with open(f"./data/{file_name}") as f:
        captions = f.read().splitlines()

    captions_dict = {}

    for line in captions:
        contents = line.split("\t")

        if len(contents) < 2:
            continue

        filename, caption = contents[0], contents[1]

        filename = filename[:-2]

        if filename in captions_dict.keys():
            captions_dict[filename].append(caption)
        else:
            captions_dict[filename] = [caption]

    return captions_dict


def get_split_ids(split: str) -> list:
    """
    Get the image IDs for the specified split.

    Args:
    - split (str): Split for which to get the image IDs.

    Returns:
    - list: List of image IDs for the specified split.
    """

    with open(f"./data/Flickr_8k.{split}Images.txt") as f:
        return f.read().splitlines()


class ImageTextDataset(Dataset):
    """
    Dataset class for the image-text dataset.

    Args:
    - tokenizer (AutoTokenizer): Tokenizer for the captions.
    - captions_dict (dict): Dictionary containing the captions.
    - target_size (tuple): Target size for the images.
    - split_ids (list): List of image IDs for the split to use.
    - max_length (int): Maximum length of the captions.

    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        captions_dict: dict,
        target_size: (int, int),
        split_ids: list,
        max_length: int = 200,
    ):
        super().__init__()

        self.tokenizer = tokenizer
        self.target_size = target_size
        self.max_length = max_length
        self.image_ids = split_ids
        self.captions_dict = captions_dict

        self.transforms = v2.Compose(
            [
                v2.Resize(target_size),
                v2.ToImage(),
                v2.ToDtype(dtype=torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.images, self.captions = self.fetch_dataset()
        self.encoded_captions = self.tokenizer(
            self.captions,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

    def fetch_dataset(self) -> tuple:
        images = [id for id in self.image_ids]

        random.seed(42)

        captions = [random.choice(self.captions_dict[id]) for id in self.image_ids]

        return images, captions

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index) -> dict:
        """
        Get the item at the specified index.

        Args:
        - index (int): Index of the item.

        Returns:
        - dict: Dictionary containing the image, caption, tokenized input IDs, and attention mask.
        """
        item = {key: value[index] for key, value in self.encoded_captions.items()}

        image_id = self.images[index]

        image = Image.open(f"./data/Flicker8k_Dataset/{image_id}").convert("RGB")
        image = self.transforms(image)

        item["image"] = image
        item["caption"] = self.captions[index]

        return item
