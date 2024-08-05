import re
from random import choice, seed

import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {
            0: "<PAD>",
            1: "<SOS>",
            2: "<EOS>",
            3: "<UNK>",
        }

        self.stoi = {v: k for k, v in self.itos.items()}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenize(text):
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", "", text)
        return text.split()

    def build_vocabulary(self, captions):
        frequencies = {}
        idx = 4

        for caption in captions:
            for word in self.tokenize(caption):
                if word not in frequencies:
                    frequencies[word] = 1
                else:
                    frequencies[word] += 1

                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenize(text)

        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
        ]


class CustomCollate:
    def __init__(self, pad_idx, split="train"):
        self.pad_idx = pad_idx
        self.split = split

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=True, padding_value=self.pad_idx)

        if self.split != "train":
            return (
                imgs,
                targets,
                torch.stack([item[2] for item in batch]).squeeze(1),
                [item[3] for item in batch],
            )

        return imgs, targets, torch.stack([item[2] for item in batch]).squeeze(1)


class Flickr8kDataset(Dataset):
    def __init__(
        self, tokenizer, image_ids, captions_dict, split="train", transform=None
    ):
        self.data = []
        self.transform = transform
        self.tokenizer = tokenizer
        self.split = split

        seed(42)
        for img_id in image_ids:
            if img_id in captions_dict:
                caption = choice(captions_dict[img_id])
                self.data.append((img_id, caption))

        if self.split != "train":
            self.captions_dict = captions_dict

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_id, caption = self.data[idx]
        img_path = f"./data/images/{img_id}"
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        numericalized_captions = [self.tokenizer.stoi["<SOS>"]]
        numericalized_captions += self.tokenizer.numericalize(caption)
        numericalized_captions.append(self.tokenizer.stoi["<EOS>"])

        if self.split != "train":
            return (
                image,
                torch.tensor(numericalized_captions),
                torch.tensor([len(numericalized_captions)]),
                self.captions_dict[img_id],
            )

        return (
            image,
            torch.tensor(numericalized_captions),
            torch.tensor([len(numericalized_captions)]),
        )


def load_captions():
    with open("./data/Flickr8k.token.txt") as f:
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


def get_vocab(captions_dict):
    tokenizer = Vocabulary(freq_threshold=5)

    tokenizer.build_vocabulary(
        [caption for captions in captions_dict.values() for caption in captions]
    )

    print(f"Total number of tokens in vocabulary: {len(tokenizer)}")
    print(
        f'Sample tokenized caption: {tokenizer.numericalize("A black dog is running")}'
    )

    return tokenizer
