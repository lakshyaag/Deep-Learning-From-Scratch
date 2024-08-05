import matplotlib.pyplot as plt
import numpy as np
import skimage.transform
import torch
from PIL import Image
from torchvision import transforms


def get_split_ids(split):
    with open(f"./data/Flickr_8k.{split}Images.txt") as f:
        return f.read().splitlines()


def get_transforms(split):
    if split == "train":
        train_transform = transforms.Compose(
            [
                transforms.Resize(size=(256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        return train_transform

    test_transform = transforms.Compose(
        [
            transforms.Resize(size=(256, 256)),
            transforms.ToTensor(),
        ]
    )

    return test_transform


def reverse_transform(x, train=True):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if train:
        x = x * torch.tensor(std).unsqueeze(1).unsqueeze(1)
        x = x + torch.tensor(mean).unsqueeze(1).unsqueeze(1)

    return transforms.ToPILImage()(x)


def visualize_model_attention(image, generated_caption, alphas, tokenizer, smooth=True):
    alphas = torch.FloatTensor(alphas)

    image = reverse_transform(image, train=False)
    image = image.resize([14 * 24, 14 * 24], Image.LANCZOS)

    caption = [tokenizer.itos[i] for i in generated_caption]

    plt.figure(figsize=(10, 6))
    for t in range(len(caption)):
        plt.subplot(int(np.ceil(len(caption) / 5.0)), 5, t + 1)

        plt.text(0, 1, caption[t], color="black", backgroundcolor="white", fontsize=12)
        plt.imshow(image)

        alpha = alphas[t, :]

        if smooth:
            alpha = skimage.transform.pyramid_expand(alpha.numpy(), upscale=24, sigma=8)
        else:
            alpha = skimage.transform.resize(alpha.numpy(), [14 * 24, 14 * 24])

        if t == 0:
            plt.imshow(alpha, alpha=0.0)
        else:
            plt.imshow(alpha, alpha=0.7)

        plt.set_cmap("grey")
        plt.axis("off")

    plt.show()
