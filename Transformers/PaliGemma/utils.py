import einops
import numpy as np
from typing import List, Dict
from torchvision import transforms as T
from PIL import Image
import torch

IMAGENET_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STD = [0.5, 0.5, 0.5]


def add_image_tokens_to_prompt(
    prefix_prompt: str, bos_token: str, image_seq_len: int, image_token: str
) -> str:
    return f"{image_token*image_seq_len}{bos_token}{prefix_prompt}\n"


def reverse_process_images(
    images: torch.Tensor,
):
    # images: (b, c, h, w)

    images = images * torch.tensor(IMAGENET_STD).view(1, 3, 1, 1) + torch.tensor(
        IMAGENET_MEAN
    ).view(1, 3, 1, 1)

    return T.ToPILImage()(images)


def process_images(
    images: List[Image.Image],
    size: Dict[str, int] = None,
    resample: T.InterpolationMode = None,
) -> List[np.ndarray]:
    # images: b[h, w, c]
    transform = T.Compose(
        [
            T.Resize(size, interpolation=resample),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    processed_images = [transform(image) for image in images]

    return processed_images


def repeat_kv(
    x: torch.Tensor,
    n_repeat: int,
) -> torch.Tensor:
    b, n, s, d = x.shape

    if n_repeat == 1:
        return x

    x = einops.repeat(x, "b n s d -> b n r s d", r=n_repeat)
    x = einops.rearrange(x, "b n r s d -> b (n r) s d")

    return x


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    # Build the `sin` part of the rotation matrix

    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]

    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    q_embd = (q * cos) + (rotate_half(q) * sin)
    k_embd = (k * cos) + (rotate_half(k) * sin)

    return q_embd, k_embd
