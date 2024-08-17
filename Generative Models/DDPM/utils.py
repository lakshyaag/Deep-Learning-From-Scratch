from torchvision import transforms
import torch


def get_transforms(IMAGE_SIZE: int):
    transform = transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x - 0.5) * 2),
        ]
    )
    return transform


def reverse_transform(x: torch.Tensor):
    return transforms.ToPILImage()((x / 2 + 0.5).clamp(0, 1))


def linear_schedule(start_value: float, end_value: float, n_steps: int) -> torch.Tensor:
    return torch.linspace(
        start_value,
        end_value,
        n_steps,
    )


def cosine_schedule(n_steps: int, offset: float = 0.008) -> torch.Tensor:
    """
    Cosine noise schedule as described in the "Improved Denoising Diffusion Probabilistic Models" paper by Nichol & Dhariwal (2021).
    """
    steps = n_steps + 1
    t = torch.linspace(0, n_steps, steps)

    alphas_hat = torch.cos(((t / n_steps) + offset) / (1 + offset) * torch.pi * 0.5)
    alphas_hat = alphas_hat / alphas_hat[0]

    betas = 1 - (alphas_hat[1:] / alphas_hat[:-1])
    return torch.clip(betas, 0.0001, 0.9999)
