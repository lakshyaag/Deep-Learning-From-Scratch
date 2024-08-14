from torchvision.transforms import v2
import torch


def reverse_transform(x: torch.Tensor):
    """
    Reverse the normalization of the image tensor. 

    Args:
    - x (torch.Tensor): Normalized image tensor.

    Returns:
    - PIL.Image: Image tensor in PIL format.
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).unsqueeze(1).unsqueeze(1)
    std = torch.tensor([0.229, 0.224, 0.225]).unsqueeze(1).unsqueeze(1)

    x = x * std + mean

    return v2.ToPILImage()(x)

