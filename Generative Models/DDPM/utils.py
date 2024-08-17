from torchvision import transforms


def get_transforms(IMAGE_SIZE):
    transform = transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x - 0.5) * 2),
        ]
    )
    return transform


def reverse_transform(x):
    return transforms.ToPILImage()((x / 2 + 0.5).clamp(0, 1))
