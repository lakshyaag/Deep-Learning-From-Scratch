from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST


def load_data(train=True):
    dataset = MNIST(
        root="./data/", download=True, train=train, transform=transforms.ToTensor()
    )
    mean_pixel = dataset.data.float().mean() / 255
    std_pixel = dataset.data.float().std() / 255
    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((mean_pixel,), (std_pixel,)),
        ]
    )
    return MNIST(root="./data/", download=True, train=train, transform=transform)


def prepare_data_loaders(BATCH_SIZE: int, train_split: float = 0.7):
    full_train_dataset = load_data(train=True)
    test_dataset = load_data(train=False)

    # Calculate the sizes for training and validation datasets (70-30 split)
    train_size = int(train_split * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size

    # Split the full training dataset into training and validation datasets
    train_dataset, val_dataset = random_split(
        full_train_dataset, [train_size, val_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    return train_loader, val_loader, test_loader
