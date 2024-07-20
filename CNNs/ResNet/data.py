from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import CIFAR100


def load_data(train=True):
    dataset = CIFAR100(
        root="./data/", download=True, train=train, transform=transforms.ToTensor()
    )

    mean_pixel = dataset.data.mean(axis=(0, 1, 2)) / 255
    std_pixel = dataset.data.std(axis=(0, 1, 2)) / 255

    if train:
        transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean_pixel, std_pixel),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean_pixel, std_pixel),
            ]
        )

    return CIFAR100(root="./data/", download=True, train=train, transform=transform)


def prepare_data_loaders(BATCH_SIZE: int, train_split: float = 0.7):
    true_labels = [
        "apple",
        "aquarium_fish",
        "baby",
        "bear",
        "beaver",
        "bed",
        "bee",
        "beetle",
        "bicycle",
        "bottle",
        "bowl",
        "boy",
        "bridge",
        "bus",
        "butterfly",
        "camel",
        "can",
        "castle",
        "caterpillar",
        "cattle",
        "chair",
        "chimpanzee",
        "clock",
        "cloud",
        "cockroach",
        "couch",
        "crab",
        "crocodile",
        "cup",
        "dinosaur",
        "dolphin",
        "elephant",
        "flatfish",
        "forest",
        "fox",
        "girl",
        "hamster",
        "house",
        "kangaroo",
        "keyboard",
        "lamp",
        "lawn_mower",
        "leopard",
        "lion",
        "lizard",
        "lobster",
        "man",
        "maple_tree",
        "motorcycle",
        "mountain",
        "mouse",
        "mushroom",
        "oak_tree",
        "orange",
        "orchid",
        "otter",
        "palm_tree",
        "pear",
        "pickup_truck",
        "pine_tree",
        "plain",
        "plate",
        "poppy",
        "porcupine",
        "possum",
        "rabbit",
        "raccoon",
        "ray",
        "road",
        "rocket",
        "rose",
        "sea",
        "seal",
        "shark",
        "shrew",
        "skunk",
        "skyscraper",
        "snail",
        "snake",
        "spider",
        "squirrel",
        "streetcar",
        "sunflower",
        "sweet_pepper",
        "table",
        "tank",
        "telephone",
        "television",
        "tiger",
        "tractor",
        "train",
        "trout",
        "tulip",
        "turtle",
        "wardrobe",
        "whale",
        "willow_tree",
        "wolf",
        "woman",
        "worm",
    ]

    full_train_dataset = load_data(train=True)
    test_dataset = load_data(train=False)

    # Calculate the sizes for training and validation datasets (70-30 split)
    train_size = int(0.7 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size

    # Split the full training dataset into training and validation datasets
    train_dataset, val_dataset = random_split(
        full_train_dataset, [train_size, val_size]
    )

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

    return train_loader, val_loader, test_loader, true_labels
