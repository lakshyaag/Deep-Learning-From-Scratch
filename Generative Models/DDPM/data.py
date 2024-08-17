from torch.utils.data import Dataset


class DDPMDataset(Dataset):
    """
    Custom dataset class that wraps a list of images, applying a transformation if provided.

    Args:
    - dataset (list): List of images.
    - transform (callable, optional): A function/transform to apply to each image
    """

    def __init__(self, dataset: list, transform: callable = None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]
        if self.transform:
            img = self.transform(img)
        return img
