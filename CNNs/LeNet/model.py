import torch
from torch import nn
from torch.nn import functional as F


class LeNet5(nn.Module):
    def __init__(self, n_classes: int = 10, debug: bool = False):
        super().__init__()
        self.debug = debug
        self.n_classes = n_classes

        # B x 1 x 32 x 32 -> B x 6 x 28 x 28
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0
        )
        # B x 6 x 28 x 28 -> B x 6 x 14 x 14
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # B x 6 x 14 x 14 -> B x 16 x 10 x 10
        self.conv2 = nn.Conv2d(
            in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0
        )

        # B x 16 x 10 x 10 -> B x 16 x 5 x 5
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # B x 16 x 5 x 5 -> # B x 400
        self.flatten = nn.Flatten()

        # B x 400 -> # B x 120
        self.fc1 = nn.Linear(400, 120)

        # B x 120 -> # B x 84
        self.fc2 = nn.Linear(120, 84)

        # B x 84 -> # B x N_CLASSES
        self.fc3 = nn.Linear(84, self.n_classes)

    def _debug_print(self, tensor, name):
        if self.debug:
            print(f"{name}: {tensor.shape}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._debug_print(x, "Input")
        x = F.relu(self.conv1(x))
        self._debug_print(x, "Conv1")
        x = self.pool1(x)
        self._debug_print(x, "Pool1")
        x = F.relu(self.conv2(x))
        self._debug_print(x, "Conv2")
        x = self.pool2(x)
        self._debug_print(x, "Pool2")
        x = self.flatten(x)
        self._debug_print(x, "Flatten")
        x = F.relu(self.fc1(x))
        self._debug_print(x, "FC1")
        x = F.relu(self.fc2(x))
        self._debug_print(x, "FC2")
        x = F.relu(self.fc3(x))
        self._debug_print(x, "FC3")

        return x
