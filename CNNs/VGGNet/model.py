import torch
from torch import nn


class Conv2dBlock(nn.Module):
    """
    A helper class to create a block of Conv2d -> BatchNorm2d -> ReLU layers to be used in the VGGNet model.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 64,
        kernel_size: int = 3,
        padding: int = 1,
        stride: int = 1,
    ):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class VGGNet(nn.Module):
    def __init__(self, n_classes: int = 10, debug: bool = False):
        super().__init__()
        self.debug = debug
        self.n_classes = n_classes

        # B x 3 x 224 x 224 -> B x 64 x 112 x 112
        self.layer1 = nn.Sequential(
            # B x 3 x 224 x 224 -> B x 64 x 224 x 224
            Conv2dBlock(in_channels=3, out_channels=64, kernel_size=3),
            # B x 64 x 224 x 224 -> B x 64 x 224 x 224
            Conv2dBlock(in_channels=64, out_channels=64, kernel_size=3),
            # B x 64 x 224 x 224 -> B x 64 x 112 x 112
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # B x 64 x 112 x 112 -> B x 128 x 56 x 56
        self.layer2 = nn.Sequential(
            # B x 64 x 112 x 112 -> B x 128 x 112 x 112
            Conv2dBlock(in_channels=64, out_channels=128, kernel_size=3),
            # B x 128 x 112 x 112 -> B x 128 x 112 x 112
            Conv2dBlock(in_channels=128, out_channels=128, kernel_size=3),
            # B x 128 x 112 x 112 -> B x 128 x 56 x 56
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # B x 128 x 56 x 56 -> B x 256 x 28 x 28
        self.layer3 = nn.Sequential(
            # B x 128 x 56 x 56 -> B x 256 x 56 x 56
            Conv2dBlock(in_channels=128, out_channels=256, kernel_size=3),
            # B x 256 x 56 x 56 -> B x 256 x 56 x 56
            Conv2dBlock(in_channels=256, out_channels=256, kernel_size=3),
            # B x 256 x 56 x 56 -> B x 256 x 56 x 56
            Conv2dBlock(in_channels=256, out_channels=256, kernel_size=3),
            # B x 256 x 56 x 56 -> B x 256 x 28 x 28
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # B x 256 x 28 x 28 -> B x 512 x 14 x 14
        self.layer4 = nn.Sequential(
            # B x 256 x 28 x 28 -> B x 512 x 28 x 28
            Conv2dBlock(in_channels=256, out_channels=512, kernel_size=3),
            # B x 512 x 28 x 28 -> B x 512 x 28 x 28
            Conv2dBlock(in_channels=512, out_channels=512, kernel_size=3),
            # B x 512 x 28 x 28 -> B x 512 x 28 x 28
            Conv2dBlock(in_channels=512, out_channels=512, kernel_size=3),
            # B x 512 x 28 x 28 -> B x 512 x 14 x 14
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # B x 512 x 14 x 14 -> B x 512 x 7 x 7
        self.layer5 = nn.Sequential(
            # B x 512 x 14 x 14 -> B x 512 x 14 x 14
            Conv2dBlock(in_channels=512, out_channels=512, kernel_size=3),
            # B x 512 x 14 x 14 -> B x 512 x 14 x 14
            Conv2dBlock(in_channels=512, out_channels=512, kernel_size=3),
            # B x 512 x 14 x 14 -> B x 512 x 14 x 14
            Conv2dBlock(in_channels=512, out_channels=512, kernel_size=3),
            # B x 512 x 14 x 14 -> B x 512 x 7 x 7
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # B x 512 x 7 x 7 -> B x 4096
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=(512 * 7 * 7), out_features=4096),
            nn.ReLU(),
        )

        # B x 4096 -> B x 4096
        self.fc2 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
        )

        # B x 4096 -> # B x N_CLASSES
        self.fc3 = nn.Linear(4096, self.n_classes)

    def _debug_print(self, tensor, name):
        if self.debug:
            print(f"{name}: {tensor.shape}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._debug_print(x, "Input")
        x = self.layer1(x)
        self._debug_print(x, "Layer1")
        x = self.layer2(x)
        self._debug_print(x, "Layer2")
        x = self.layer3(x)
        self._debug_print(x, "Layer3")
        x = self.layer4(x)
        self._debug_print(x, "Layer4")
        x = self.layer5(x)
        self._debug_print(x, "Layer5")
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        self._debug_print(x, "FC1")
        x = self.fc2(x)
        self._debug_print(x, "FC2")
        x = self.fc3(x)
        self._debug_print(x, "Output")

        return x
