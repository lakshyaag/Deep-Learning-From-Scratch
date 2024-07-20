import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    """
    A residual block for the ResNet architecture. The block consists of two convolutional layers with batch normalization and ReLU activation functions.
    The left path consists of two convolutional layers, while the right path consists of an optional down-sampling convolutional layer if the input and output dimensions do not match.
    """

    def __init__(self, in_channels: int, out_channels: int, first_stride: int = 1):
        super().__init__()

        self.left = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=first_stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=out_channels),
        )

        if first_stride > 1:
            self.right = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=first_stride,
                    bias=False,
                ),
                nn.BatchNorm2d(num_features=out_channels),
            )
        else:
            assert (
                in_channels == out_channels
            ), "in_channels must be equal to out_channels"
            self.right = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        left = self.left(x)
        right = self.right(x)
        return F.relu(left + right)


class ResNet(nn.Module):
    def __init__(self, n_classes: int = 10, n_blocks: int = 5, debug: bool = False):
        super().__init__()

        self.debug = debug
        self.n_blocks = n_blocks
        self.n_classes = n_classes

        self.layer1 = nn.Sequential(
            # B x 3 x 32 x 32 -> B x 16 x 32 x 32
            nn.Conv2d(3, 16, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=16),
        )

        # B x 16 x 32 x 32 -> B x 16 x 32 x 32
        self.residual_layer1 = nn.Sequential(
            *[ResidualBlock(16, 16, 1) for _ in range(self.n_blocks)]
        )

        # B x 16 x 32 x 32 -> B x 32 x 16 x 16
        self.residual_layer2 = nn.Sequential(
            ResidualBlock(16, 32, 2),
            *[ResidualBlock(32, 32, 1) for _ in range(self.n_blocks - 1)],
        )

        # B x 32 x 16 x 16 -> B x 64 x 8 x 8
        self.residual_layer3 = nn.Sequential(
            ResidualBlock(32, 64, 2),
            *[ResidualBlock(64, 64, 1) for _ in range(self.n_blocks - 1)],
        )

        # B x 64 x 8 x 8 -> B x 10
        self.out_layers = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_features=64, out_features=self.n_classes),
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def _debug_print(self, tensor, name):
        if self.debug:
            print(f"{name}: {tensor.shape}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._debug_print(x, "Input")
        x = self.layer1(x)
        self._debug_print(x, "Layer1")
        x = self.residual_layer1(x)
        self._debug_print(x, "ResidualLayer1")
        x = self.residual_layer2(x)
        self._debug_print(x, "ResidualLayer2")
        x = self.residual_layer3(x)
        self._debug_print(x, "ResidualLayer3")
        x = self.out_layers(x)
        self._debug_print(x, "Output")

        return x
