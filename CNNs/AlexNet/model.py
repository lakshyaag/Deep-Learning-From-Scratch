import torch
from torch import nn


class AlexNet(nn.Module):
    def __init__(self, n_classes: int = 10, debug: bool = False):
        super().__init__()
        self.debug = debug
        self.n_classes = n_classes

        # B x 3 x 227 x 227 -> B x 96 x 27 x 27
        self.layer1 = nn.Sequential(
            # B x 3 x 227 x 227 -> B x 96 x 55 x 55
            nn.Conv2d(
                in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=0
            ),
            nn.ReLU(),
            nn.BatchNorm2d(96),
            # B x 96 x 55 x 55 -> B x 96 x 27 x 27
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        # B x 96 x 27 x 27 -> B x 256 x 13 x 13
        self.layer2 = nn.Sequential(
            # B x 96 x 27 x 27 -> B x 256 x 27 x 27
            nn.Conv2d(
                in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2
            ),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            # B x 256 x 27 x 27 -> B x 256 x 13 x 13
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        # B x 256 x 13 x 13 -> B x 384 x 13 x 13
        self.layer3 = nn.Sequential(
            nn.Conv2d(
                in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.BatchNorm2d(384),
        )

        # B x 384 x 13 x 13 -> B x 384 x 13 x 13
        self.layer4 = nn.Sequential(
            nn.Conv2d(
                in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.BatchNorm2d(384),
        )

        # B x 384 x 13 x 13 -> B x 256 x 6 x 6
        self.layer5 = nn.Sequential(
            # B x 384 x 13 x 13 -> B x 256 x 13 x 13
            nn.Conv2d(
                in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            # B x 256 x 13 x 13 -> B x 256 x 6 x 6
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        # B x 256 x 6 x 6 -> B x 4096
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=(256 * 6 * 6), out_features=4096),
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

        self._init_layers()

    def _debug_print(self, tensor, name):
        if self.debug:
            print(f"{name}: {tensor.shape}")

    def _init_layers(self):
        for layer in self.children():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                # Initializing weights with mean=0, std=0.01
                nn.init.normal_(layer.weight, mean=0, std=0.01)

                if isinstance(layer, nn.Conv2d):
                    # Initializing bias with 0 for conv layers
                    nn.init.constant_(layer.bias, 0)
                elif isinstance(layer, nn.Linear):
                    # Initializing bias with 1 for linear layers
                    nn.init.constant_(layer.bias, 1)

        # Original paper mentions setting bias for 2nd, 4th and 5th conv layers to 1
        nn.init.constant_(self.layer2[0].bias, 1)
        nn.init.constant_(self.layer4[0].bias, 1)
        nn.init.constant_(self.layer5[0].bias, 1)

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
        x = x.reshape(x.size(0), -1)
        self._debug_print(x, "Resized")
        x = self.fc1(x)
        self._debug_print(x, "FC1")
        x = self.fc2(x)
        self._debug_print(x, "FC2")
        x = self.fc3(x)
        self._debug_print(x, "Output")

        return x
