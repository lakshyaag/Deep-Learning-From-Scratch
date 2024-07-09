import torch
from attention import SelfAttention
from torch import nn
from torch.nn import functional as F


class VaeResidualBlock(nn.Module):
    """
    This class defines a residual block used in the decoder part of a Variational Autoencoder (VAE).
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, padding=0
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B C_in H W

        residue = x

        # B C_in H W -> B C_in H W
        x = self.groupnorm_1(x)
        x = F.silu(x)

        # B C_in H W -> B C_out H W
        x = self.conv_1(x)

        # B C_out H W -> B C_out H W
        x = self.groupnorm_2(x)

        x = F.silu(x)

        # B C_out H W -> B C_out H W
        x = self.conv_2(x)

        # x_out: B C_out H W
        # residue: B C_in H W -> B C_out H W
        return x + self.residual_layer(residue)


class VaeAttentionBlock(nn.Module):
    """
    This class defines the attention block used in the decoder part of a Variational Autoencoder (VAE).
    """

    def __init__(self, channels: int):
        super().__init__()

        self.groupnorm = nn.GroupNorm(32, channels)

        self.attention = SelfAttention(1, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B C H W

        residue = x

        n, c, h, w = x.shape

        # B C H W -> B C H*W
        x = x.view(n, c, h * w)

        # B C H*W -> B H*W C
        x = x.transpose(-1, -2)

        # B H*W C -> B H*W C
        x = self.attention(x)

        # B H*W C -> B C H*W
        x = x.transpose(-1, -2)

        # B C H*W -> B C H W
        x = x.view((n, c, h, w))

        x += residue

        # x_out: B C H W
        return x


class VaeDecoder(nn.Sequential):
    """
    This class defines the decoder part of a Variational Autoencoder (VAE).
    The decoder's role is to take a latent space representation and transform it back into an image.
    """

    def __init__(self):
        super().__init__(
            # ---------------------------------------------------- #
            # B 4 H/8 W/8 -> B 4 H/8 W/8
            nn.Conv2d(4, 4, kernel_size=1, padding=0),
            # ---------------------------------------------------- #
            # B 4 H/8 W/8 -> B 512 H/8 W/8
            nn.Conv2d(4, 512, kernel_size=3, padding=1),
            # ---------------------------------------------------- #
            # B 512 H/8 W/8 -> B 512 H/8 W/8
            VaeResidualBlock(512, 512),
            # ---------------------------------------------------- #
            # B 512 H/8 W/8 -> B 512 H/8 W/8
            VaeAttentionBlock(512),
            # ---------------------------------------------------- #
            # B 512 H/8 W/8 -> B 512 H/8 W/8
            VaeResidualBlock(512, 512),
            VaeResidualBlock(512, 512),
            VaeResidualBlock(512, 512),
            VaeResidualBlock(512, 512),
            # ---------------------------------------------------- #
            # B 512 H/8 W/8 -> B 512 H/4 W/4
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            # ---------------------------------------------------- #
            # B 512 H/4 W/4 -> B 512 H/4 W/4
            VaeResidualBlock(512, 512),
            VaeResidualBlock(512, 512),
            VaeResidualBlock(512, 512),
            # ---------------------------------------------------- #
            # B 512 H/8 W/8 -> B 512 H/2 W/2
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            # ---------------------------------------------------- #
            # B 512 H/2 W/2 -> B 256 H/2 W/2
            VaeResidualBlock(512, 256),
            # ---------------------------------------------------- #
            # B 256 H/2 W/2 -> B 256 H/2 W/2
            VaeResidualBlock(256, 256),
            VaeResidualBlock(256, 256),
            # ---------------------------------------------------- #
            # B 256 H/2 W/2 -> B 256 H W
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            # ---------------------------------------------------- #
            # B 256 H W -> B 128 H W
            VaeResidualBlock(256, 128),
            # ---------------------------------------------------- #
            # B 128 H W -> B 128 H W
            VaeResidualBlock(128, 128),
            VaeResidualBlock(128, 128),
            # ---------------------------------------------------- #
            # B 128 H W -> B 128 H W
            nn.GroupNorm(32, 128),
            nn.SiLU(),
            # ---------------------------------------------------- #
            # B 128 H W -> B 3 H W
            nn.Conv2d(128, 3, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B 4 H/8 W/8

        x /= 0.18215

        for module in self:
            x = module(x)

        # x_out: B 3 H W
        return x
