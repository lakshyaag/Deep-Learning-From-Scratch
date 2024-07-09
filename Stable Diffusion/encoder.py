import torch
from decoder import VaeAttentionBlock, VaeResidualBlock
from torch import nn
from torch.nn import functional as F


class VaeEncoder(nn.Sequential):
    """
    This class defines the encoder part of a Variational Autoencoder (VAE)
    The encoder's role is to take an input image and transform it into a latent space representation.
    """

    def __init__(self):
        super().__init__(
            # ---------------------------------------------------- #
            # B 3 H W -> B 128 H W
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            VaeResidualBlock(128, 128),
            VaeResidualBlock(128, 128),
            # ---------------------------------------------------- #
            # B 128 H W -> B 128 H/2 W/2
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),
            # ---------------------------------------------------- #
            # B 128 H W -> B 256 H/2 W/2
            VaeResidualBlock(128, 256),
            VaeResidualBlock(256, 256),
            # ---------------------------------------------------- #
            # B 256 H/2 W/2 -> B 256 H/4 W/4
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),
            # ---------------------------------------------------- #
            # B 256 H/2 W/2 -> B 512 H/4 W/4
            VaeResidualBlock(256, 512),
            # ---------------------------------------------------- #
            # B 512 H/4 W/4 -> B 512 H/8 W/8
            VaeResidualBlock(512, 512),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),
            VaeResidualBlock(512, 512),
            VaeResidualBlock(512, 512),
            VaeResidualBlock(512, 512),
            # ---------------------------------------------------- #
            # B 512 H/8 W/8 -> B 512 H/8 W/8
            VaeAttentionBlock(512),
            VaeResidualBlock(512, 512),
            # ---------------------------------------------------- #
            # B 512 H/8 W/8 -> B 512 H/8 W/8
            nn.GroupNorm(32, 512),
            nn.SiLU(),
            # ---------------------------------------------------- #
            # B 512 H/8 W/8 -> B 8 H/8 W/8
            nn.Conv2d(512, 8, kernel_size=3, padding=1),
            # ---------------------------------------------------- #
            # B 8 H/8 W/8 -> B 8 H/8 W/8
            nn.Conv2d(8, 8, kernel_size=1, padding=0),
        )

    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # x: B 3 H W
        # noise: B 8 H/8 W/8

        for module in self:
            if getattr(module, "stride", None) == (2, 2):
                # Pad the right and bottom part of the image
                x = F.pad(x, (0, 1, 0, 1))

            x = module(x)

        # B 8 H/8 W/8 -> B 4 H/8 W/8, B 4 H/8 W/8
        mean, log_variance = torch.chunk(x, 2, dim=1)

        # B 4 H/8 W/8 -> B 4 H/8 W/8
        log_variance = torch.clamp(log_variance, -30, 30)

        # Calculate standard deviation
        variance = log_variance.exp()
        std_dev = variance.sqrt()

        # Sample from the distribution
        x = mean + std_dev * noise

        # Scale output (info here: https://github.com/huggingface/diffusers/issues/437#issuecomment-1241827515)
        x *= 0.18215

        # x_out: B 4 H/8 W/8
        return x
