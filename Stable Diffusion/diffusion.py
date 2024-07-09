import torch
from torch import nn
from torch.nn import functional as F
from unet import UNet, UNetOutput


class TimeEmbedding(nn.Module):
    """
    This class defines the time embedding layer for the diffusion model.
    """

    def __init__(self, n_embd: int):
        super().__init__()

        self.linear_1 = nn.Linear(n_embd, n_embd * 4)
        self.linear_2 = nn.Linear(n_embd * 4, n_embd * 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (1, 320)

        x = self.linear_1(x)
        x = F.silu(x)
        x = self.linear_2(x)

        # x_out: (1, 1280)
        return x


class Diffusion(nn.Module):
    """
    This class defines the diffusion model used in the Stable Diffusion framework.
    It consists of a time embedding layer, a UNet model, and an output layer.

    The time embedding layer encodes the time step information into a higher-dimensional space.
    The UNet model processes the latent variable along with the context and time embeddings to generate intermediate feature maps.
    The output layer reduces the number of channels in the feature maps to produce the final output.
    """

    def __init__(self):
        super().__init__()
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNet()
        self.final = UNetOutput(320, 4)

    def forward(
        self, latent: torch.Tensor, context: torch.LongTensor, time: torch.Tensor
    ):
        # latent: B 4 H/8 W/8
        # context: B S E
        # time: (1, 320)

        # (1, 320) -> (1, 1280)
        time = self.time_embedding(time)

        # B 4 H/8 W/8 -> B 320 H/8 W/8
        output = self.unet(latent, context, time)

        # B 320 H/8 W/8 -> # B 4 H/8 W/8
        output = self.final(output)

        return output
