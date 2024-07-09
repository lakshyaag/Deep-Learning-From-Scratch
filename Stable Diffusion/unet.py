import torch
from attention import CrossAttention, SelfAttention
from torch import nn
from torch.nn import functional as F


class UNetUpsample(nn.Module):
    """
    This class defines the upsampling layer for the UNet model.
    It includes a convolutional layer that processes the upsampled feature maps.
    """

    def __init__(self, channels: int):
        super().__init__()

        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B C H W

        # B C H W -> B C H*2 W*2
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        # out: B C H*2 W*2
        return self.conv(x)


class UNetResidualBlock(nn.Module):
    """
    This class defines a residual block for the UNet model.
    It processes the input feature maps with the time embeddings, and adds a residual connection.
    """

    def __init__(self, in_channels: int, out_channels: int, n_time=1280):
        super().__init__()

        self.groupnorm_feature = nn.GroupNorm(32, in_channels)
        self.conv_feature = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1
        )
        self.linear_time = nn.Linear(n_time, out_channels)

        self.groupnorm_merged = nn.GroupNorm(32, out_channels)
        self.conv_merged = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1
        )

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, padding=0
            )

    def forward(self, feature: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        # feature: (B C_in H W)
        # time: (1, 1280)

        residue = feature

        feature = self.groupnorm_feature(feature)
        feature = F.silu(feature)
        # B C_in H W -> B C_out H W
        feature = self.conv_feature(feature)

        time = F.silu(time)
        # 1, 1280 -> 1, C_out
        time = self.linear_time(time)

        # B C_out H W + 1 C_out 1 1 -> B C_out H W
        merged = feature + time.unsqueeze(-1).unsqueeze(-1)
        merged = self.groupnorm_merged(merged)
        merged = F.silu(merged)
        merged = self.conv_merged(merged)

        return merged + self.residual_layer(residue)


class UNetAttentionBlock(nn.Module):
    """
    This class defines an attention block for the UNet model.
    It includes self-attention and cross-attention mechanisms, as well as a feed-forward neural network with a gated linear unit (GeGLU) activation.
    The input feature maps are processed through these layers, and residual connections are added to preserve the original information.
    """

    def __init__(self, n_head: int, n_embd: int, d_context: int = 768):
        super().__init__()

        channels = n_head * n_embd

        self.groupnorm = nn.GroupNorm(32, channels, eps=1e-6)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(n_head, channels, in_proj_bias=False)
        self.layernorm_2 = nn.LayerNorm(channels)
        self.attention_2 = CrossAttention(
            n_head, channels, d_context, in_proj_bias=False
        )
        self.layernorm_3 = nn.LayerNorm(channels)
        self.linear_geglu_1 = nn.Linear(channels, 4 * channels * 2)
        self.linear_geglu_2 = nn.Linear(4 * channels, channels)

        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor, context: torch.LongTensor) -> torch.Tensor:
        # x: B C H W
        # context: B S E

        residue_long = x

        x = self.groupnorm(x)
        x = self.conv_input(x)

        b, c, h, w = x.shape

        # B C H W -> B C H*W
        x = x.view(b, c, h * w)
        # B C H*W -> B H*W C
        x = x.transpose(-1, -2)

        # Normalization + Self-attention with residual connection
        residue_short = x
        x = self.layernorm_1(x)
        x = self.attention_1(x)

        x += residue_short

        # Normalization + Cross-attention with residual connection
        residue_short = x
        x = self.layernorm_2(x)
        x = self.attention_2(x, context)

        x += residue_short

        # Normalization + Feed-forward with GeGLU with residual connection
        residue_short = x

        x = self.layernorm_3(x)
        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1)
        x = x * F.gelu(gate)

        x = self.linear_geglu_2(x)

        x += residue_short

        # B H*W C -> B C H*W
        x = x.transpose(-1, -2)
        # B C H*W -> B C H W
        x = x.view((b, c, h, w))

        return self.conv_output(x) + residue_long


class SwitchSequential(nn.Sequential):
    """
    This class extends nn.Sequential to handle different types of layers in the UNet model.
    It switches between different forward methods based on the layer type.
    For attention, we pass the latent variable and context
    For residual block, we pass the latent variable and time
    For all other layer types, we pass only the latent variable
    """

    def forward(
        self, x: torch.Tensor, context: torch.LongTensor, time: torch.Tensor
    ) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, UNetAttentionBlock):
                x = layer(x, context)
            elif isinstance(layer, UNetResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)

        return x


class UNet(nn.Module):
    """
    This class defines the UNet model used in the Stable Diffusion framework.
    It consists of an encoder and a decoder with skip connections between corresponding layers in the encoder and decoder.

    The encoder progressively reduces the spatial dimensions of the input while increasing the number of feature channels.
    The decoder progressively increases the spatial dimensions of the feature maps while reducing the number of feature channels.

    The UNet model also includes attention and residual blocks to enhance the feature representation and capture long-range dependencies.
    """

    def __init__(self):
        super().__init__()

        self.encoders = nn.ModuleList(
            [
                # ---------------------------------------------------- #
                # B 4 H/8 W/8 -> B 320 H/8 W/8
                SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),
                # ---------------------------------------------------- #
                # B 320 H/8 W/8 -> B 320 H/8 W/8
                SwitchSequential(
                    UNetResidualBlock(320, 320), UNetAttentionBlock(8, 40)
                ),
                SwitchSequential(
                    UNetResidualBlock(320, 320), UNetAttentionBlock(8, 40)
                ),
                # ---------------------------------------------------- #
                # B 320 H/8 W/8 -> B 320 H/16 W/16
                SwitchSequential(
                    nn.Conv2d(320, 320, kernel_size=3, padding=1, stride=2)
                ),
                # ---------------------------------------------------- #
                # B 320 H/16 W/16 -> B 640 H/16 W/16
                SwitchSequential(
                    UNetResidualBlock(320, 640), UNetAttentionBlock(8, 80)
                ),
                SwitchSequential(
                    UNetResidualBlock(640, 640), UNetAttentionBlock(8, 80)
                ),
                # ---------------------------------------------------- #
                # B 640 H/16 W/16 -> B 640 H/32 W/32
                SwitchSequential(
                    nn.Conv2d(640, 640, kernel_size=3, padding=1, stride=2)
                ),
                # ---------------------------------------------------- #
                # B 640 H/32 W/32 -> B 1280 H/32 W/32
                SwitchSequential(
                    UNetResidualBlock(640, 1280), UNetAttentionBlock(8, 160)
                ),
                SwitchSequential(
                    UNetResidualBlock(1280, 1280), UNetAttentionBlock(8, 160)
                ),
                # ---------------------------------------------------- #
                # B 1280 H/32 W/32 -> B 1280 H/64 W/64
                SwitchSequential(
                    nn.Conv2d(1280, 1280, kernel_size=3, padding=1, stride=2)
                ),
                SwitchSequential(UNetResidualBlock(1280, 1280)),
                SwitchSequential(UNetResidualBlock(1280, 1280)),
            ]
        )

        self.bottleneck = SwitchSequential(
            UNetResidualBlock(1280, 1280),
            UNetAttentionBlock(8, 160),
            UNetResidualBlock(1280, 1280),
        )

        self.decoders = nn.ModuleList(
            [
                # ---------------------------------------------------- #
                # B 2560 H/64 W/64 -> B 1280 H/64 W/64
                SwitchSequential(UNetResidualBlock(2560, 1280)),
                SwitchSequential(UNetResidualBlock(2560, 1280)),
                # ---------------------------------------------------- #
                # B 2560 H/64 W/64 -> B 1280 H/32 W/32
                SwitchSequential(UNetResidualBlock(2560, 1280), UNetUpsample(1280)),
                # ---------------------------------------------------- #
                # B 2560 H/32 W/32 -> B 1280 H/32 W/32
                SwitchSequential(
                    UNetResidualBlock(2560, 1280), UNetAttentionBlock(8, 160)
                ),
                SwitchSequential(
                    UNetResidualBlock(2560, 1280), UNetAttentionBlock(8, 160)
                ),
                # ---------------------------------------------------- #
                # B 1920 H/32 W/32 -> B 1280 H/16 W/16
                SwitchSequential(
                    UNetResidualBlock(1920, 1280),
                    UNetAttentionBlock(8, 160),
                    UNetUpsample(1280),
                ),
                # ---------------------------------------------------- #
                # B 1920 H/16 W/16 -> B 640 H/16 W/16
                SwitchSequential(
                    UNetResidualBlock(1920, 640), UNetAttentionBlock(8, 80)
                ),
                # ---------------------------------------------------- #
                # B 1280 H/16 W/16 -> B 640 H/16 W/16
                SwitchSequential(
                    UNetResidualBlock(1280, 640), UNetAttentionBlock(8, 80)
                ),
                # ---------------------------------------------------- #
                # B 960 H/16 W/16 -> B 640 H/8 W/8
                SwitchSequential(
                    UNetResidualBlock(960, 640),
                    UNetAttentionBlock(8, 80),
                    UNetUpsample(640),
                ),
                # ---------------------------------------------------- #
                # B 960 H/8 W/8 -> B 320 H/8 W/8
                SwitchSequential(
                    UNetResidualBlock(960, 320), UNetAttentionBlock(8, 40)
                ),
                # ---------------------------------------------------- #
                # B 640 H/8 W/8 -> B 320 H/8 W/8
                SwitchSequential(
                    UNetResidualBlock(640, 320), UNetAttentionBlock(8, 40)
                ),
                SwitchSequential(
                    UNetResidualBlock(640, 320), UNetAttentionBlock(8, 40)
                ),
            ]
        )

    def forward(
        self, x: torch.Tensor, context: torch.LongTensor, time: torch.Tensor
    ) -> torch.Tensor:
        # x: B 4 H/8 W/8
        # context: B S E
        # time: (1, 1280)

        residual_connections = []

        for layer in self.encoders:
            x = layer(x, context, time)
            residual_connections.append(x)

        x = self.bottleneck(x, context, time)

        for layer in self.decoders:
            x = torch.cat((x, residual_connections.pop()), dim=1)
            x = layer(x, context, time)

        return x


class UNetOutput(nn.Module):
    """
    This class defines the output layer for the UNet model.
    It includes a group normalization layer followed by a convolutional layer.
    The group normalization layer normalizes the input feature maps, and the convolutional layer reduces the number of channels to the desired output channels.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.groupnorm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B 320 H/8 W/8

        x = self.groupnorm(x)
        x = F.silu(x)

        # B 320 H/8 W/8 -> B 4 H/8 W/8
        x = self.conv(x)

        return x
