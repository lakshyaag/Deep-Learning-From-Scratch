import einops
import torch
import torch.nn.functional as F
from torch import nn


class TransformerPositionalEmbedding(nn.Module):
    """
    Transformer sinusoidal positional embedding.
    """

    def __init__(self, dimension, max_timesteps=1000):
        super(TransformerPositionalEmbedding, self).__init__()

        assert dimension % 2 == 0, "Embedding dimension must be even"
        self.dimension = dimension

        self.pos_embd_matrix = torch.zeros(max_timesteps, dimension)
        even_indices = torch.arange(0, self.dimension, 2)

        log_term = torch.log(torch.tensor(10000.0)) / self.dimension
        div_term = torch.exp(even_indices * -log_term)

        timesteps = torch.arange(max_timesteps).unsqueeze(1)

        self.pos_embd_matrix[:, 0::2] = torch.sin(timesteps * div_term)
        self.pos_embd_matrix[:, 1::2] = torch.cos(timesteps * div_term)

    def forward(self, timestep):
        return self.pos_embd_matrix.to(timestep.device)[timestep]


class ConvBlock(nn.Module):
    """
    Convolutional block with GroupNorm and SiLU activation.
    """

    def __init__(self, in_channels, out_channels, groups=8, debug=False):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(groups, out_channels)
        self.act = nn.SiLU()

        self.debug = debug

    def _debug_print(self, tensor, name):
        if self.debug:
            print(f"{name}: {tensor.shape}")

    def forward(self, x):
        self._debug_print(x, "[ConvBlock] Input")
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        self._debug_print(x, "[ConvBlock] Output")
        return x


class ResNetBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        *,
        time_embd_channels=None,
        n_groups=8,
        debug=False,
    ):
        """
        ResNet block with 2 convolutional layers and a residual connection.
        """
        super(ResNetBlock, self).__init__()

        self.time_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embd_channels, out_channels) if time_embd_channels else None,
        )

        self.block1 = ConvBlock(in_channels, out_channels, groups=n_groups)
        self.block2 = ConvBlock(out_channels, out_channels, groups=n_groups)
        self.residual_conv = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

        self.debug = debug

    def _debug_print(self, tensor, name):
        if self.debug:
            print(f"{name}: {tensor.shape}")

    def forward(self, x, time_embd=None):
        input = x
        self._debug_print(input, "[ResNetBlock] Input")

        h = self.block1(x)
        self._debug_print(h, "[ResNetBlock] Block1")

        # Add time embedding after the first block
        time_embd = self.time_projection(time_embd)
        time_embd = time_embd[:, :, None, None]
        x = h + time_embd

        self._debug_print(x, "[ResNetBlock] Time projection")

        x = self.block2(x)
        self._debug_print(x, "[ResNetBlock] Block2")

        return x + self.residual_conv(input)


class ConvDownBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        n_layers,
        time_embd_channels,
        n_groups,
        downsample=True,
        debug=False,
    ):
        """
        Convolutional block in the downsampling path of the U-Net, made of n_layers of ResNet blocks.
        """
        super(ConvDownBlock, self).__init__()

        self.resnet_blocks = nn.ModuleList([])

        for i in range(n_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnet_block = ResNetBlock(
                in_channels,
                out_channels,
                time_embd_channels=time_embd_channels,
                n_groups=n_groups,
            )
            self.resnet_blocks.append(resnet_block)

        self.downsample = (
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            if downsample
            else None
        )
        self.debug = debug

    def _debug_print(self, tensor, name):
        if self.debug:
            print(f"{name}: {tensor.shape}")

    def forward(self, x, time_embd):
        self._debug_print(x, "[ConvDownBlock] Input")

        for ix, block in enumerate(self.resnet_blocks):
            x = block(x, time_embd)
            self._debug_print(x, f"[ConvDownBlock] Block {ix}")

        if self.downsample:
            x = self.downsample(x)
            self._debug_print(x, "[ConvDownBlock] Downsample")

        return x


class ConvUpBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        n_layers,
        time_embd_channels,
        n_groups,
        upsample=True,
        scale_factor=2.0,
        debug=False,
    ):
        """
        Convolutional block in the upsampling path of the U-Net, made of n_layers of ResNet blocks.
        """
        super(ConvUpBlock, self).__init__()

        self.resnet_blocks = nn.ModuleList([])

        for i in range(n_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnet_block = ResNetBlock(
                in_channels,
                out_channels,
                time_embd_channels=time_embd_channels,
                n_groups=n_groups,
            )
            self.resnet_blocks.append(resnet_block)

        self.upsample = (
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            if upsample
            else None
        )
        self.scale_factor = scale_factor
        self.debug = debug

    def _debug_print(self, tensor, name):
        if self.debug:
            print(f"{name}: {tensor.shape}")

    def forward(self, x, time_embd):
        self._debug_print(x, "[ConvUpBlock] Input")

        for ix, block in enumerate(self.resnet_blocks):
            x = block(x, time_embd)
            self._debug_print(x, f"[ConvUpBlock] Block {ix}")

        if self.upsample:
            # Upsample using bilinear interpolation, could optionally use transposed convolution instead
            x = F.interpolate(
                x, scale_factor=self.scale_factor, mode="bilinear", align_corners=True
            )
            x = self.upsample(x)
            self._debug_print(x, "[ConvUpBlock] Upsample")

        return x


class SelfAttentionBlock(nn.Module):
    """
    Self-attention block with multi-head attention, as described in Attention is All You Need (Vaswani et al., 2017).
    """

    def __init__(self, n_heads, in_channels, d_embd=256, n_groups=32, debug=False):
        super(SelfAttentionBlock, self).__init__()

        self.n_heads = n_heads
        self.d_embd = d_embd
        self.d_head = d_embd // n_heads
        self.scale = self.d_head**-0.5

        self.qkv = nn.Linear(in_channels, d_embd * 3)
        self.proj = nn.Linear(d_embd, d_embd)
        self.norm = nn.GroupNorm(n_groups, d_embd)

        self.debug = debug

    def _debug_print(self, tensor, name):
        if self.debug:
            print(f"{name}: {tensor.shape}")

    def forward(self, x):
        self._debug_print(x, "[SelfAttentionBlock] Input")

        inp = x

        _, _, h, w = inp.shape

        # Rearrange input tensor to have the same shape as the query, key, and value matrices
        inp = einops.rearrange(inp, "b c h w -> b (h w) c")
        self._debug_print(inp, "[SelfAttentionBlock] Rearrange")

        # Compute query, key, and value matrices
        qkv = self.qkv(inp).chunk(3, dim=-1)

        # Split the query, key, and value matrices into multiple heads
        q, k, v = map(
            lambda t: einops.rearrange(t, "b n (h d) -> b h n d", h=self.n_heads),
            qkv,
        )

        self._debug_print(q, "[SelfAttentionBlock] Q")
        self._debug_print(k, "[SelfAttentionBlock] K")
        self._debug_print(v, "[SelfAttentionBlock] V")

        # Compute the scaled dot-product attention (scale = 1 / sqrt(d_head))
        dots = torch.einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        attn = dots.softmax(dim=-1)
        self._debug_print(attn, "[SelfAttentionBlock] Attention")

        # Apply the attention to the value matrix
        out = torch.einsum("b h i j, b h j d -> b h i d", attn, v)
        out = einops.rearrange(out, "b h n d -> b n (h d)")
        self._debug_print(out, "[SelfAttentionBlock] Out")

        # Project the output back to the original dimension
        proj = self.proj(out)
        proj = einops.rearrange(proj, "b (h w) c -> b c h w", h=h, w=w)
        self._debug_print(proj, "[SelfAttentionBlock] Projection")

        # Apply a residual connection and layer normalization
        x = self.norm(proj + x)
        self._debug_print(x, "[SelfAttentionBlock] Output")
        return x


class AttentionDownBlock(nn.Module):
    """
    Attention block in the downsampling path of the U-Net, made of `n_layers` of ResNet blocks and self-attention blocks.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        n_layers,
        n_heads,
        n_groups,
        time_embd_channels,
        downsample=True,
        debug=False,
    ):
        super(AttentionDownBlock, self).__init__()

        self.resnet_blocks = nn.ModuleList([])
        self.attn_blocks = nn.ModuleList([])

        for i in range(n_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnet_block = ResNetBlock(
                in_channels,
                out_channels,
                time_embd_channels=time_embd_channels,
                n_groups=n_groups,
            )

            attn_block = SelfAttentionBlock(
                n_heads,
                in_channels=out_channels,
                d_embd=out_channels,
                n_groups=n_groups,
            )
            self.resnet_blocks.append(resnet_block)
            self.attn_blocks.append(attn_block)

        self.downsample = (
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            if downsample
            else None
        )

        self.debug = debug

    def _debug_print(self, tensor, name):
        if self.debug:
            print(f"{name}: {tensor.shape}")

    def forward(self, x, time_embd):
        for ix, (resnet_block, attn_block) in enumerate(
            zip(self.resnet_blocks, self.attn_blocks)
        ):
            x = resnet_block(x, time_embd)
            self._debug_print(x, f"[AttentionDownBlock] ResNet Block {ix}")

            x = attn_block(x)
            self._debug_print(x, f"[AttentionDownBlock] Attention Block {ix}")

        if self.downsample:
            x = self.downsample(x)
            self._debug_print(x, "[AttentionDownBlock] Downsample")

        return x


class AttentionUpBlock(nn.Module):
    """
    Attention block in the upsampling path of the U-Net, made of `n_layers` of ResNet blocks and self-attention blocks.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        n_layers,
        n_heads,
        n_groups,
        time_embd_channels,
        upsample=True,
        scale_factor=2.0,
        debug=False,
    ):
        super(AttentionUpBlock, self).__init__()

        self.resnet_blocks = nn.ModuleList([])
        self.attn_blocks = nn.ModuleList([])

        for i in range(n_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnet_block = ResNetBlock(
                in_channels,
                out_channels,
                time_embd_channels=time_embd_channels,
                n_groups=n_groups,
            )

            attn_block = SelfAttentionBlock(
                n_heads,
                in_channels=out_channels,
                d_embd=out_channels,
                n_groups=n_groups,
            )
            self.resnet_blocks.append(resnet_block)
            self.attn_blocks.append(attn_block)

        self.upsample = (
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            if upsample
            else None
        )
        self.scale_factor = scale_factor

        self.debug = debug

    def _debug_print(self, tensor, name):
        if self.debug:
            print(f"{name}: {tensor.shape}")

    def forward(self, x, time_embd):
        for ix, (resnet_block, attn_block) in enumerate(
            zip(self.resnet_blocks, self.attn_blocks)
        ):
            x = resnet_block(x, time_embd)
            self._debug_print(x, f"[AttentionUpBlock] ResNet Block {ix}")

            x = attn_block(x)
            self._debug_print(x, f"[AttentionUpBlock] Attention Block {ix}")

        if self.upsample:
            x = F.interpolate(
                x, scale_factor=self.scale_factor, mode="bilinear", align_corners=True
            )
            x = self.upsample(x)
            self._debug_print(x, "[AttentionUpBlock] Upsample")

        return x


class Unet(nn.Module):
    """
    UNet model with multi-head self-attention.

    Args:
    - image_size (int): Size of the input images.
    - in_channels (int): Number of input channels. Default is 3 for RGB images.
    - base_channels (int): Number of channels in the first convolutional layer
    - n_layers(int): Number of ResNet/Attention blocks in each downsample/upsample block.
    - n_heads (int): Number of attention heads in the self-attention blocks.
    - n_groups (int): Number of groups for GroupNorm.
    - debug (bool): Whether to print debug information. Default is False.
    """

    def __init__(
        self,
        image_size: int = 256,
        in_channels: int = 3,
        base_channels: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        n_groups: int = 32,
        debug: bool = False,
    ):
        super(Unet, self).__init__()

        self.image_size = image_size
        self.base_channels = base_channels

        # Convert input image to base_channels
        self.initial_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.base_channels,
            kernel_size=3,
            stride=1,
            padding="same",
        )
        self.time_embd_channels = self.base_channels * 4

        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_groups = n_groups

        # Positional embedding for the timestep
        self.positional_encoding = nn.Sequential(
            TransformerPositionalEmbedding(self.base_channels),
            nn.Linear(self.base_channels, self.time_embd_channels),
            nn.GELU(),
            nn.Linear(self.time_embd_channels, self.time_embd_channels),
        )

        # Downsample blocks
        self.downsample_blocks = nn.ModuleList(
            [
                # b c_0 h w -> b c_0 h/2 w/2
                ConvDownBlock(
                    in_channels=self.base_channels,
                    out_channels=self.base_channels,
                    n_layers=self.n_layers,
                    n_groups=self.n_groups,
                    time_embd_channels=self.time_embd_channels,
                ),
                # b c_0 h/2 w/2 -> b c_1 h/4 w/4
                AttentionDownBlock(
                    in_channels=self.base_channels,
                    out_channels=self.base_channels * 2,
                    n_layers=self.n_layers,
                    n_heads=self.n_heads,
                    n_groups=self.n_groups,
                    time_embd_channels=self.time_embd_channels,
                ),
                # b c_1 h/4 w/4 -> b c_1 h/8 w/8
                ConvDownBlock(
                    in_channels=self.base_channels * 2,
                    out_channels=self.base_channels * 2,
                    n_layers=self.n_layers,
                    n_groups=self.n_groups,
                    time_embd_channels=self.time_embd_channels,
                ),
                # b c_1 h/8 w/8 -> b c_1 h/16 w/16
                ConvDownBlock(
                    in_channels=self.base_channels * 2,
                    out_channels=self.base_channels * 2,
                    n_layers=self.n_layers,
                    n_groups=self.n_groups,
                    time_embd_channels=self.time_embd_channels,
                ),
            ]
        )

        # b c_1 h/16 w/16 -> b c_1 h/16 w/16
        self.bottleneck = AttentionDownBlock(
            in_channels=self.base_channels * 2,
            out_channels=self.base_channels * 2,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            n_groups=self.n_groups,
            time_embd_channels=self.time_embd_channels,
            downsample=False,
        )

        self.upsample_blocks = nn.ModuleList(
            [
                ConvUpBlock(
                    in_channels=self.base_channels * 2 + self.base_channels * 2,
                    out_channels=self.base_channels * 2,
                    n_layers=self.n_layers,
                    n_groups=self.n_groups,
                    time_embd_channels=self.time_embd_channels,
                ),
                ConvUpBlock(
                    in_channels=self.base_channels * 2 + self.base_channels * 2,
                    out_channels=self.base_channels * 2,
                    n_layers=self.n_layers,
                    n_groups=self.n_groups,
                    time_embd_channels=self.time_embd_channels,
                ),
                AttentionUpBlock(
                    in_channels=self.base_channels * 2 + self.base_channels * 2,
                    out_channels=self.base_channels * 2,
                    n_layers=self.n_layers,
                    n_heads=self.n_heads,
                    n_groups=self.n_groups,
                    time_embd_channels=self.time_embd_channels,
                ),
                ConvUpBlock(
                    in_channels=self.base_channels * 2 + self.base_channels,
                    out_channels=self.base_channels,
                    n_layers=self.n_layers,
                    n_groups=self.n_groups,
                    time_embd_channels=self.time_embd_channels,
                ),
            ]
        )

        self.output_conv = nn.Sequential(
            nn.GroupNorm(num_groups=self.n_groups, num_channels=self.base_channels * 2),
            nn.SiLU(),
            nn.Conv2d(
                in_channels=self.base_channels * 2,
                out_channels=in_channels,
                kernel_size=3,
                padding=1,
            ),
        )

        self.debug = debug

    def _debug_print(self, tensor, name):
        if self.debug:
            print(f"{name}: {tensor.shape}")

    def forward(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        self._debug_print(x, "[Unet] Input")

        t_encoded = self.positional_encoding(timestep)
        self._debug_print(t_encoded, "[Unet] Time Encoding")

        init_x = self.initial_conv(x)
        self._debug_print(init_x, "[Unet] Initial Conv")

        skip_connections = [init_x]

        x = init_x
        for ix, block in enumerate(self.downsample_blocks):
            x = block(x, t_encoded)
            self._debug_print(x, f"[Unet] Downsample Block {ix}")
            skip_connections.append(x)

        skip_connections = skip_connections[::-1]

        x = self.bottleneck(x, t_encoded)
        self._debug_print(x, "[Unet] Bottleneck")

        for ix, (block, skip) in enumerate(zip(self.upsample_blocks, skip_connections)):
            x = torch.cat([x, skip], dim=1)
            self._debug_print(x, f"[Unet] Concatenation {ix}")

            x = block(x, t_encoded)
            self._debug_print(x, f"[Unet] Upsample Block {ix}")

        x = torch.cat([x, skip_connections[-1]], dim=1)
        self._debug_print(x, "[Unet] Final Concatenation")
        x = self.output_conv(x)
        self._debug_print(x, "[Unet] Output")

        return x
