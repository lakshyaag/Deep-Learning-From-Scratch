import einops
import torch
from rich import print
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, groups=8, debug=False):
        super(ConvBlock, self).__init__()

        self.norm = nn.GroupNorm(groups, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.act = nn.SiLU()

        self.debug = debug

    def _debug_print(self, tensor, name):
        if self.debug:
            print(f"{name}: {tensor.shape}")

    def forward(self, x):
        self._debug_print(x, "[ConvBlock] Input")
        x = self.norm(x)
        self._debug_print(x, "[ConvBlock] Norm")
        x = self.act(x)
        self._debug_print(x, "[ConvBlock] SiLU")
        x = self.conv(x)
        self._debug_print(x, "[ConvBlock] Output")
        return x


class ResNetBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        *,
        d_time_embd=None,
        n_groups=8,
        debug=False,
    ):
        super(ResNetBlock, self).__init__()

        self.block1 = ConvBlock(in_channels, out_channels, groups=n_groups)
        self.block2 = ConvBlock(out_channels, out_channels, groups=n_groups)
        self.residual_conv = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.time_projection = (
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(d_time_embd, out_channels),
            )
            if d_time_embd is not None
            else None
        )

        self.debug = debug

    def _debug_print(self, tensor, name):
        if self.debug:
            print(f"{name}: {tensor.shape}")

    def forward(self, x, t=None):
        input = x
        self._debug_print(input, "[ResNetBlock] Input")

        h = self.block1(x)
        self._debug_print(h, "[ResNetBlock] Block1")

        if t is not None:
            time_embd = self.time_projection(t)
            self._debug_print(time_embd, "[ResNetBlock] TimeProjection")
            x = h + time_embd[:, :, None, None]
        else:
            x = h

        x = self.block2(x)
        self._debug_print(x, "[ResNetBlock] Block2")

        return x + self.residual_conv(input)


class SelfAttentionBlock(nn.Module):
    def __init__(self, n_heads, d_embd, debug=False):
        super(SelfAttentionBlock, self).__init__()

        self.n_heads = n_heads
        self.d_embd = d_embd
        self.d_head = d_embd // n_heads
        self.scale = self.d_head**-0.5

        self.qkv = nn.Linear(d_embd, d_embd * 3)
        self.proj = nn.Linear(d_embd, d_embd)

        self.debug = debug

    def _debug_print(self, tensor, name):
        if self.debug:
            print(f"{name}: {tensor.shape}")

    def forward(self, x, causal_mask=None):
        self._debug_print(x, "[SelfAttentionBlock] Input")

        qkv = self.qkv(x).chunk(3, dim=-1)

        q, k, v = map(
            lambda t: einops.rearrange(t, "b n (h d) -> b h n d", h=self.n_heads),
            qkv,
        )

        self._debug_print(q, "[SelfAttentionBlock] Q")
        self._debug_print(k, "[SelfAttentionBlock] K")
        self._debug_print(v, "[SelfAttentionBlock] V")

        dots = torch.einsum("b h i d, b h j d -> b h i j", q, k)

        if causal_mask:
            mask = torch.ones_like(dots, dtype=torch.bool).triu(1)
            dots.masked_fill_(mask, -torch.inf)

        dots *= self.scale
        attn = dots.softmax(dim=-1)
        self._debug_print(attn, "[SelfAttentionBlock] Attention")

        out = torch.einsum("b h i j, b h j d -> b h i d", attn, v)
        out = einops.rearrange(out, "b h n d -> b n (h d)")
        self._debug_print(out, "[SelfAttentionBlock] Out")

        proj = self.proj(out)
        self._debug_print(proj, "[SelfAttentionBlock] Projection")

        return proj


class CrossAttentionBlock(nn.Module):
    def __init__(self, n_heads, d_embd, d_context, debug=False):
        super(CrossAttentionBlock, self).__init__()

        self.n_heads = n_heads
        self.d_embd = d_embd
        self.d_head = d_embd // n_heads
        self.scale = self.d_head**-0.5

        self.w_q = nn.Linear(d_embd, d_embd)
        self.w_k = nn.Linear(d_context, d_embd)
        self.w_v = nn.Linear(d_context, d_embd)

        self.proj = nn.Linear(d_embd, d_embd)

        self.debug = debug

    def _debug_print(self, tensor, name):
        if self.debug:
            print(f"{name}: {tensor.shape}")

    def forward(self, x, y):
        self._debug_print(x, "[CrossAttentionBlock] Input")
        self._debug_print(y, "[CrossAttentionBlock] Context")

        q = self.w_q(x)
        k = self.w_k(y)
        v = self.w_v(y)

        q, k, v = map(
            lambda t: einops.rearrange(t, "b n (h d) -> b h n d", h=self.n_heads),
            (q, k, v),
        )

        self._debug_print(q, "[CrossAttentionBlock] Q")
        self._debug_print(k, "[CrossAttentionBlock] K")
        self._debug_print(v, "[CrossAttentionBlock] V")

        dots = torch.einsum("b h i d, b h j d -> b h i j", q, k)

        dots *= self.scale
        attn = dots.softmax(dim=-1)
        self._debug_print(attn, "[CrossAttentionBlock] Attention")

        out = torch.einsum("b h i j, b h j d -> b h i d", attn, v)
        out = einops.rearrange(out, "b h n d -> b n (h d)")
        self._debug_print(out, "[CrossAttentionBlock] Out")

        proj = self.proj(out)

        self._debug_print(x, "[CrossAttentionBlock] Output")

        return proj


class VAEAttentionBlock(nn.Module):
    def __init__(self, n_heads, d_embd, n_groups=32, debug=False):
        super(VAEAttentionBlock, self).__init__()

        self.norm = nn.GroupNorm(n_groups, d_embd)
        self.attn = SelfAttentionBlock(n_heads, d_embd)

        self.debug = debug

    def _debug_print(self, tensor, name):
        if self.debug:
            print(f"{name}: {tensor.shape}")

    def forward(self, x):
        inp = x

        b, c, h, w = x.shape
        self._debug_print(inp, "[VAEAttentionBlock] Input")

        x = self.norm(x)

        x = einops.rearrange(x, "b c h w -> b (h w) c")

        x = self.attn(x)

        x = einops.rearrange(x, "b (h w) c -> b c h w", h=h, w=w)

        x += inp

        self._debug_print(x, "[VAEAttentionBlock] Output")

        return x


class VAEEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        latent_size=4,
        base_channels=128,
        n_groups=32,
        n_attn_heads=1,
        debug=False,
    ):
        super(VAEEncoder, self).__init__()

        self.scaling_factor = 0.18215
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        self.block1 = nn.ModuleList(
            [
                ResNetBlock(base_channels, base_channels, n_groups=n_groups),
                ResNetBlock(base_channels, base_channels, n_groups=n_groups),
                nn.Conv2d(
                    base_channels, base_channels, kernel_size=3, stride=2, padding=0
                ),
            ]
        )

        self.block2 = nn.ModuleList(
            [
                ResNetBlock(base_channels, base_channels * 2, n_groups=n_groups),
                ResNetBlock(base_channels * 2, base_channels * 2, n_groups=n_groups),
                nn.Conv2d(
                    base_channels * 2,
                    base_channels * 2,
                    kernel_size=3,
                    stride=2,
                    padding=0,
                ),
            ]
        )

        self.block3 = nn.ModuleList(
            [
                ResNetBlock(base_channels * 2, base_channels * 4, n_groups=n_groups),
                ResNetBlock(base_channels * 4, base_channels * 4, n_groups=n_groups),
                nn.Conv2d(
                    base_channels * 4,
                    base_channels * 4,
                    kernel_size=3,
                    stride=2,
                    padding=0,
                ),
            ]
        )

        self.mid = nn.ModuleList(
            [
                ResNetBlock(base_channels * 4, base_channels * 4, n_groups=n_groups),
                ResNetBlock(base_channels * 4, base_channels * 4, n_groups=n_groups),
                ResNetBlock(base_channels * 4, base_channels * 4, n_groups=n_groups),
                VAEAttentionBlock(
                    n_attn_heads,
                    base_channels * 4,
                    n_groups=n_groups,
                ),
            ]
        )

        self.final = nn.ModuleList(
            [
                ResNetBlock(base_channels * 4, base_channels * 4, n_groups=n_groups),
                nn.GroupNorm(n_groups, base_channels * 4),
                nn.SiLU(),
                nn.Conv2d(base_channels * 4, latent_size, kernel_size=3, padding=1),
                nn.Conv2d(latent_size, latent_size * 2, kernel_size=1, padding=0),
            ]
        )

        self.blocks = nn.ModuleList(
            [
                self.block1,
                self.block2,
                self.block3,
                self.mid,
                self.final,
            ]
        )

        self.debug = debug

    def _debug_print(self, tensor, name):
        if self.debug:
            print(f"{name}: {tensor.shape}")

    def forward(self, x, noise):
        self._debug_print(x, "[VAEEncoder] Input")

        x = self.init_conv(x)
        self._debug_print(x, "[VAEEncoder] Init Conv")

        for ix, block in enumerate(self.blocks):
            for layer in block:
                if getattr(layer, "stride", None) == (2, 2):
                    x = F.pad(x, (0, 1, 0, 1))

                x = layer(x)

            self._debug_print(x, f"[VAEEncoder] Block {ix}")

        mean, log_variance = x.chunk(2, dim=1)
        log_variance = log_variance.clamp(-30, 20)
        variance = log_variance.exp()
        stddev = variance.sqrt()

        # Normalize the encoded tensor
        z = mean + stddev * noise
        self._debug_print(z, "[VAEEncoder] Z")

        z *= self.scaling_factor

        self._debug_print(z, "[VAEEncoder] Output")
        return z


class VAEDecoder(nn.Module):
    def __init__(
        self,
        latent_size=4,
        base_channels=128,
        in_channels=3,
        n_groups=32,
        n_attn_heads=1,
        debug=False,
    ):
        super(VAEDecoder, self).__init__()

        self.scaling_factor = 0.18215

        self.init_conv = nn.Conv2d(latent_size, latent_size, kernel_size=1, padding=0)

        self.initial = nn.ModuleList(
            [
                nn.Conv2d(latent_size, latent_size, kernel_size=1, padding=0),
                nn.Conv2d(latent_size, base_channels * 4, kernel_size=3, padding=1),
                ResNetBlock(base_channels * 4, base_channels * 4, n_groups=n_groups),
            ]
        )

        self.mid = nn.ModuleList(
            [
                VAEAttentionBlock(
                    n_attn_heads,
                    base_channels * 4,
                    n_groups=n_groups,
                ),
                ResNetBlock(base_channels * 4, base_channels * 4, n_groups=n_groups),
                ResNetBlock(base_channels * 4, base_channels * 4, n_groups=n_groups),
                ResNetBlock(base_channels * 4, base_channels * 4, n_groups=n_groups),
                ResNetBlock(base_channels * 4, base_channels * 4, n_groups=n_groups),
            ]
        )

        self.block1 = nn.ModuleList(
            [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(
                    base_channels * 4,
                    base_channels * 4,
                    kernel_size=3,
                    padding=1,
                ),
                ResNetBlock(base_channels * 4, base_channels * 4, n_groups=n_groups),
                ResNetBlock(base_channels * 4, base_channels * 4, n_groups=n_groups),
                ResNetBlock(base_channels * 4, base_channels * 4, n_groups=n_groups),
            ]
        )

        self.block2 = nn.ModuleList(
            [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(
                    base_channels * 4,
                    base_channels * 4,
                    kernel_size=3,
                    padding=1,
                ),
                ResNetBlock(base_channels * 4, base_channels * 2, n_groups=n_groups),
                ResNetBlock(base_channels * 2, base_channels * 2, n_groups=n_groups),
                ResNetBlock(base_channels * 2, base_channels * 2, n_groups=n_groups),
            ]
        )

        self.block3 = nn.ModuleList(
            [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(
                    base_channels * 2,
                    base_channels * 2,
                    kernel_size=3,
                    padding=1,
                ),
                ResNetBlock(base_channels * 2, base_channels, n_groups=n_groups),
                ResNetBlock(base_channels, base_channels, n_groups=n_groups),
                ResNetBlock(base_channels, base_channels, n_groups=n_groups),
            ]
        )

        self.final = nn.ModuleList(
            [
                nn.GroupNorm(n_groups, base_channels),
                nn.SiLU(),
                nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1),
            ]
        )

        self.blocks = nn.ModuleList(
            [
                self.initial,
                self.mid,
                self.block1,
                self.block2,
                self.block3,
                self.final,
            ]
        )

        self.debug = debug

    def _debug_print(self, tensor, name):
        if self.debug:
            print(f"{name}: {tensor.shape}")

    def forward(self, x):
        self._debug_print(x, "[VAEDecoder] Input")

        x /= self.scaling_factor

        x = self.init_conv(x)
        self._debug_print(x, "[VAEDecoder] Init Conv")

        for ix, block in enumerate(self.blocks):
            for layer in block:
                x = layer(x)

            self._debug_print(x, f"[VAEDecoder] Block {ix}")

        self._debug_print(x, "[VAEDecoder] Output")
        return x


class TimeEmbedding(nn.Module):
    def __init__(self, d_time_embd):
        super(TimeEmbedding, self).__init__()

        self.dimension = d_time_embd
        self.time_projection = nn.Sequential(
            nn.Linear(d_time_embd, d_time_embd * 4),
            nn.SiLU(),
            nn.Linear(d_time_embd * 4, d_time_embd * 4),
        )

    def forward(self, t):
        return self.time_projection(t)


class UnetAttentionBlock(nn.Module):
    def __init__(self, n_heads, d_embd, d_context, n_groups=32, debug=False):
        super(UnetAttentionBlock, self).__init__()

        self.n_heads = n_heads
        self.d_embd = d_embd
        self.d_head = d_embd // n_heads

        self.norm = nn.GroupNorm(n_groups, d_embd)
        self.init_conv = nn.Conv2d(d_embd, d_embd, kernel_size=1, padding=0)

        self.ln1 = nn.LayerNorm(d_embd)
        self.sa = SelfAttentionBlock(n_heads, d_embd)

        self.ln2 = nn.LayerNorm(d_embd)
        self.ca = CrossAttentionBlock(n_heads, d_embd, d_context)

        self.ln3 = nn.LayerNorm(d_embd)

        self.geglu1 = nn.Linear(d_embd, 4 * d_embd * 2)
        self.geglu2 = nn.Linear(4 * d_embd, d_embd)

        self.out_conv = nn.Conv2d(d_embd, d_embd, kernel_size=1, padding=0)

        self.debug = debug

    def _debug_print(self, tensor, name):
        if self.debug:
            print(f"{name}: {tensor.shape}")

    def forward(self, x, y):
        self._debug_print(x, "[UnetAttentionBlock] Input")
        self._debug_print(y, "[UnetAttentionBlock] Context")

        parent_residual = x

        x = self.norm(x)
        x = self.init_conv(x)

        b, c, h, w = x.shape

        x = einops.rearrange(x, "b c h w -> b (h w) c")

        # Self-Attention
        sa_residual = x

        x = self.ln1(x)
        x = self.sa(x)

        x += sa_residual

        self._debug_print(x, "[UnetAttentionBlock] Self-Attention")

        # Cross-Attention
        ca_residual = x

        x = self.ln2(x)
        x = self.ca(x, y)

        x += ca_residual

        self._debug_print(x, "[UnetAttentionBlock] Cross-Attention")

        # Feed-forward with GeGLU
        ffn_residual = x

        x = self.ln3(x)

        x, gate = self.geglu1(x).chunk(2, dim=-1)

        x = x * F.gelu(gate)

        x = self.geglu2(x)

        x += ffn_residual

        self._debug_print(x, "[UnetAttentionBlock] Feed-Forward")

        # Output
        x = einops.rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        x = self.out_conv(x)
        self._debug_print(x, "[UnetAttentionBlock] Output")

        return x + parent_residual


class Upsample(nn.Module):
    def __init__(self, channels, scale_factor=2):
        super(Upsample, self).__init__()

        self.scale_factor = scale_factor
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode="nearest")
        x = self.conv(x)
        return x


class Switch(nn.Sequential):
    def forward(self, x, y, t):
        for layer in self:
            if isinstance(layer, ResNetBlock):
                x = layer(x, t)
            elif isinstance(layer, UnetAttentionBlock):
                x = layer(x, y)
            else:
                x = layer(x)

        return x


class Unet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        base_channels=320,
        d_context=768,
        d_time_embd=1280,
        n_attn_heads=8,
        n_groups=32,
        debug=False,
    ):
        super(Unet, self).__init__()

        self.encoders = nn.ModuleList(
            [
                # Block 1
                Switch(nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)),
                Switch(
                    ResNetBlock(
                        in_channels=base_channels,
                        out_channels=base_channels,
                        n_groups=n_groups,
                        d_time_embd=d_time_embd,
                    ),
                    UnetAttentionBlock(
                        n_heads=n_attn_heads,
                        d_embd=base_channels,
                        d_context=d_context,
                        n_groups=n_groups,
                    ),
                ),
                Switch(
                    ResNetBlock(
                        in_channels=base_channels,
                        out_channels=base_channels,
                        n_groups=n_groups,
                        d_time_embd=d_time_embd,
                    ),
                    UnetAttentionBlock(
                        n_heads=n_attn_heads,
                        d_embd=base_channels,
                        d_context=d_context,
                        n_groups=n_groups,
                    ),
                ),
                # Block 2
                Switch(
                    nn.Conv2d(
                        base_channels, base_channels, kernel_size=3, stride=2, padding=1
                    )
                ),
                Switch(
                    ResNetBlock(
                        in_channels=base_channels,
                        out_channels=base_channels * 2,
                        n_groups=n_groups,
                        d_time_embd=d_time_embd,
                    ),
                    UnetAttentionBlock(
                        n_heads=n_attn_heads,
                        d_embd=base_channels * 2,
                        d_context=d_context,
                        n_groups=n_groups,
                    ),
                ),
                Switch(
                    ResNetBlock(
                        in_channels=base_channels * 2,
                        out_channels=base_channels * 2,
                        n_groups=n_groups,
                        d_time_embd=d_time_embd,
                    ),
                    UnetAttentionBlock(
                        n_heads=n_attn_heads,
                        d_embd=base_channels * 2,
                        d_context=d_context,
                        n_groups=n_groups,
                    ),
                ),
                # Block 3
                Switch(
                    nn.Conv2d(
                        base_channels * 2,
                        base_channels * 2,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    )
                ),
                Switch(
                    ResNetBlock(
                        in_channels=base_channels * 2,
                        out_channels=base_channels * 4,
                        n_groups=n_groups,
                        d_time_embd=d_time_embd,
                    ),
                    UnetAttentionBlock(
                        n_heads=n_attn_heads,
                        d_embd=base_channels * 4,
                        d_context=d_context,
                        n_groups=n_groups,
                    ),
                ),
                Switch(
                    ResNetBlock(
                        in_channels=base_channels * 4,
                        out_channels=base_channels * 4,
                        n_groups=n_groups,
                        d_time_embd=d_time_embd,
                    ),
                    UnetAttentionBlock(
                        n_heads=n_attn_heads,
                        d_embd=base_channels * 4,
                        d_context=d_context,
                        n_groups=n_groups,
                    ),
                ),
                # Block 4
                Switch(
                    nn.Conv2d(
                        base_channels * 4,
                        base_channels * 4,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    )
                ),
                Switch(
                    ResNetBlock(
                        in_channels=base_channels * 4,
                        out_channels=base_channels * 4,
                        n_groups=n_groups,
                        d_time_embd=d_time_embd,
                    ),
                    UnetAttentionBlock(
                        n_heads=n_attn_heads,
                        d_embd=base_channels * 4,
                        d_context=d_context,
                        n_groups=n_groups,
                    ),
                ),
                Switch(
                    ResNetBlock(
                        in_channels=base_channels * 4,
                        out_channels=base_channels * 4,
                        n_groups=n_groups,
                        d_time_embd=d_time_embd,
                    ),
                    UnetAttentionBlock(
                        n_heads=n_attn_heads,
                        d_embd=base_channels * 4,
                        d_context=d_context,
                        n_groups=n_groups,
                    ),
                ),
            ]
        )

        self.bottleneck = Switch(
            ResNetBlock(
                in_channels=base_channels * 4,
                out_channels=base_channels * 4,
                n_groups=n_groups,
                d_time_embd=d_time_embd,
            ),
            UnetAttentionBlock(
                n_heads=n_attn_heads,
                d_embd=base_channels * 4,
                d_context=d_context,
                n_groups=n_groups,
            ),
            ResNetBlock(
                in_channels=base_channels * 4,
                out_channels=base_channels * 4,
                n_groups=n_groups,
                d_time_embd=d_time_embd,
            ),
        )

        self.decoder = nn.ModuleList(
            [
                Switch(
                    ResNetBlock(
                        2 * base_channels * 4,
                        base_channels * 4,
                        n_groups=n_groups,
                        d_time_embd=d_time_embd,
                    )
                ),
                Switch(
                    ResNetBlock(
                        2 * base_channels * 4,
                        base_channels * 4,
                        n_groups=n_groups,
                        d_time_embd=d_time_embd,
                    )
                ),
                Switch(
                    ResNetBlock(
                        2 * base_channels * 4,
                        base_channels * 4,
                        n_groups=n_groups,
                        d_time_embd=d_time_embd,
                    ),
                    Upsample(base_channels * 4),
                ),
                Switch(
                    ResNetBlock(
                        2 * base_channels * 4,
                        base_channels * 4,
                        n_groups=n_groups,
                        d_time_embd=d_time_embd,
                    ),
                    UnetAttentionBlock(
                        n_heads=n_attn_heads,
                        d_embd=base_channels * 4,
                        d_context=d_context,
                        n_groups=n_groups,
                    ),
                ),
                Switch(
                    ResNetBlock(
                        2 * base_channels * 4,
                        base_channels * 4,
                        n_groups=n_groups,
                        d_time_embd=d_time_embd,
                    ),
                    UnetAttentionBlock(
                        n_heads=n_attn_heads,
                        d_embd=base_channels * 4,
                        d_context=d_context,
                        n_groups=n_groups,
                    ),
                ),
                Switch(
                    ResNetBlock(
                        base_channels * 4 + base_channels * 2,
                        base_channels * 4,
                        n_groups=n_groups,
                        d_time_embd=d_time_embd,
                    ),
                    UnetAttentionBlock(
                        n_heads=n_attn_heads,
                        d_embd=base_channels * 4,
                        d_context=d_context,
                        n_groups=n_groups,
                    ),
                    Upsample(base_channels * 4),
                ),
                Switch(
                    ResNetBlock(
                        base_channels * 4 + base_channels * 2,
                        base_channels * 2,
                        n_groups=n_groups,
                        d_time_embd=d_time_embd,
                    ),
                    UnetAttentionBlock(
                        n_heads=n_attn_heads,
                        d_embd=base_channels * 2,
                        d_context=d_context,
                        n_groups=n_groups,
                    ),
                ),
                Switch(
                    ResNetBlock(
                        2 * base_channels * 2,
                        base_channels * 2,
                        n_groups=n_groups,
                        d_time_embd=d_time_embd,
                    ),
                    UnetAttentionBlock(
                        n_heads=n_attn_heads,
                        d_embd=base_channels * 2,
                        d_context=d_context,
                        n_groups=n_groups,
                    ),
                ),
                Switch(
                    ResNetBlock(
                        base_channels * 2 + base_channels,
                        base_channels * 2,
                        n_groups=n_groups,
                        d_time_embd=d_time_embd,
                    ),
                    UnetAttentionBlock(
                        n_heads=n_attn_heads,
                        d_embd=base_channels * 2,
                        d_context=d_context,
                        n_groups=n_groups,
                    ),
                    Upsample(base_channels * 2),
                ),
                Switch(
                    ResNetBlock(
                        base_channels * 2 + base_channels,
                        base_channels,
                        n_groups=n_groups,
                        d_time_embd=d_time_embd,
                    ),
                    UnetAttentionBlock(
                        n_heads=n_attn_heads,
                        d_embd=base_channels,
                        d_context=d_context,
                        n_groups=n_groups,
                    ),
                ),
                Switch(
                    ResNetBlock(
                        base_channels * 2,
                        base_channels,
                        n_groups=n_groups,
                        d_time_embd=d_time_embd,
                    ),
                    UnetAttentionBlock(
                        n_heads=n_attn_heads,
                        d_embd=base_channels,
                        d_context=d_context,
                        n_groups=n_groups,
                    ),
                ),
                Switch(
                    ResNetBlock(
                        base_channels * 2,
                        base_channels,
                        n_groups=n_groups,
                        d_time_embd=d_time_embd,
                    ),
                    UnetAttentionBlock(
                        n_heads=n_attn_heads,
                        d_embd=base_channels,
                        d_context=d_context,
                        n_groups=n_groups,
                    ),
                ),
            ]
        )

        self.final = nn.Sequential(
            nn.GroupNorm(n_groups, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1),
        )

        self.debug = debug

    def _debug_print(self, tensor, name):
        if self.debug:
            print(f"{name}: {tensor.shape}")

    def forward(self, x, y, t):
        self._debug_print(x, "[Unet] Input")
        self._debug_print(y, "[Unet] Context")
        self._debug_print(t, "[Unet] Time")

        skip_connections = []
        for ix, encoder in enumerate(self.encoders):
            x = encoder(x, y, t)
            self._debug_print(x, f"[Unet] Encoder {ix}")
            skip_connections.append(x)

        x = self.bottleneck(x, y, t)
        self._debug_print(x, "[Unet] Bottleneck")

        for ix, decoder in enumerate(self.decoder):
            x = torch.cat((x, skip_connections.pop()), dim=1)
            x = decoder(x, y, t)

            self._debug_print(x, f"[Unet] Decoder {ix}")

        self._debug_print(x, "[Unet] Decoded")

        x = self.final(x)
        self._debug_print(x, "[Unet] Output")

        return x


class Diffusion(nn.Module):
    def __init__(self, base_channels: int, unet: Unet):
        super(Diffusion, self).__init__()

        self.time_projection = TimeEmbedding(base_channels)
        self.unet = unet


    def forward(self, x, y, t):
        t = self.time_projection(t)
        return self.unet(x, y, t)
