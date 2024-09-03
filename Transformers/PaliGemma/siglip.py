from typing import Tuple

import einops
import torch
from config import SiglipVisionConfig
from torch import nn
from torch.nn import functional as F


class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()

        self.config = config

        self.d_embd = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            padding="valid",
        )

        self.n_patches = (self.image_size // self.patch_size) ** 2
        self.n_positions = self.n_patches

        self.pos_embedding = nn.Embedding(self.n_positions, self.d_embd)

        self.register_buffer(
            "position_ids",
            torch.arange(self.n_positions).expand((1, -1)),
            persistent=False,
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # bchw -> b, n_patches, h // patch_size, w // patch_size
        x = self.patch_embedding(pixel_values)

        # b, n_patches, h // patch_size, w // patch_size -> b, n_patches, d_embd
        x = einops.rearrange(
            x,
            "b c n1 n2 -> b (n1 n2) c",
            n1=x.shape[-2],
            n2=x.shape[-1],
        )

        x += self.pos_embedding(self.position_ids)

        return x


class SiglipAttention(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()

        self.config = config

        self.d_embd = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.d_head = self.d_embd // self.n_heads
        self.scale = self.d_head**-0.5

        self.qkv = nn.Linear(self.d_embd, 3 * self.d_embd)
        self.attn_dropout = config.attention_dropout

        self.proj = nn.Linear(self.d_embd, self.d_embd)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, n_patches, d_embd = hidden_states.shape

        # b, n_patches, d_embd -> b, n_patches, 3 * d_embd -> ([b, n_patches, d_embd], [b, n_patches, d_embd], [b, n_patches, d_embd])
        qkv = self.qkv(hidden_states).chunk(3, dim=-1)

        # b, n_patches, d_embd -> b, n_heads, n_patches, d_head
        q, k, v = map(
            lambda t: einops.rearrange(t, "b n (h d) -> b h n d", h=self.n_heads),
            qkv,
        )

        # q @ k
        # [b, n_heads, n_patches, d_head] @ [b, n_heads, n_patches, d_head] -> [b, n_heads, n_patches, n_patches]
        dots = torch.einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        attn = F.softmax(dots, dim=-1)
        attn = F.dropout(attn, p=self.attn_dropout, training=self.training)

        # attn @ v
        # [b, n_heads, n_patches, n_patches] @ [b, n_heads, n_patches, d_head] -> [b, n_heads, n_patches, d_head]
        out = torch.einsum("b h i j, b h j d -> b h i d", attn, v)
        # b, n_heads, n_patches, d_head -> b, n_patches, n_heads, d_head
        out = einops.rearrange(out, "b h n d -> b n (h d)")

        # b, n_patches, n_heads, d_head -> b, n_patches, d_embd
        out = self.proj(out)

        return out, attn


class SiglipMLP(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()

        self.config = config

        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # b, n_patches, d_embd -> b, n_patches, intermediate_size
        hidden_states = self.fc1(hidden_states)
        hidden_states = F.gelu(hidden_states, approximate="tanh")

        # b, n_patches, intermediate_size -> b, n_patches, d_embd
        hidden_states = self.fc2(hidden_states)

        return hidden_states


class SiglipEncoderLayer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()

        self.config = config

        self.d_embd = config.hidden_size

        self.ln_1 = nn.LayerNorm(self.d_embd, eps=config.layer_norm_eps)
        self.self_attn = SiglipAttention(config)

        self.ln_2 = nn.LayerNorm(self.d_embd, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # attn_residual: b, n_patches, d_embd
        attn_residual = hidden_states

        hidden_states = self.ln_1(hidden_states)
        # b, n_patches, d_embd -> b, n_patches, d_embd, ?
        hidden_states, _ = self.self_attn(hidden_states)

        hidden_states = hidden_states + attn_residual

        # mlp_residual: b, n_patches, d_embd
        mlp_residual = hidden_states

        hidden_states = self.ln_2(hidden_states)

        # b, n_patches, d_embd -> b, n_patches, d_embd
        hidden_states = self.mlp(hidden_states)

        hidden_states = hidden_states + mlp_residual

        return hidden_states


class SiglipEncoder(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()

        self.config = config

        self.layers = nn.ModuleList(
            [SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states)

        return hidden_states


class SiglipVisionTransformer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config

        d_embd = config.hidden_size

        self.embedding = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)

        self.post_ln = nn.LayerNorm(d_embd, eps=config.layer_norm_eps)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # bchw -> b, n_patches, d_embd
        hidden_states = self.embedding(pixel_values)

        # b, n_patches, d_embd -> b, n_patches, d_embd
        last_hidden_state = self.encoder(hidden_states)

        last_hidden_state = self.post_ln(last_hidden_state)

        return last_hidden_state


class SiglipVisionModel(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()

        self.config = config
        self.vision_model = SiglipVisionTransformer(config)

    def forward(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor]:
        # bchw -> b, n_patches, d_embd
        return self.vision_model(pixel_values=pixel_values)
