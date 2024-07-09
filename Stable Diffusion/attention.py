import math

import torch
from torch import nn
from torch.nn import functional as F


class SelfAttention(nn.Module):
    """
    This class implements multi-head self-attention
    """

    def __init__(
        self,
        n_heads: int,
        d_embed: int,
        in_proj_bias: bool = True,
        out_proj_bias: bool = True,
    ):
        super().__init__()

        self.in_proj = nn.Linear(d_embed, d_embed * 3, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)

        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x: torch.Tensor, causal_mask: bool = False) -> torch.Tensor:
        # x: B S C

        input_shape = x.shape
        b, s, c = input_shape

        interim_shape = (b, s, self.n_heads, self.d_head)

        # B S C -(in_proj)> B S C * 3 -(chunk)> B S C, B S C, B S C
        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        # B S C -(view)> B S H C/H -(transpose)> B H S C/H
        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        # B H S C/H x B H C/H S -> B H S S
        attn_score = q @ k.transpose(-1, -2)

        if causal_mask:
            mask = torch.ones_like(attn_score, dtype=torch.bool).triu(1)
            attn_score.masked_fill(mask, -torch.inf)

        attn_score /= math.sqrt(self.d_head)

        attn_score = F.softmax(attn_score, dim=-1)

        # B H S S x B H S C/H -> B H S C/H
        output = attn_score @ v

        # B H S C/H -> B S H C/H
        output = output.transpose(1, 2)

        # B S H C/H -> B S C
        output = output.reshape(input_shape)

        # B S C -> B S C
        output = self.out_proj(output)

        # output: B S C
        return output


class CrossAttention(nn.Module):
    """
    This class implements multi-head cross-attention
    """

    def __init__(
        self,
        n_heads: int,
        d_embed: int,
        d_cross: int,
        in_proj_bias: bool = True,
        out_proj_bias: bool = True,
    ):
        super().__init__()

        self.q_proj = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.k_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.v_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)

        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)

        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x: B S_Q C_Q
        # y: B S_K C_K

        input_shape = x.shape
        b, s_q, c_q = input_shape

        interim_shape = (b, -1, self.n_heads, self.d_head)

        # B S_Q C_Q -> B S_Q C_Q
        q = self.q_proj(x)
        # B S_K C_K -> B S_K C_Q
        k = self.k_proj(y)
        # B S_K C_K -> B S_K C_Q
        v = self.v_proj(y)

        # B S C_Q -(view)> B S H C_Q/H -(transpose)> B H S C_Q/H
        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        # B H S_Q C_Q/H x B H C_Q/H S_K -> B H S_Q S_K
        attn_score = q @ k.transpose(-1, -2)

        attn_score /= math.sqrt(self.d_head)

        attn_score = F.softmax(attn_score, dim=-1)

        # B H S_Q S_K x B H S_K C_Q/H -> B H S_Q C_Q/H
        output = attn_score @ v

        # B H S_Q C_Q/H -> B S_Q H C_Q/H
        output = output.transpose(1, 2).contiguous()

        # B S_Q H C_Q/H -> B S_Q C_Q
        output = output.reshape(input_shape)

        # B S_Q C_Q -> B S_Q C_Q
        output = self.out_proj(output)

        # output: B S_Q C_Q
        return output
