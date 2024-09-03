from typing import Optional, Tuple

import einops
import torch
from config import GemmaLMConfig
from processor import KVCache
from torch import nn
from torch.nn import functional as F
from utils import apply_rotary_pos_emb, repeat_kv


class GemmaRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int, base: int):
        super().__init__()

        self.dim = dim

        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim)
        )

        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        seq_len: Optional[int] = None,
    ):
        self.inv_freq.to(x.device)

        inv_freq_expanded = einops.repeat(
            self.inv_freq.float(), "d -> b d 1", b=position_ids.shape[0]
        )
        position_ids_expanded = einops.repeat(position_ids, "b s -> b 1 s")

        device_type = x.device

        device_type = (
            device_type
            if isinstance(device_type, str) and device_type != "mps"
            else "cpu"
        )

        with torch.autocast(device_type=device_type, enabled=False):
            freqs = torch.einsum(
                "bdi,bis->bsd",
                inv_freq_expanded.float(),
                position_ids_expanded.float(),
            )

            embd = torch.cat((freqs, freqs), dim=-1)

            cos = torch.cos(embd)
            sin = torch.sin(embd)

        return cos.to(x.dtype), sin.to(x.dtype)


class GemmaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()

        self.eps = eps

        self.scale = nn.Parameter(torch.zeros(dim))

    def _norm(self, x: torch.Tensor):
        return x * torch.rsqrt(
            torch.mean(torch.pow(x, 2), dim=-1, keepdim=True) + self.eps
        )

    def forward(self, x: torch.Tensor):
        output = self._norm(x.float())
        output = output * (1 + self.scale.float())

        return output


class GemmaMLP(nn.Module):
    def __init__(self, config: GemmaLMConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.fc1 = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.fc2 = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.gelu(self.gate_proj(x), approximate="tanh")
        x = gate * self.fc1(x)
        x = self.fc2(x)

        return x


class GemmaAttention(nn.Module):
    def __init__(self, config: GemmaLMConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.attn_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.d_head = config.d_head
        self.num_kv_heads = config.num_kv_heads
        self.num_kv_groups = self.num_attention_heads // self.num_kv_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        assert (
            self.hidden_size % self.num_attention_heads == 0
        ), "Hidden size must be divisible by the number of attention heads"

        self.w_q = nn.Linear(
            self.hidden_size,
            self.num_attention_heads * self.d_head,
            bias=config.attention_bias,
        )
        self.w_k = nn.Linear(
            self.hidden_size,
            self.num_kv_heads * self.d_head,
            bias=config.attention_bias,
        )
        self.w_v = nn.Linear(
            self.hidden_size,
            self.num_kv_heads * self.d_head,
            bias=config.attention_bias,
        )
        self.w_o = nn.Linear(
            self.num_attention_heads * self.d_head,
            self.hidden_size,
            bias=config.attention_bias,
        )
        self.scale = self.d_head**-0.5

        self.rotary_pos_emb = GemmaRotaryEmbedding(
            self.d_head,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
        **kwargs,
    ):
        b, s, _ = x.shape

        q = einops.rearrange(
            self.w_q(x), "b s (n d) -> b n s d", n=self.num_attention_heads
        )
        k = einops.rearrange(self.w_k(x), "b s (n d) -> b n s d", n=self.num_kv_heads)
        v = einops.rearrange(self.w_v(x), "b s (n d) -> b n s d", n=self.num_kv_heads)

        cos, sin = self.rotary_pos_emb(v, position_ids, seq_len=None)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if kv_cache is not None:
            k, v = kv_cache.update(k, v, self.layer_idx)

        k = repeat_kv(k, self.num_kv_groups)
        v = repeat_kv(v, self.num_kv_groups)

        attn_weights = torch.einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_weights = F.dropout(
            attn_weights, p=self.attn_dropout, training=self.training
        )

        out = torch.einsum("b h i j, b h j d -> b h i d", attn_weights, v)
        out = einops.rearrange(out, "b h n d -> b n (h d)")

        out = self.w_o(out)

        return out, attn_weights


class GemmaDecoderLayer(nn.Module):
    def __init__(self, config: GemmaLMConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size

        self.self_attn = GemmaAttention(config, layer_idx)

        self.mlp = GemmaMLP(config)
        self.input_ln = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn_ln = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple:
        residual = x

        # Attention block
        x = self.input_ln(x)

        x, _ = self.self_attn(
            x,
            position_ids=position_ids,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
        )

        x = residual + x

        # MLP block
        residual = x

        x = self.attn_ln(x)
        x = self.mlp(x)

        x = residual + x

        return x


class GemmaModel(nn.Module):
    def __init__(self, config: GemmaLMConfig):
        super().__init__()
        self.config = config

        self.padding_idx = config.pad_token_id
        self.d_vocab = config.d_vocab

        self.embed_tokens = nn.Embedding(
            config.d_vocab, config.hidden_size, padding_idx=self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                GemmaDecoderLayer(config, l_idx)
                for l_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(
        self,
        input_embd: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> torch.Tensor:
        h = input_embd

        normalizer = torch.tensor(self.config.hidden_size**0.5, dtype=h.dtype)
        h = h * normalizer

        for decoder_layer in self.layers:
            h = decoder_layer(
                h,
                position_ids=position_ids,
                attention_mask=attention_mask,
                kv_cache=kv_cache,
            )

        h = self.norm(h)

        return h


class GemmaLM(nn.Module):
    def __init__(self, config: GemmaLMConfig):
        super().__init__()
        self.config = config

        self.model = GemmaModel(config)
        self.d_vocab = config.d_vocab
        self.lm_head = nn.Linear(config.hidden_size, config.d_vocab, bias=False)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def tie_weights(self):
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_embd: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple:
        outputs = self.model(
            input_embd,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_cache=kv_cache,
        )

        # Convert the final contextualized embeddings to logits
        h = outputs
        logits = self.lm_head(h)
        logits = logits.float()

        return_data = {"logits": logits}

        if kv_cache is not None:
            return_data["kv_cache"] = kv_cache

        return return_data
