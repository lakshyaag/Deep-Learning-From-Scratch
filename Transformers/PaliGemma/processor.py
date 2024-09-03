from typing import List, Optional, Tuple

import einops
import torch
import torch.nn as nn
from config import PaliGemmaConfig
from gemma import GemmaLM, KVCache, SiglipVisionModel
from PIL import Image
from torchvision import transforms as T
from utils import add_image_tokens_to_prompt, process_images


class PaliGemmaProcessor:
    IMAGE_TOKEN = "<image>"

    def __init__(self, tokenizer, num_image_tokens: int, image_size: int):
        super().__init__()

        self.image_seq_length = num_image_tokens
        self.image_size = image_size

        tokens_to_add = {"additional_special_tokens": [self.IMAGE_TOKEN]}
        tokenizer.add_special_tokens(tokens_to_add)
        EXTRA_TOKENS = [f"<loc{i:04d}>" for i in range(1024)]

        EXTRA_TOKENS += [f"<seg{i:03d}>" for i in range(128)]

        tokenizer.add_tokens(EXTRA_TOKENS)

        self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)

        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False

        self.tokenizer = tokenizer

    def __call__(
        self,
        text: List[str],
        images: List[Image.Image],
        padding: str = "longest",
        truncation: bool = True,
    ):
        pixel_values = process_images(
            images,
            size=(self.image_size, self.image_size),
            resample=T.InterpolationMode.BICUBIC,
            rescale_factor=1 / 255.0,
        )

        pixel_values = torch.stack(pixel_values)

        input_strings = [
            add_image_tokens_to_prompt(
                prefix_prompt=prompt,
                bos_token=self.tokenizer.bos_token,
                image_seq_len=self.image_seq_length,
                image_token=self.IMAGE_TOKEN,
            )
            for prompt in text
        ]

        inputs = self.tokenizer(
            input_strings,
            return_tensors="pt",
            padding=padding,
            truncation=truncation,
        )

        return {"pixel_values": pixel_values, **inputs}


class PaliGemmaMMProjector(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.config = config

        self.linear = nn.Linear(
            config.vision_config.hidden_size,
            config.vision_config.projection_dim,
            bias=True,
        )

        def forward(self, image_embd: torch.Tensor) -> torch.Tensor:
            # (b, n_patches, d_embd) -> (b, n_patches, d_proj)
            h = self.linear(image_embd)
            return h


class PaliGemmaConditional(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.config = config

        self.vision_model = SiglipVisionModel(config.vision_config)

        self.mm_projector = PaliGemmaMMProjector(config)

        self.d_vocab = config.d_vocab

        self.language_model = GemmaLM(config.text_config)

        self.pad_token_id = (
            self.config.pad_token_id if self.config.pad_token_id is not None else -1
        )

    def tie_weights(self):
        # use the same embedding matrix for the language model and the mm projector or not
        self.language_model.tie_weights()

    def _merge_modalities(
        self,
        image_embd: torch.Tensor,
        input_embd: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ):
        b, n_patches, d_vocab = image_embd.shape

        b, s = input_ids.shape

        dtype, device = input_embd.dtype, input_embd.device

        # (b, n_patches, d_vocab) -> (b, n_patches, d_vocab) / sqrt(hidden_size)
        scaled_image_embd = image_embd / (self.config.hidden_size**0.5)

        final_embd = torch.zeros(b, s, d_vocab, dtype=dtype, device=device)

        # Get the masks for each token type
        text_mask = (input_ids != self.config.image_token_index) * (
            input_ids != self.pad_token_id
        )
        image_mask = input_ids == self.config.image_token_index
        pad_mask = input_ids == self.pad_token_id

        # Expand the masks to the embedding dimension
        text_mask, image_mask, pad_mask = map(
            lambda t: einops.repeat(t, pattern="b s -> b s d_vocab", d_vocab=d_vocab),
            [text_mask, image_mask, pad_mask],
        )

        # IF - ELSE logic to replace the zeros in the final embedding with the respective token embeddings
        final_embd = torch.where(text_mask, input_embd, final_embd)
        final_embd = final_embd.masked_scatter(image_mask, scaled_image_embd)

        final_embd = torch.where(pad_mask, torch.zeros_like(final_embd), final_embd)

        # Create the attention mask

        dtype, device = input_embd.dtype, input_embd.device
        min_dtype = torch.finfo(dtype).min

        if kv_cache is None or kv_cache.num_items() == 0:
            # Pre-fill phase - do not mask any tokens
            causal_mask = torch.full(
                (b, s, s), fill_value=0, dtype=dtype, device=device
            )
        else:
            assert s == 1, "Only single token generation is supported for now"

            kv_cache_size = kv_cache.num_items() + s

            causal_mask = torch.full(
                (b, s, kv_cache_size), fill_value=min_dtype, dtype=dtype, device=device
            )

        # (b, s, s) -> (b, n_heads, s, s)
        causal_mask = causal_mask.unsqueeze(1)

        # Positional embeddings for the mask
        if kv_cache is None or kv_cache.num_items() > 0:
            position_ids = attention_mask.cumsum(-1)[:, -1]
            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0)

        else:
            position_ids = (
                (attention_mask.cumsum(-1))
                .masked_fill_((attention_mask == 0), 1)
                .to(device)
            )

        return final_embd, causal_mask, position_ids

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple:
        # input_ids: (b, s)
        # pixel_values: (b, c, h, w)

        # (b, s) -> (b, s, d_vocab)
        input_embd = self.language_model.get_input_embeddings()(input_ids)

        # (b, c, h, w) -> (b, n_patches, d_embd)
        image_embd = self.vision_model(pixel_values.to(input_embd.dtype))

        # (b, n_patches, d_embd) -> (b, n_patches, d_vocab)
        image_embd = self.mm_projector(image_embd)

        # Merge the modalities along the embedding dimension - replace the placeholder image tokens with the image embeddings
        input_embd, attention_mask, position_ids = self._merge_modalities(
            image_embd, input_embd, input_ids, attention_mask, kv_cache
        )

        # (b, s, d_vocab) -> (b, s, d_vocab)
        outputs = self.language_model(
            input_embd,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_cache=kv_cache,
        )

        return outputs


class KVCache:
    def __init__(self):
        self.k_cache: List[torch.Tensor] = []
        self.v_cache: List[torch.Tensor] = []

    def num_items(self):
        if len(self.k_cache) == 0:
            return 0

        else:
            return self.k_cache[0].shape[-2]

    def update(self, k: torch.Tensor, v: torch.Tensor, layer_idx: int):
        if len(self.k_cache) <= layer_idx:
            self.k_cache.append(k)
            self.v_cache.append(v)
        else:
            self.k_cache[layer_idx] = torch.cat([self.k_cache[layer_idx], k], dim=-2)
            self.v_cache[layer_idx] = torch.cat([self.v_cache[layer_idx], v], dim=-2)

        return self.k_cache[layer_idx], self.v_cache[layer_idx]
