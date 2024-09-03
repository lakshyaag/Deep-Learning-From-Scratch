class SiglipVisionConfig:
    def __init__(
        self,
        hidden_size: int = 768,
        intermediate_size: int = 3072,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        num_channels: int = 3,
        image_size: int = 32,
        patch_size: int = 8,
        layer_norm_eps: float = 1e-6,
        attention_dropout: float = 0.0,
        num_image_tokens: int = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.num_image_tokens = num_image_tokens


class GemmaLMConfig:
    def __init__(
        self,
        d_vocab: int = 256000,
        hidden_size: int = 768,
        intermediate_size: int = 3072,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        num_kv_heads: int = 1,
        d_head: int = 256,
        max_position_embeddings: int = 8192,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 10000.0,
        attention_bias: bool = True,
        attention_dropout: float = 0.0,
        pad_token_id: int = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.d_vocab = d_vocab
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_kv_heads = num_kv_heads
        self.d_head = d_head
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.pad_token_id = pad_token_id


class PaliGemmaConfig:
    def __init__(
        self,
        vision_config: SiglipVisionConfig,
        text_config: GemmaLMConfig,
        ignore_index: int = -100,
        image_token_index: int = 256000,
        d_vocab: int = 257152,
        projection_dim: int = 2048,
        hidden_size: int = 2048,
        pad_token_id: int = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.d_vocab = d_vocab
        self.projection_dim = projection_dim
        self.hidden_size = hidden_size

        self.vision_config = SiglipVisionConfig(**vision_config)
        self.is_encoder_decoder: bool = False

        self.pad_token_id = pad_token_id
        self.text_config = GemmaLMConfig(**text_config, pad_token_id=pad_token_id)
        self.d_vocab = self.text_config.d_vocab
        self.text_config.num_image_tokens = (
            self.vision_config.image_size // self.vision_config.patch_size
        ) ** 2
        self.vision_config.projection_dim = projection_dim
