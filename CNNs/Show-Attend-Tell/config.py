from dataclasses import dataclass


@dataclass
class Config:
    VOCAB_SIZE: int

    BATCH_SIZE: int = 64
    N_EPOCHS: int = 20
    LEARNING_RATE: float = 3e-4
    ENCODED_IMAGE_SIZE: int = 14
    EMBED_DIM: int = 512
    DECODER_DIM: int = 512
    ATTENTION_DIM: int = 768
    ENCODER_DIM: int = 2048
    DROPOUT: float = 0.5
    BEAM_SIZE: int = 5
    MAX_CAPTION_LENGTH: int = 50

    EXPERIMENT_NAME: str = "show-attend-tell"
