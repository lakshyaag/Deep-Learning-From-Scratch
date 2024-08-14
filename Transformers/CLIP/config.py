from dataclasses import dataclass


@dataclass
class Config:
    D_IMG_MODEL: int = 2048
    D_TEXT_MODEL: int = 768
    D_EMBD: int = 512
    IMAGE_SHAPE: tuple = (3, 224, 224)
    TOKENIZER_MAX_LEN: int = 200
    INITIAL_TEMPERATURE: float = 0.07
    DROPOUT: float = 0.1
    IMAGE_MODEL: str = "resnet50"
    TEXT_MODEL: str = "distilbert-base-uncased"
    TRAINABLE: bool = False

    BATCH_SIZE: int = 32
    N_EPOCHS: int = 3
    IMAGE_LEARNING_RATE: float = 1e-4
    TEXT_LEARNING_RATE: float = 1e-5
    LEARNING_RATE: float = 1e-3
    WEIGHT_DECAY: float = 1e-3

    EXPERIMENT_NAME: str = "clip-openai"
