from dataclasses import dataclass

from torch.cuda import is_available


@dataclass
class Config:
    BATCH_SIZE: int = 256
    LEARNING_RATE: float = 2e-5
    N_EPOCHS: int = 50
    N_TIMESTEPS: int = 1000

    IMAGE_SIZE: int = 32
    BETA_START: float = 1e-4
    BETA_END: float = 1e-2
    NOISE_SCHEDULE: str = "cosine"

    N_LAYERS: int = 2
    N_HEADS: int = 4
    N_GROUPS: int = 32

    IN_CHANNELS: int = 3
    BASE_CHANNELS: int = 128

    DEVICE: str = "cuda" if is_available() else "cpu"

    DATA_DIR: str = "./data"
    MODEL_DIR: str = "./data/models"
    SAMPLE_DIR: str = "./data/samples"

    EVAL_BATCH_SIZE: int = 8
    EVAL_EVERY_EPOCH: int = 5
    CHECKPOINT_EVERY_EPOCH: int = 5
    SAVE_INTERVAL: int = 100

    EXPERIMENT_NAME: str = "ddpm-cifar10"
