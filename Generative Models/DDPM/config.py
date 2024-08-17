from dataclasses import dataclass

from torch.cuda import is_available


@dataclass
class Config:
    BATCH_SIZE = 128
    LEARNING_RATE = 2e-5
    N_EPOCHS = 150
    N_TIMESTEPS = 1000

    IMAGE_SIZE = 32
    BETA_START = 1e-4
    BETA_END = 1e-2

    N_LAYERS = 2
    N_HEADS = 4
    N_GROUPS = 32

    IN_CHANNELS = 3
    BASE_CHANNELS = 128

    DEVICE = "cuda" if is_available() else "cpu"

    DATA_DIR = "./data"
    MODEL_DIR = "./data/model"
    SAMPLE_DIR = "./data/sample"

    EVAL_BATCH_SIZE = 8
    EVAL_EVERY_EPOCH = 10
    CHECKPOINT_EVERY_EPOCH = 10

    EXPERIMENT_NAME = "ddpm-cifar10"
