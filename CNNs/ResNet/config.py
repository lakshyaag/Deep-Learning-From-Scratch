from dataclasses import dataclass


@dataclass
class Config:
    BATCH_SIZE: int = 128
    N_EPOCHS: int = 90
    LEARNING_RATE: float = 0.01
    MOMENTUM: float = 0.9
    WEIGHT_DECAY: float = 0.0005
    N_CLASSES: int = 10
    N_BLOCKS: int = 5

    MLFLOW_TRACKING_URI: str = "http://localhost:5000"
    EXPERIMENT_NAME: str = "resnet"
