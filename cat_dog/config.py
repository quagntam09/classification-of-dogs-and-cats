from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainingConfig:
    dataset_root: Path = Path("Datasets")
    image_size: tuple[int, int] = (180, 180)
    batch_size: int = 32
    epochs: int = 100
    validation_split: float = 0.2
    seed: int = 123
    learning_rate: float = 5e-4
    patience: int = 10
    best_model_path: Path = Path("best_cat_dog_model.keras")
    final_model_path: Path = Path("cat_dog_cnn_model_final.keras")
