from __future__ import annotations

from pathlib import Path

from cat_dog import ImagePredictor, TrainingConfig
from setup_gpu import TensorFlowConfig


def first_image_in_test_set(dataset_root: Path) -> Path:
    candidates = [
        dataset_root / "test" / "cats",
        dataset_root / "test" / "dogs",
    ]

    for folder in candidates:
        if not folder.is_dir():
            continue
        for image_path in sorted(folder.glob("*.jpg")):
            return image_path
        for image_path in sorted(folder.glob("*.png")):
            return image_path

    raise FileNotFoundError("Không tìm thấy ảnh .jpg/.png trong Datasets/test")


def main() -> None:
    TensorFlowConfig.init_gpu()

    config = TrainingConfig()
    image_path = first_image_in_test_set(config.dataset_root)

    predictor = ImagePredictor(
        model_path=config.best_model_path,
        image_size=config.image_size,
    )
    result = predictor.predict(image_path)

    print("\n--- DỰ ĐOÁN MẪU ---")
    print(f"Ảnh: {image_path}")
    print(f"Nhãn dự đoán: {result.label}")
    print(f"Độ tin cậy: {result.confidence * 100:.2f}%")


if __name__ == "__main__":
    main()
