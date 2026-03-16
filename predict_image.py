from __future__ import annotations

import argparse
from pathlib import Path

from cat_dog import ImagePredictor, TrainingConfig
from setup_gpu import TensorFlowConfig


def parse_args() -> argparse.Namespace:
    config = TrainingConfig()

    parser = argparse.ArgumentParser(
        description="Nhận diện 1 ảnh mới là mèo hay chó bằng model đã train.",
    )
    parser.add_argument(
        "image",
        type=Path,
        help="Đường dẫn ảnh cần dự đoán.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=config.best_model_path,
        help="Đường dẫn model .keras (mặc định: best_cat_dog_model.keras)",
    )
    return parser.parse_args()


def resolve_image_path(image_path: Path, dataset_root: Path) -> Path:
    if image_path.is_file():
        return image_path

    normalized_parts = [part for part in image_path.parts if part not in (".", "")]
    normalized_path = Path(*normalized_parts) if normalized_parts else image_path

    candidates = [
        dataset_root / normalized_path,
        dataset_root / "test" / normalized_path,
    ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    searched = "\n".join(f"- {path}" for path in [image_path, *candidates])
    raise FileNotFoundError(
        "Không tìm thấy ảnh. Đã thử các đường dẫn:\n"
        f"{searched}\n"
        "Ví dụ đúng: python predict_image.py Datasets/test/cats/cat.jpg"
    )


def main() -> None:
    args = parse_args()

    TensorFlowConfig.init_gpu()
    config = TrainingConfig()
    image_path = resolve_image_path(args.image, config.dataset_root)

    predictor = ImagePredictor(
        model_path=args.model,
        image_size=config.image_size,
    )
    result = predictor.predict(image_path)

    print("\n--- KẾT QUẢ DỰ ĐOÁN ---")
    print(f"Ảnh: {image_path}")
    print(f"Model: {args.model}")
    print(f"Nhãn dự đoán: {result.label}")
    print(f"Độ tin cậy: {result.confidence * 100:.2f}%")
    print(f"Xác suất là chó (dog): {result.dog_probability * 100:.2f}%")


if __name__ == "__main__":
    main()
