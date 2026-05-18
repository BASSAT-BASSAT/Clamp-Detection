from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path
import torch


def load_coco_annotations(annotations_path: Path) -> tuple[list[dict], list[dict], dict[int, int], list[str]]:
    with annotations_path.open("r", encoding="utf-8") as file_handle:
        dataset = json.load(file_handle)

    images = dataset.get("images", [])
    annotations = dataset.get("annotations", [])
    categories = dataset.get("categories", [])

    class_names: list[str] = []
    category_id_to_class_index: dict[int, int] = {}

    for category in categories:
        category_name = str(category["name"]).strip()
        if category_name not in class_names:
            class_names.append(category_name)
        category_id_to_class_index[int(category["id"])] = class_names.index(category_name)

    return images, annotations, category_id_to_class_index, class_names


def convert_bbox_to_yolo(bbox: list[float], width: int, height: int) -> tuple[float, float, float, float] | None:
    x, y, box_width, box_height = map(float, bbox)
    if box_width <= 0 or box_height <= 0:
        return None

    x1 = max(0.0, x)
    y1 = max(0.0, y)
    x2 = min(float(width), x + box_width)
    y2 = min(float(height), y + box_height)

    clipped_width = x2 - x1
    clipped_height = y2 - y1
    if clipped_width <= 0 or clipped_height <= 0:
        return None

    x_center = (x1 + x2) / 2.0 / width
    y_center = (y1 + y2) / 2.0 / height
    normalized_width = clipped_width / width
    normalized_height = clipped_height / height
    return x_center, y_center, normalized_width, normalized_height


def prepare_yolo_dataset(dataset_root: Path, output_root: Path, seed: int = 42, validation_ratio: float = 0.2) -> Path:
    annotations_path = dataset_root / "train" / "_annotations.coco.json"
    if not annotations_path.exists():
        raise FileNotFoundError(f"Missing annotations file: {annotations_path}")

    images, annotations, category_id_to_class_index, class_names = load_coco_annotations(annotations_path)

    source_image_dir = dataset_root / "train"
    if not source_image_dir.exists():
        raise FileNotFoundError(f"Missing image directory: {source_image_dir}")

    image_annotations: dict[int, list[dict]] = defaultdict(list)
    for annotation in annotations:
        image_annotations[int(annotation["image_id"])].append(annotation)

    shuffled_images = list(images)
    random.Random(seed).shuffle(shuffled_images)

    validation_count = max(1, int(len(shuffled_images) * validation_ratio))
    if validation_count >= len(shuffled_images):
        validation_count = 1
    validation_images = shuffled_images[:validation_count]
    training_images = shuffled_images[validation_count:]

    if not training_images:
        raise ValueError("Not enough images to create a training split.")

    images_train_dir = output_root / "images" / "train"
    images_val_dir = output_root / "images" / "val"
    labels_train_dir = output_root / "labels" / "train"
    labels_val_dir = output_root / "labels" / "val"

    for directory in [images_train_dir, images_val_dir, labels_train_dir, labels_val_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    def write_split(split_images: list[dict], images_dir: Path, labels_dir: Path) -> None:
        for image_info in split_images:
            source_name = image_info["file_name"]
            source_image_path = source_image_dir / source_name
            if not source_image_path.exists():
                raise FileNotFoundError(f"Missing image file: {source_image_path}")

            target_image_path = images_dir / source_image_path.name
            shutil.copy2(source_image_path, target_image_path)

            label_path = labels_dir / f"{source_image_path.stem}.txt"
            lines: list[str] = []
            for annotation in image_annotations.get(int(image_info["id"]), []):
                class_index = category_id_to_class_index[int(annotation["category_id"])]
                yolo_bbox = convert_bbox_to_yolo(annotation["bbox"], int(image_info["width"]), int(image_info["height"]))
                if yolo_bbox is None:
                    continue
                x_center, y_center, box_width, box_height = yolo_bbox
                lines.append(f"{class_index} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}")

            label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    write_split(training_images, images_train_dir, labels_train_dir)
    write_split(validation_images, images_val_dir, labels_val_dir)

    data_yaml = output_root / "data.yaml"
    class_names_yaml = "\n".join(f"  {index}: {name}" for index, name in enumerate(class_names))
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {output_root.resolve().as_posix()}",
                "train: images/train",
                "val: images/val",
                f"nc: {len(class_names)}",
                "names:",
                class_names_yaml,
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    return data_yaml


def train_model(data_yaml: Path, model_name: str, epochs: int, imgsz: int, project_dir: Path) -> None:
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise SystemExit(
            "ultralytics is not installed. Run: pip install ultralytics"
        ) from error

    def _get_device_str() -> str:
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch, "has_mps", False) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    device_str = _get_device_str()
    print(f"Using device: {device_str}")

    model = YOLO(model_name)
    # Try moving the model to the chosen device if supported by the ultralytics model wrapper
    try:
        model.to(device_str)
    except Exception:
        try:
            model.to(torch.device(device_str))
        except Exception:
            pass

    # Pass device to trainer when available; this keeps training device-agnostic
    try:
        model.train(data=str(data_yaml), epochs=epochs, imgsz=imgsz, project=str(project_dir), name="clamp_detection", device=device_str)
    except TypeError:
        # Fallback if ultralytics version does not accept device kwarg
        model.train(data=str(data_yaml), epochs=epochs, imgsz=imgsz, project=str(project_dir), name="clamp_detection")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the Roboflow COCO export and fine-tune a YOLO model.")
    parser.add_argument("--dataset-root", type=Path, default=Path(__file__).resolve().parent / "Clamp Detection.coco")
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent / "clamp_detection_yolo")
    parser.add_argument("--model", type=str, default="yolo11m.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_root.exists():
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)

    data_yaml = prepare_yolo_dataset(
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        seed=args.seed,
        validation_ratio=args.val_ratio,
    )
    print(f"Prepared YOLO dataset at {args.output_root}")
    print(f"Data config: {data_yaml}")

    train_model(
        data_yaml=data_yaml,
        model_name=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        project_dir=Path(__file__).resolve().parent / "runs" / "detect",
    )


if __name__ == "__main__":
    main()