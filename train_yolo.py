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


def prepare_yolo_dataset(
    dataset_root: Path,
    output_root: Path,
    seed: int = 42,
    train_ratio: float = 0.80,
    val_ratio: float = 0.15,
    test_ratio: float = 0.05,
) -> Path:
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio, val_ratio, and test_ratio must sum to 1.0")

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

    total_images = len(shuffled_images)
    if total_images < 3:
        raise ValueError(f"Need at least 3 images for train/val/test split, got {total_images}.")

    train_count = int(total_images * train_ratio)
    val_count = int(total_images * val_ratio)
    test_count = total_images - train_count - val_count

    if min(train_count, val_count, test_count) < 1:
        raise ValueError(
            f"Split produced train={train_count}, val={val_count}, test={test_count}. "
            "Each split needs at least 1 image — add more data or adjust ratios."
        )

    train_images = shuffled_images[:train_count]
    val_images = shuffled_images[train_count : train_count + val_count]
    test_images = shuffled_images[train_count + val_count :]

    split_dirs = {
        "train": (output_root / "images" / "train", output_root / "labels" / "train", train_images),
        "val": (output_root / "images" / "val", output_root / "labels" / "val", val_images),
        "test": (output_root / "images" / "test", output_root / "labels" / "test", test_images),
    }

    for images_dir, labels_dir, _ in split_dirs.values():
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

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
                yolo_bbox = convert_bbox_to_yolo(
                    annotation["bbox"],
                    int(image_info["width"]),
                    int(image_info["height"]),
                )
                if yolo_bbox is None:
                    continue
                x_center, y_center, box_width, box_height = yolo_bbox
                lines.append(
                    f"{class_index} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"
                )

            label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    for images_dir, labels_dir, split_images in split_dirs.values():
        write_split(split_images, images_dir, labels_dir)

    print(
        f"Split ({total_images} images): "
        f"train={len(train_images)} ({len(train_images) / total_images:.0%}), "
        f"val={len(val_images)} ({len(val_images) / total_images:.0%}), "
        f"test={len(test_images)} ({len(test_images) / total_images:.0%})"
    )

    data_yaml = output_root / "data.yaml"
    class_names_yaml = "\n".join(f"  {index}: {name}" for index, name in enumerate(class_names))
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {output_root.resolve().as_posix()}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                f"nc: {len(class_names)}",
                "names:",
                class_names_yaml,
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    return data_yaml


def get_device_str() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch, "has_mps", False) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train_model(
    data_yaml: Path,
    model_name: str,
    epochs: int,
    imgsz: int,
    project_dir: Path,
    run_name: str = "clamp_detection",
) -> Path:
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise SystemExit("ultralytics is not installed. Run: pip install ultralytics") from error

    device_str = get_device_str()
    print(f"Using device: {device_str}")

    model = YOLO(model_name)
    try:
        model.to(device_str)
    except Exception:
        try:
            model.to(torch.device(device_str))
        except Exception:
            pass

    try:
        model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=imgsz,
            project=str(project_dir),
            name=run_name,
            device=device_str,
        )
    except TypeError:
        model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=imgsz,
            project=str(project_dir),
            name=run_name,
        )

    return project_dir / run_name / "weights" / "best.pt"


def find_best_weights(project_dir: Path, run_name: str = "clamp_detection") -> Path:
    candidates = [
        project_dir / run_name / "weights" / "best.pt",
        Path("runs") / run_name / "weights" / "best.pt",
        Path("runs/detect/runs/detect") / run_name / "weights" / "best.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Could not find best.pt for run '{run_name}'. Train the model first or pass weights_path explicitly."
    )


def predict_video(
    input_video: Path,
    weights_path: Path | None = None,
    project_dir: Path | None = None,
    run_name: str = "clamp_detection",
    inference_name: str = "video_inference",
    conf: float = 0.25,
    imgsz: int = 640,
) -> Path:
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise SystemExit("ultralytics is not installed. Run: pip install ultralytics") from error

    if not input_video.exists():
        raise FileNotFoundError(f"Input video not found: {input_video}")

    project_dir = project_dir or Path("runs")
    weights_path = weights_path or find_best_weights(project_dir, run_name=run_name)

    print(f"Weights: {weights_path}")
    print(f"Input video: {input_video}")

    model = YOLO(str(weights_path))
    results = model.predict(
        source=str(input_video),
        save=True,
        conf=conf,
        imgsz=imgsz,
        project=str(project_dir),
        name=inference_name,
        device=get_device_str(),
    )

    save_dir = Path(results[0].save_dir)
    saved_videos = sorted(save_dir.glob("*.mp4")) + sorted(save_dir.glob("*.avi"))
    print(f"Saved outputs to: {save_dir}")

    if saved_videos:
        print(f"Annotated video: {saved_videos[0]}")
        return saved_videos[0]

    return save_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the Roboflow COCO export and fine-tune a YOLO model.")
    parser.add_argument("--dataset-root", type=Path, default=Path(__file__).resolve().parent / "Clamp Detection.coco")
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent / "clamp_detection_yolo")
    parser.add_argument("--model", type=str, default="yolo11m.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parent / "runs")
    parser.add_argument("--run-name", type=str, default="clamp_detection")
    parser.add_argument("--predict-video", type=Path, default=None, help="Run inference on this video after training")
    parser.add_argument("--conf", type=float, default=0.25)
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
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
    print(f"Prepared YOLO dataset at {args.output_root}")
    print(f"Data config: {data_yaml}")

    train_model(
        data_yaml=data_yaml,
        model_name=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        project_dir=args.project_dir,
        run_name=args.run_name,
    )

    if args.predict_video is not None:
        predict_video(
            input_video=args.predict_video,
            project_dir=args.project_dir,
            run_name=args.run_name,
            conf=args.conf,
            imgsz=args.imgsz,
        )


if __name__ == "__main__":
    main()
