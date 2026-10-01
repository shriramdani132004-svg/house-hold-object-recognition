"""Run the COCO-pretrained baseline on a single image.

Run from the project root:

    python scripts/baseline_inference.py --source path/to/image.jpg
        [--model models/baseline/yolo26n.pt] [--conf 0.25] [--imgsz 640]
        [--device cpu] [--save]

Loads the model once, predicts on the image, prints one line per detected
object (confidence, class name, pixel bounding box), reports the inference
time, and with --save writes the annotated copy under examples/output/.
Exit code 0 on success, 1 on invalid input or missing weights.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "baseline.yaml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "examples" / "output"


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load the Phase 5 baseline YAML configuration."""
    if not path.is_file():
        raise FileNotFoundError(f"baseline config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    return data


def resolve_path(relative: str | Path) -> Path:
    """Resolve a project-root-relative path to an absolute one."""
    path = Path(relative)
    return path if path.is_absolute() else PROJECT_ROOT / path


def ensure_model(model_path: Path) -> Path:
    """Return an existing weight file, downloading yolo26n.pt if missing."""
    if model_path.is_file():
        return model_path
    from ultralytics.utils.downloads import attempt_download_asset

    print(
        f"weights not found at {model_path.relative_to(PROJECT_ROOT)}; "
        "downloading yolo26n.pt ...",
        file=sys.stderr,
    )
    downloaded = Path(
        attempt_download_asset(
            "yolo26n.pt",
            repo="ultralytics/assets",
            release="v8.4.0",
        )
    )
    if not downloaded.is_file():
        print("error: download failed for yolo26n.pt", file=sys.stderr)
        raise SystemExit(1)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if downloaded.resolve() != model_path.resolve():
        shutil.copy2(downloaded, model_path)
    return model_path


def format_detection(
    class_name: str,
    confidence: float,
    box_px: tuple[int, int, int, int],
) -> str:
    """One human-readable line for a single detection."""
    x0, y0, x1, y1 = box_px
    return f"{confidence:6.3f}  {class_name:<14} box=[{x0}, {y0}, {x1}, {y1}]"


def run_inference(
    source: Path,
    model_path: Path,
    conf: float,
    imgsz: int,
    device: str,
    save: bool,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Predict once with the baseline model and return structured results."""
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    started = time.perf_counter()
    result = model.predict(
        source=str(source),
        conf=conf,
        imgsz=imgsz,
        device=device,
        verbose=False,
        save=False,
    )[0]
    wall_ms = (time.perf_counter() - started) * 1000.0
    names: dict[int, str] = result.names
    detections: list[dict[str, Any]] = []
    if result.boxes is not None and len(result.boxes):
        boxes = result.boxes.xyxy.cpu().tolist()
        confidences = result.boxes.conf.cpu().tolist()
        classes = result.boxes.cls.cpu().tolist()
        for box, score, class_index in zip(boxes, confidences, classes):
            x0, y0, x1, y1 = (int(round(value)) for value in box)
            detections.append(
                {
                    "class_id": int(class_index),
                    "class_name": names[int(class_index)],
                    "confidence": float(score),
                    "bbox_px": (x0, y0, x1, y1),
                }
            )
    saved_to: Path | None = None
    if save:
        output_dir.mkdir(parents=True, exist_ok=True)
        saved_to = output_dir / f"{source.stem}_baseline{source.suffix or '.jpg'}"
        result.save(filename=str(saved_to))
    return {
        "detections": detections,
        "wall_ms": wall_ms,
        "speed_ms": float(result.speed.get("inference", 0.0)),
        "saved_to": saved_to,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the COCO-pretrained baseline on one image."
    )
    parser.add_argument(
        "--source",
        required=True,
        help="path to an input image (absolute or relative to project root)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="model weights (default: model.path from configs/baseline.yaml)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="minimum confidence for reported detections (default: 0.25)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="inference image size (default: 640)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="inference device (default: cpu)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="save the annotated image under examples/output/",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = load_config()
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    source = resolve_path(args.source)
    if not source.is_file():
        print(f"error: input image not found: {source}", file=sys.stderr)
        return 1
    if source.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        print(
            f"error: unsupported image type '{source.suffix}' for {source}",
            file=sys.stderr,
        )
        return 1
    relative_model = args.model or config["model"]["path"]
    model_path = ensure_model(resolve_path(relative_model))
    try:
        outcome = run_inference(
            source=source,
            model_path=model_path,
            conf=args.conf,
            imgsz=args.imgsz,
            device=args.device,
            save=args.save,
        )
    except Exception as error:  # noqa: BLE001 - report any inference failure clearly
        print(f"error: inference failed: {error}", file=sys.stderr)
        return 1
    detections = outcome["detections"]
    print(f"model: {model_path.name}  device: {args.device}  conf>={args.conf:.2f}")
    print(f"image: {source.name}  detections: {len(detections)}")
    for detection in detections:
        print(
            "  "
            + format_detection(
                detection["class_name"],
                detection["confidence"],
                detection["bbox_px"],
            )
        )
    print(
        f"timing: model {outcome['speed_ms']:.1f} ms | "
        f"total {outcome['wall_ms']:.1f} ms"
    )
    if outcome["saved_to"] is not None:
        relative_saved = outcome["saved_to"].relative_to(PROJECT_ROOT)
        print(f"saved: {relative_saved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
