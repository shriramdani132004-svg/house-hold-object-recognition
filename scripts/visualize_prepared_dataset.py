"""Draw converted YOLO boxes on sampled prepared images.

Run from the project root:

    python scripts/visualize_prepared_dataset.py [--per-split 4] [--seed 42]

Randomly samples images from the prepared train/val/test splits, draws
the YOLO bounding boxes with class names, and saves the annotated copies
under reports/phase3_samples/<split>/. Exit code 0 on success.
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "dataset.yaml"
OUTPUT_ROOT = PROJECT_ROOT / "reports" / "phase3_samples"
SPLITS = ("train", "val", "test")


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    return data


def class_color(class_id: int) -> tuple[int, int, int]:
    hue = (class_id * 0.618033988749895) % 1.0
    return hsv_to_rgb(hue, 0.85, 0.95)


def hsv_to_rgb(hue: float, saturation: float, value: float) -> tuple[int, int, int]:
    index = int(hue * 6.0)
    fraction = hue * 6.0 - index
    p = value * (1.0 - saturation)
    q = value * (1.0 - fraction * saturation)
    t = value * (1.0 - (1.0 - fraction) * saturation)
    red, green, blue = (
        (value, t, p),
        (q, value, p),
        (p, value, t),
        (p, q, value),
        (t, p, value),
        (value, p, q),
    )[index % 6]
    return int(red * 255), int(green * 255), int(blue * 255)


def draw_boxes(
    image_path: Image.Image,
    label_path: Path,
    names: list[str],
) -> Image.Image:
    width, height = image_path.size
    draw = ImageDraw.Draw(image_path)
    font = ImageFont.load_default(size=13)
    content = label_path.read_text(encoding="utf-8").strip()
    if not content:
        return image_path
    for line in content.splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        class_id = int(parts[0])
        xc, yc, box_w, box_h = (float(value) for value in parts[1:])
        x0 = (xc - box_w / 2.0) * width
        y0 = (yc - box_h / 2.0) * height
        x1 = (xc + box_w / 2.0) * width
        y1 = (yc + box_h / 2.0) * height
        color = class_color(class_id)
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
        label = names[class_id] if class_id < len(names) else str(class_id)
        text_box = draw.textbbox((x0, y0), label, font=font)
        draw.rectangle(
            [text_box[0], text_box[1] - 2, text_box[2] + 2, text_box[3] + 2],
            fill=color,
        )
        draw.text((x0 + 2, y0), label, fill=(255, 255, 255), font=font)
    return image_path


def sample_and_draw(
    output_dir: Path,
    names: list[str],
    destination: Path,
    per_split: int,
    seed: int,
) -> dict[str, list[str]]:
    if destination.is_dir():
        shutil.rmtree(destination)
    written: dict[str, list[str]] = {}
    for split in SPLITS:
        images_dir = output_dir / "images" / split
        labels_dir = output_dir / "labels" / split
        if not images_dir.is_dir():
            written[split] = []
            continue
        image_files = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        rng = random.Random(f"{seed}-{split}")
        chosen = rng.sample(image_files, min(per_split, len(image_files)))
        split_dir = destination / split
        split_dir.mkdir(parents=True, exist_ok=True)
        written[split] = []
        for image_path in chosen:
            with Image.open(image_path) as opened:
                canvas = opened.convert("RGB")
            label_path = labels_dir / f"{image_path.stem}.txt"
            if label_path.is_file():
                canvas = draw_boxes(canvas, label_path, names)
            target = split_dir / image_path.name
            canvas.save(target, quality=90)
            written[split].append(target.name)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--per-split", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    try:
        config = load_yaml(args.config)
        class_map = load_classes(config["classes_config"])
        output_dir = PROJECT_ROOT / config["output"]["dir"]
    except (OSError, ValueError, KeyError) as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1

    if not output_dir.is_dir():
        print(f"[FAIL] output directory not found: {output_dir}", file=sys.stderr)
        return 1

    names = [class_map[index] for index in range(len(class_map))]
    try:
        written = sample_and_draw(
            output_dir, names, OUTPUT_ROOT, args.per_split, args.seed
        )
    except (OSError, ValueError) as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1

    total = 0
    for split in SPLITS:
        files = written.get(split, [])
        total += len(files)
        print(f"[ok] {split}: {len(files)} sample(s) -> reports/phase3_samples/{split}/")
    print(f"[ok] wrote {total} annotated samples")
    return 0 if total else 1


def load_classes(path_value: Any) -> dict[int, str]:
    path = Path(path_value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    data = load_yaml(path)
    class_map = {int(key): str(value) for key, value in data["classes"].items()}
    if sorted(class_map) != list(range(len(class_map))):
        raise ValueError(f"{path}: class ids must be consecutive starting at 0")
    return class_map


if __name__ == "__main__":
    raise SystemExit(main())
