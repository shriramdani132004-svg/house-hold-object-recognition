"""Build the YOLO-format household-object dataset from raw COCO 2017 data.

Run from the project root:

    python scripts/prepare_dataset.py

Reads configs/dataset.yaml and configs/classes.yaml, filters the raw COCO
train/val instance annotations to the selected classes, converts boxes to
YOLO normalized format, assigns a deterministic image-level split (COCO
train -> project train + val, COCO val -> project test), copies only the
relevant images, and writes data/processed/household_objects/.

The output directory is rebuilt on every run (safe to re-run). Files
under data/raw/ are never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "dataset.yaml"
SPLITS = ("train", "val", "test")
DUPLICATE_KEEP_ORDER = ("test", "val", "train")


class PrepareError(RuntimeError):
    """Raised when the preparation pipeline cannot continue safely."""


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise PrepareError(f"{path}: expected a YAML mapping")
    return data


def load_classes(path: Path) -> dict[int, str]:
    data = load_yaml(path)
    raw = data.get("classes")
    if not isinstance(raw, dict) or not raw:
        raise PrepareError(f"{path}: 'classes' must be a non-empty mapping")
    class_map: dict[int, str] = {}
    for key, value in raw.items():
        try:
            class_id = int(key)
        except (TypeError, ValueError) as exc:
            raise PrepareError(f"{path}: class id {key!r} is not an integer") from exc
        if not isinstance(value, str) or not value.strip():
            raise PrepareError(f"{path}: class {class_id} has an empty name")
        class_map[class_id] = value
    if sorted(class_map) != list(range(len(class_map))):
        raise PrepareError(f"{path}: class ids must be consecutive starting at 0")
    if len(set(class_map.values())) != len(class_map):
        raise PrepareError(f"{path}: duplicate class names")
    return class_map


def coco_to_yolo(
    bbox: Any, img_width: int, img_height: int, min_box_size: float = 0.0
) -> tuple[float, float, float, float] | None:
    """Convert a COCO [x, y, w, h] box to normalized YOLO values.

    Returns (x_center, y_center, width, height) in [0, 1], clipped to the
    image bounds, or None when the box is malformed or too small.
    """
    if img_width <= 0 or img_height <= 0:
        return None
    try:
        x, y, width, height = (float(value) for value in bbox)
    except (TypeError, ValueError):
        return None
    values = (x, y, width, height)
    if not all(math.isfinite(value) for value in values):
        return None
    if width <= 0 or height <= 0:
        return None
    x0 = max(0.0, x)
    y0 = max(0.0, y)
    x1 = min(float(img_width), x + width)
    y1 = min(float(img_height), y + height)
    clipped_w = x1 - x0
    clipped_h = y1 - y0
    if clipped_w < min_box_size or clipped_h < min_box_size:
        return None
    return (
        (x0 + x1) / 2.0 / img_width,
        (y0 + y1) / 2.0 / img_height,
        clipped_w / img_width,
        clipped_h / img_height,
    )


def load_instances(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise PrepareError(f"annotation file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    for key in ("images", "annotations", "categories"):
        if not isinstance(data.get(key), list):
            raise PrepareError(f"{path.name}: '{key}' missing or not a list")
    return data


def collect_labels(
    instances: dict[str, Any],
    name_to_class: dict[str, int],
    include_iscrowd: bool,
    min_box_size: float,
) -> tuple[dict[int, list[tuple[int, tuple[float, float, float, float]]]], dict[str, int]]:
    category_names = {
        int(category["id"]): str(category["name"])
        for category in instances["categories"]
    }
    images_by_id = {int(image["id"]): image for image in instances["images"]}
    kept: dict[int, list[tuple[int, tuple[float, float, float, float]]]] = {}
    rejected = {"unselected": 0, "crowd": 0, "missing_image": 0, "invalid_box": 0}
    for annotation in instances["annotations"]:
        try:
            category_id = int(annotation.get("category_id", -1))
            image_id = int(annotation.get("image_id", -1))
        except (TypeError, ValueError):
            rejected["missing_image"] += 1
            continue
        class_name = category_names.get(category_id)
        if class_name is None or class_name not in name_to_class:
            rejected["unselected"] += 1
            continue
        if not include_iscrowd and int(annotation.get("iscrowd", 0)) == 1:
            rejected["crowd"] += 1
            continue
        image = images_by_id.get(image_id)
        if image is None:
            rejected["missing_image"] += 1
            continue
        box = coco_to_yolo(
            annotation.get("bbox"),
            int(image.get("width", 0)),
            int(image.get("height", 0)),
            min_box_size,
        )
        if box is None:
            rejected["invalid_box"] += 1
            continue
        kept.setdefault(image_id, []).append((name_to_class[class_name], box))
    return kept, rejected


def split_image_ids(
    image_ids: list[int], seed: int, val_fraction: float
) -> dict[str, list[int]]:
    ids = sorted(image_ids)
    random.Random(seed).shuffle(ids)
    count = len(ids)
    if count < 2:
        return {"train": ids, "val": []}
    val_count = int(round(count * val_fraction))
    val_count = min(max(1, val_count), count - 1)
    return {"train": ids[val_count:], "val": ids[:val_count]}


def copy_with_hash(source: Path, destination: Path) -> tuple[str, int]:
    sha = hashlib.sha1()
    written = 0
    with source.open("rb") as src, destination.open("wb") as dst:
        for chunk in iter(lambda: src.read(1024 * 1024), b""):
            sha.update(chunk)
            dst.write(chunk)
            written += len(chunk)
    return sha.hexdigest(), written


def _existing_parent(path: Path) -> Path:
    probe = path
    while not probe.exists():
        parent = probe.parent
        if parent == probe:
            break
        probe = parent
    return probe


def _rebuild_output(output_dir: Path) -> None:
    for name in ("images", "labels"):
        target = output_dir / name
        if target.is_dir():
            shutil.rmtree(target)
        target.mkdir(parents=True)
        for split in SPLITS:
            (target / split).mkdir()
    for name in ("dataset.yaml", "metadata.json"):
        target = output_dir / name
        if target.is_file():
            target.unlink()


def write_dataset_yaml(output_dir: Path, class_map: dict[int, str]) -> None:
    names = [class_map[index] for index in range(len(class_map))]
    payload = {
        "path": ".",
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(names),
        "names": names,
    }
    text = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
    (output_dir / "dataset.yaml").write_text(text, encoding="utf-8")


def _write_label(
    path: Path, boxes: list[tuple[int, tuple[float, float, float, float]]]
) -> None:
    lines = [
        f"{class_id} {xc:.6f} {yc:.6f} {width:.6f} {height:.6f}"
        for class_id, (xc, yc, width, height) in boxes
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _count_split(output_dir: Path, split: str) -> dict[str, int]:
    image_files = sorted((output_dir / "images" / split).iterdir())
    label_files = sorted((output_dir / "labels" / split).iterdir())
    instances = 0
    for label in label_files:
        content = label.read_text(encoding="utf-8").strip()
        if content:
            instances += len(content.splitlines())
    return {
        "images": len(image_files),
        "labels": len(label_files),
        "instances": instances,
    }


def prepare(
    raw_dir: Path,
    output_dir: Path,
    class_map: dict[int, str],
    *,
    train_annotations: str = "annotations/instances_train2017.json",
    val_annotations: str = "annotations/instances_val2017.json",
    seed: int = 42,
    val_fraction: float = 0.1,
    include_iscrowd: bool = False,
    min_box_size: float = 1.0,
) -> dict[str, Any]:
    if not 0.0 < val_fraction < 1.0:
        raise PrepareError(f"val_fraction must be in (0, 1), got {val_fraction}")
    name_to_class = {name: class_id for class_id, name in class_map.items()}

    train_data = load_instances(raw_dir / train_annotations)
    val_data = load_instances(raw_dir / val_annotations)

    train_labels, train_rejected = collect_labels(
        train_data, name_to_class, include_iscrowd, min_box_size
    )
    val_labels, val_rejected = collect_labels(
        val_data, name_to_class, include_iscrowd, min_box_size
    )
    rejected = {
        key: train_rejected[key] + val_rejected[key] for key in train_rejected
    }

    image_info: dict[int, dict[str, Any]] = {}
    for data, source_split in ((train_data, "train2017"), (val_data, "val2017")):
        for image in data["images"]:
            info = dict(image)
            info["source_split"] = source_split
            image_info[int(image["id"])] = info

    assigned: dict[int, str] = {}
    split_ids = split_image_ids(sorted(train_labels), seed, val_fraction)
    for image_id in split_ids["train"]:
        assigned[image_id] = "train"
    for image_id in split_ids["val"]:
        assigned[image_id] = "val"
    overlapping_ids = 0
    for image_id in val_labels:
        if assigned.get(image_id) == "train":
            overlapping_ids += 1
        assigned[image_id] = "test"

    duplicate_names_removed = 0
    seen_names: dict[str, int] = {}
    priority = {split: index for index, split in enumerate(DUPLICATE_KEEP_ORDER)}
    for image_id in sorted(assigned, key=lambda i: (image_info[i]["file_name"], i)):
        file_name = image_info[image_id]["file_name"]
        previous = seen_names.get(file_name)
        if previous is None:
            seen_names[file_name] = image_id
            continue
        loser = image_id if priority[assigned[image_id]] > priority[assigned[previous]] else previous
        del assigned[loser]
        if loser != image_id:
            seen_names[file_name] = image_id
        duplicate_names_removed += 1

    selected_train = sum(1 for value in assigned.values() if value == "train")
    selected_val = sum(1 for value in assigned.values() if value == "val")
    selected_test = sum(1 for value in assigned.values() if value == "test")

    total_bytes = 0
    for image_id in assigned:
        info = image_info[image_id]
        source = raw_dir / info["source_split"] / info["file_name"]
        if not source.is_file():
            raise PrepareError(f"missing source image: {source}")
        total_bytes += source.stat().st_size
    free_bytes = shutil.disk_usage(_existing_parent(output_dir)).free
    if free_bytes < total_bytes + 64 * 1024 * 1024:
        raise PrepareError(
            f"insufficient disk space: need ~{total_bytes} bytes + 64 MiB margin,"
            f" free {free_bytes} bytes"
        )

    _rebuild_output(output_dir)

    hash_map: dict[str, list[tuple[str, str]]] = {}
    bytes_copied = 0
    for split in SPLITS:
        for image_id in sorted(i for i, s in assigned.items() if s == split):
            info = image_info[image_id]
            source = raw_dir / info["source_split"] / info["file_name"]
            destination = output_dir / "images" / split / info["file_name"]
            digest, written = copy_with_hash(source, destination)
            bytes_copied += written
            hash_map.setdefault(digest, []).append((split, destination.name))
            _write_label(
                output_dir / "labels" / split / f"{destination.stem}.txt",
                train_labels.get(image_id) or val_labels.get(image_id) or [],
            )

    duplicate_hashes_removed = {"train": 0, "val": 0, "test": 0}
    for occurrences in hash_map.values():
        present = {split for split, _ in occurrences}
        if len(present) < 2:
            continue
        keep_split = next(split for split in DUPLICATE_KEEP_ORDER if split in present)
        for split, file_name in occurrences:
            if split == keep_split:
                continue
            image_path = output_dir / "images" / split / file_name
            if image_path.is_file():
                image_path.unlink()
            label_path = output_dir / "labels" / split / f"{Path(file_name).stem}.txt"
            if label_path.is_file():
                label_path.unlink()
            duplicate_hashes_removed[split] += 1

    split_stats = {split: _count_split(output_dir, split) for split in SPLITS}
    class_totals = {class_map[index]: 0 for index in range(len(class_map))}
    class_images = {
        split: {class_map[index]: 0 for index in range(len(class_map))}
        for split in SPLITS
    }
    for split in SPLITS:
        for label in (output_dir / "labels" / split).iterdir():
            content = label.read_text(encoding="utf-8").strip()
            if not content:
                continue
            present: set[str] = set()
            for line in content.splitlines():
                class_id = int(line.split()[0])
                name = class_map[class_id]
                class_totals[name] += 1
                present.add(name)
            for name in present:
                class_images[split][name] += 1

    excluded_images = _count_excluded_images(
        train_data, val_data, train_labels, val_labels
    )
    metadata: dict[str, Any] = {
        "source": "COCO 2017",
        "classes": {str(index): class_map[index] for index in range(len(class_map))},
        "split_strategy": "coco_official_val_as_test",
        "seed": seed,
        "val_fraction": val_fraction,
        "include_iscrowd": include_iscrowd,
        "min_box_size_px": min_box_size,
        "selected_source_images": {
            "train2017": selected_train + selected_val,
            "val2017": selected_test,
        },
        "splits": split_stats,
        "instances_per_class": class_totals,
        "images_per_class": class_images,
        "rejected_annotations": rejected,
        "source_images_excluded": excluded_images,
        "duplicates_removed": {
            "by_name": duplicate_names_removed,
            "by_hash": duplicate_hashes_removed,
            "overlapping_source_ids": overlapping_ids,
        },
        "bytes_copied": bytes_copied,
    }

    write_dataset_yaml(output_dir, class_map)
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return metadata


def _count_excluded_images(
    train_data: dict[str, Any],
    val_data: dict[str, Any],
    train_labels: dict[int, Any],
    val_labels: dict[int, Any],
) -> int:
    excluded = 0
    for data, labels in ((train_data, train_labels), (val_data, val_labels)):
        kept_ids = set(labels)
        for image in data["images"]:
            if int(image["id"]) not in kept_ids:
                excluded += 1
    return excluded


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="phase configuration (default: configs/dataset.yaml)",
    )
    args = parser.parse_args(argv)

    try:
        config = load_yaml(args.config)
        class_map = load_classes(PROJECT_ROOT / config["classes_config"])
        source = config["source"]
        split_cfg = config["split"]
        ann_cfg = config["annotations"]
        metadata = prepare(
            PROJECT_ROOT / source["raw_dir"],
            PROJECT_ROOT / config["output"]["dir"],
            class_map,
            train_annotations=source["train_annotations"],
            val_annotations=source["val_annotations"],
            seed=int(split_cfg["seed"]),
            val_fraction=float(split_cfg["val_fraction"]),
            include_iscrowd=bool(ann_cfg["include_iscrowd"]),
            min_box_size=float(ann_cfg["min_box_size_px"]),
        )
    except (OSError, ValueError, json.JSONDecodeError, KeyError, PrepareError) as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1

    for split in SPLITS:
        stats = metadata["splits"][split]
        print(
            f"[ok] {split}: {stats['images']} images, {stats['instances']} instances"
        )
    total_images = sum(metadata["splits"][split]["images"] for split in SPLITS)
    total_instances = sum(metadata["splits"][split]["instances"] for split in SPLITS)
    print(
        f"[ok] total: {total_images} images, {total_instances} instances,"
        f" {metadata['bytes_copied']} bytes copied"
    )
    print(
        "[ok] rejected annotations:"
        f" {json.dumps(metadata['rejected_annotations'], sort_keys=True)}"
    )
    print(f"[ok] wrote {config['output']['dir']}/dataset.yaml and metadata.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
