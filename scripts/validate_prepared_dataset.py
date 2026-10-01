"""Validate the prepared YOLO household-object dataset.

Run from the project root:

    python scripts/validate_prepared_dataset.py

Checks dataset.yaml, image/label correspondence for every split, label
syntax, class-id validity, coordinate ranges, image readability, unique
filenames across splits, and class coverage. Writes
reports/phase3_dataset_validation.md. Exit code 0 means PASS.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "dataset.yaml"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "household_objects"
REPORT_PATH = PROJECT_ROOT / "reports" / "phase3_dataset_validation.md"
SPLITS = ("train", "val", "test")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
MAX_LISTED = 20


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    return data


def load_classes(path: Path) -> dict[int, str]:
    data = load_yaml(path)
    raw = data.get("classes")
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"{path}: 'classes' must be a non-empty mapping")
    class_map = {int(key): str(value) for key, value in raw.items()}
    if sorted(class_map) != list(range(len(class_map))):
        raise ValueError(f"{path}: class ids must be consecutive starting at 0")
    return class_map


def parse_label(
    path: Path, class_count: int
) -> tuple[list[tuple[int, float, float, float, float]], list[str]]:
    problems: list[str] = []
    boxes: list[tuple[int, float, float, float, float]] = []
    content = path.read_text(encoding="utf-8")
    if not content.strip():
        return [], [f"{path.name}: label file is empty"]
    for line_number, line in enumerate(content.splitlines(), start=1):
        parts = line.split()
        where = f"{path.name}:{line_number}"
        if len(parts) != 5:
            problems.append(f"{where}: expected 5 fields, got {len(parts)}")
            continue
        try:
            class_id = int(parts[0])
        except ValueError:
            problems.append(f"{where}: class id is not an integer: {parts[0]!r}")
            continue
        if class_id < 0 or class_id >= class_count:
            problems.append(
                f"{where}: class id {class_id} outside [0, {class_count - 1}]"
            )
            continue
        try:
            xc, yc, width, height = (float(value) for value in parts[1:])
        except ValueError:
            problems.append(f"{where}: non-numeric coordinate in {line!r}")
            continue
        values = (xc, yc, width, height)
        if not all(math.isfinite(value) for value in values):
            problems.append(f"{where}: non-finite coordinate in {line!r}")
            continue
        if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0):
            problems.append(f"{where}: center outside [0, 1]: {line!r}")
            continue
        if width <= 0.0 or height <= 0.0:
            problems.append(f"{where}: non-positive size: {line!r}")
            continue
        if width > 1.0 or height > 1.0:
            problems.append(f"{where}: size above 1.0: {line!r}")
            continue
        boxes.append((class_id, xc, yc, width, height))
    return boxes, problems


def _check_dataset_yaml(
    output_dir: Path, class_map: dict[int, str], problems: list[str]
) -> dict[str, Any]:
    path = output_dir / "dataset.yaml"
    if not path.is_file():
        problems.append("missing dataset.yaml")
        return {}
    try:
        data = load_yaml(path)
    except ValueError as exc:
        problems.append(str(exc))
        return {}
    for key in ("path", "train", "val", "test", "nc", "names"):
        if key not in data:
            problems.append(f"dataset.yaml: missing key '{key}'")
    names = data.get("names")
    if not isinstance(names, list):
        problems.append("dataset.yaml: 'names' must be a list")
        names = []
    expected = [class_map[index] for index in range(len(class_map))]
    if names != expected:
        problems.append("dataset.yaml: names do not match configs/classes.yaml order")
    if data.get("nc") != len(class_map):
        problems.append(
            f"dataset.yaml: nc={data.get('nc')} != {len(class_map)}"
        )
    return data


def validate(
    output_dir: Path,
    class_map: dict[int, str],
    verify_images: bool = True,
) -> tuple[list[str], dict[str, Any]]:
    problems: list[str] = []
    warnings: list[str] = []
    stats: dict[str, Any] = {"splits": {}, "class_images": {}, "class_instances": {}}

    dataset_yaml = _check_dataset_yaml(output_dir, class_map, problems)
    class_count = len(class_map)
    stems_by_split: dict[str, set[str]] = {}
    used_ids: dict[str, set[int]] = {}

    for split in SPLITS:
        images_dir = output_dir / "images" / split
        labels_dir = output_dir / "labels" / split
        if not images_dir.is_dir() or not labels_dir.is_dir():
            problems.append(f"{split}: missing images/ or labels/ directory")
            continue
        image_files = sorted(
            p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES
        )
        label_files = sorted(labels_dir.glob("*.txt"))
        image_stems = {p.stem for p in image_files}
        label_stems = {p.stem for p in label_files}
        stems_by_split[split] = image_stems | label_stems

        for stem in sorted(image_stems - label_stems):
            problems.append(f"{split}: image without label file: {stem}")
        for stem in sorted(label_stems - image_stems):
            problems.append(f"{split}: label without image file: {stem}")

        instances = 0
        per_class_counts = {index: 0 for index in range(class_count)}
        per_class_images = {index: 0 for index in range(class_count)}
        broken_images = 0
        for image_path in image_files:
            if image_path.stat().st_size == 0:
                problems.append(f"{split}: empty image file: {image_path.name}")
                broken_images += 1
                continue
            if verify_images:
                try:
                    with Image.open(image_path) as image:
                        image.verify()
                except Exception as exc:
                    problems.append(
                        f"{split}: unreadable image {image_path.name}: {exc}"
                    )
                    broken_images += 1

        label_problems = 0
        for label_path in label_files:
            boxes, label_issues = parse_label(label_path, class_count)
            for issue in label_issues:
                if label_problems < MAX_LISTED:
                    problems.append(f"{split}: {issue}")
                label_problems += 1
            if boxes:
                seen_ids: set[int] = set()
                for class_id, *_ in boxes:
                    per_class_counts[class_id] += 1
                    seen_ids.add(class_id)
                    instances += 1
                for class_id in seen_ids:
                    per_class_images[class_id] += 1
            if label_path.stem in image_stems and not boxes:
                problems.append(f"{split}: label has no valid boxes: {label_path.name}")

        if label_problems > MAX_LISTED:
            problems.append(
                f"{split}: {label_problems - MAX_LISTED} further label problems omitted"
            )
        used_ids[split] = set(per_class_images)
        stats["splits"][split] = {
            "images": len(image_files),
            "labels": len(label_files),
            "instances": instances,
            "broken_images": broken_images,
            "label_problems": label_problems,
        }
        stats["class_images"][split] = per_class_images
        stats["class_instances"][split] = per_class_counts

    for first_index, first in enumerate(SPLITS):
        for second in SPLITS[first_index + 1 :]:
            shared = stems_by_split.get(first, set()) & stems_by_split.get(
                second, set()
            )
            for stem in sorted(shared):
                problems.append(f"file name appears in both {first} and {second}: {stem}")

    overall_used = set().union(*used_ids.values()) if used_ids else set()
    missing_overall = sorted(
        set(range(class_count)) - overall_used, key=lambda index: class_map[index]
    )
    for class_id in missing_overall:
        problems.append(f"class never appears in any split: {class_map[class_id]}")
    for split in SPLITS:
        missing = sorted(
            set(range(class_count)) - used_ids.get(split, set()),
            key=lambda index: class_map[index],
        )
        if missing:
            warnings.append(
                f"{split}: {len(missing)} class(es) absent:"
                f" {', '.join(class_map[index] for index in missing)}"
            )

    total_images = sum(s["images"] for s in stats["splits"].values())
    total_instances = sum(s["instances"] for s in stats["splits"].values())
    stats["totals"] = {"images": total_images, "instances": total_instances}
    stats["warnings"] = warnings
    stats["dataset_yaml"] = dataset_yaml
    return problems, stats


def render_report(
    stats: dict[str, Any], problems: list[str], class_map: dict[int, str]
) -> str:
    status = "PASS" if not problems else "FAIL"
    lines = [
        "# Phase 3 — Prepared Dataset Validation",
        "",
        f"**Result: {status}**",
        "",
        "Generated by `python scripts/validate_prepared_dataset.py`.",
        "",
        "## Split counts",
        "",
        "| Split | Images | Labels | Instances | Broken images | Label problems |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for split in SPLITS:
        entry = stats["splits"].get(split, {})
        lines.append(
            f"| {split} | {entry.get('images', 0):,} | {entry.get('labels', 0):,} |"
            f" {entry.get('instances', 0):,} | {entry.get('broken_images', 0)} |"
            f" {entry.get('label_problems', 0)} |"
        )
    totals = stats["totals"]
    lines.append(
        f"| **total** | **{totals['images']:,}** | **{totals['images']:,}** |"
        f" **{totals['instances']:,}** | | |"
    )
    lines += [
        "",
        "## Class distribution",
        "",
        "| ID | Class | Train inst | Val inst | Test inst | Train img | Val img | Test img |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for class_id in range(len(class_map)):
        lines.append(
            f"| {class_id} | {class_map[class_id]} |"
            f" {stats['class_instances'].get('train', {}).get(class_id, 0):,} |"
            f" {stats['class_instances'].get('val', {}).get(class_id, 0):,} |"
            f" {stats['class_instances'].get('test', {}).get(class_id, 0):,} |"
            f" {stats['class_images'].get('train', {}).get(class_id, 0):,} |"
            f" {stats['class_images'].get('val', {}).get(class_id, 0):,} |"
            f" {stats['class_images'].get('test', {}).get(class_id, 0):,} |"
        )
    lines += ["", "## Checks performed", ""]
    for check in (
        "dataset.yaml parses; nc and names match configs/classes.yaml",
        "every image has a label file and every label has an image",
        "label syntax: 5 numeric fields, finite, normalized, positive size",
        "class ids inside [0, nc)",
        "image files non-empty and decodable (Pillow verify)",
        "file names unique across train/val/test",
        "every selected class appears in at least one split",
    ):
        lines.append(f"- {check}")
    if stats["warnings"]:
        lines += ["", "## Warnings", ""]
        lines += [f"- {warning}" for warning in stats["warnings"]]
    if problems:
        lines += ["", "## Problems", ""]
        lines += [f"- {problem}" for problem in problems]
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--no-image-verify",
        action="store_true",
        help="skip decoding every image (faster, weaker)",
    )
    args = parser.parse_args(argv)

    try:
        config = load_yaml(args.config)
        class_map = load_classes(PROJECT_ROOT / config["classes_config"])
        output_dir = args.output or PROJECT_ROOT / config["output"]["dir"]
    except (OSError, ValueError, KeyError) as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1

    if not output_dir.is_dir():
        print(f"[FAIL] output directory not found: {output_dir}", file=sys.stderr)
        return 1

    problems, stats = validate(
        output_dir, class_map, verify_images=not args.no_image_verify
    )
    report = render_report(stats, problems, class_map)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")

    for split in SPLITS:
        entry = stats["splits"].get(split, {})
        print(
            f"[stat] {split}: {entry.get('images', 0)} images,"
            f" {entry.get('instances', 0)} instances"
        )
    print(f"[stat] totals: {stats['totals']}")
    for warning in stats["warnings"]:
        print(f"[warn] {warning}")
    print(f"[ok] wrote {REPORT_PATH.relative_to(PROJECT_ROOT)}")

    if problems:
        for problem in problems[:MAX_LISTED]:
            print(f"[FAIL] {problem}", file=sys.stderr)
        if len(problems) > MAX_LISTED:
            print(
                f"[FAIL] ... {len(problems) - MAX_LISTED} more problems",
                file=sys.stderr,
            )
        print(
            f"Validation FAILED with {len(problems)} problem(s).",
            file=sys.stderr,
        )
        return 1

    print("[PASS] prepared dataset validated successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
