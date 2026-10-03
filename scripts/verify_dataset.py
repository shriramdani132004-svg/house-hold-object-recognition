"""Verify the acquired raw dataset without modifying it.

Run from the project root:

    python scripts/verify_dataset.py [--dest data/raw/coco]

Checks that every expected archive and extracted file exists, that image
counts match the acquisition specification (including train2017 when it
has been acquired), and that the COCO instance annotation JSONs parse and
are internally consistent. Prints the dataset statistics it measured.
Exit code 0 means everything checks out.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _search_root in (PROJECT_ROOT, PROJECT_ROOT / "scripts"):
    if str(_search_root) not in sys.path:
        sys.path.insert(0, str(_search_root))

from download_dataset import (
    ANNOTATIONS,
    RAW_DIR,
    TRAIN2017,
    VAL2017,
    Archive,
    DatasetError,
    check_size,
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def default_archives(dest_dir: Path) -> list[Archive]:
    archives: list[Archive] = [ANNOTATIONS, VAL2017]
    if (dest_dir / TRAIN2017.relative_path).is_file():
        archives.append(TRAIN2017)
    return archives


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"{path.name}: expected a JSON object")
    return loaded


def _count_images(folder: Path) -> int:
    if not folder.is_dir():
        return 0
    return sum(1 for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def _check_extractions(
    archive: Archive, dest_dir: Path, problems: list[str], stats: dict[str, object]
) -> None:
    if archive.required_members:
        for member in archive.required_members:
            path = dest_dir / member
            if not path.is_file() or path.stat().st_size == 0:
                problems.append(f"missing extracted file: {member}")
    if archive.image_prefix and archive.image_count is not None:
        folder = dest_dir / archive.image_prefix.rstrip("/")
        count = _count_images(folder)
        stats[f"image_files:{archive.image_prefix.rstrip('/')}"] = count
        if count != archive.image_count:
            problems.append(
                f"{archive.image_prefix}: expected {archive.image_count} images, found {count}"
            )


def verify(
    dest_dir: Path,
    archives: Sequence[Archive] = (ANNOTATIONS, VAL2017),
) -> tuple[list[str], dict[str, object]]:
    problems: list[str] = []
    stats: dict[str, object] = {}

    for archive in archives:
        try:
            check_size(dest_dir / archive.relative_path, archive.size_bytes)
        except DatasetError as exc:
            problems.append(str(exc))
        _check_extractions(archive, dest_dir, problems, stats)

    val_json = dest_dir / "annotations" / "instances_val2017.json"
    train_json = dest_dir / "annotations" / "instances_train2017.json"

    categories: list[dict] = []
    for label, path in (("val", val_json), ("train", train_json)):
        if not path.is_file() or path.stat().st_size == 0:
            problems.append(f"missing annotation file: {path}")
            continue
        try:
            data = _load_json(path)
        except (json.JSONDecodeError, ValueError) as exc:
            problems.append(f"{path.name}: invalid JSON ({exc})")
            continue
        images = data.get("images")
        annotations = data.get("annotations")
        cats = data.get("categories")
        if not isinstance(images, list) or not images:
            problems.append(f"{path.name}: 'images' missing or empty")
        if not isinstance(annotations, list) or not annotations:
            problems.append(f"{path.name}: 'annotations' missing or empty")
        if not isinstance(cats, list) or not cats:
            problems.append(f"{path.name}: 'categories' missing or empty")
            continue
        stats[f"images_{label}"] = len(images) if isinstance(images, list) else None
        stats[f"instances_{label}"] = (
            len(annotations) if isinstance(annotations, list) else None
        )
        if label == "val":
            categories = cats
            stats["categories"] = len(cats)
            names = [c.get("name", "?") for c in cats]
            stats["category_names"] = names
            if isinstance(images, list):
                expected = next(
                    (
                        a.image_count
                        for a in archives
                        if a.image_prefix == "val2017/"
                    ),
                    None,
                )
                if expected is not None and len(images) != expected:
                    problems.append(
                        f"instances_val2017.json lists {len(images)} images, "
                        f"expected {expected}"
                    )
        else:
            stats["categories_train"] = len(cats)
            if categories and len(cats) != len(categories):
                problems.append(
                    f"category count differs between val ({len(categories)}) "
                    f"and train ({len(cats)})"
                )
            expected = next(
                (
                    a.image_count
                    for a in archives
                    if a.image_prefix == "train2017/"
                ),
                None,
            )
            if (
                expected is not None
                and isinstance(images, list)
                and len(images) != expected
            ):
                problems.append(
                    f"instances_train2017.json lists {len(images)} images, "
                    f"expected {expected}"
                )

    return problems, stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dest",
        type=Path,
        default=RAW_DIR,
        help="dataset directory to verify (default: data/raw/coco)",
    )
    args = parser.parse_args(argv)

    if not args.dest.is_dir():
        print(f"[FAIL] dataset directory does not exist: {args.dest}", file=sys.stderr)
        return 1

    archives = default_archives(args.dest)
    problems, stats = verify(args.dest, archives)

    for key in sorted(stats):
        if key == "category_names":
            continue
        print(f"[stat] {key}: {stats[key]}")
    names = stats.get("category_names")
    if names:
        print(f"[stat] category_names: {', '.join(str(n) for n in names)}")

    if problems:
        for problem in problems:
            print(f"[FAIL] {problem}", file=sys.stderr)
        print(f"Verification FAILED with {len(problems)} problem(s).", file=sys.stderr)
        return 1

    print("[PASS] dataset verified successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
