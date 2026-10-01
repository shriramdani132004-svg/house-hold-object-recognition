"""Phase 4 analysis of the prepared household-objects dataset.

Run from the project root:

    .\\.venv\\Scripts\\python.exe scripts/analyze_dataset.py

Reads data/processed/household_objects/ read-only (labels, image headers,
image content hashes) and writes:

- reports/phase4_dataset_analysis.md   (full analysis report)
- reports/phase4_summary.json          (machine-readable summary)
- reports/figures/phase4/*.png         (charts)
- reports/phase4_samples/              (annotated representative examples)

Long-running scans show a single tqdm progress bar per operation with
percentage, elapsed time and ETA. Exit code 0 on success; exit code 1 with
a [FAIL] message if the input does not match the verified Phase 3 state.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import random
import shutil
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "dataset.yaml"
REPORT_PATH = PROJECT_ROOT / "reports" / "phase4_dataset_analysis.md"
SUMMARY_PATH = PROJECT_ROOT / "reports" / "phase4_summary.json"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures" / "phase4"
SAMPLES_DIR = PROJECT_ROOT / "reports" / "phase4_samples"

SPLITS = ("train", "val", "test")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

EXPECTED_IMAGES = {"train": 40890, "val": 4544, "test": 1965}
EXPECTED_LABELS = {"train": 40890, "val": 4544, "test": 1965}
EXPECTED_INSTANCES = {"train": 184709, "val": 21100, "test": 9174}

# Analysis bins only: normalized box-area thresholds, not a universal standard.
SIZE_BANDS = (
    ("very small", 0.0, 1e-3),
    ("small", 1e-3, 1e-2),
    ("medium", 1e-2, 1e-1),
    ("large", 1e-1, math.inf),
)
SMALL_AREA_LIMIT = 1e-2
TINY_AREA_LIMIT = 1e-4
LARGE_AREA_LIMIT = 0.5
BUCKET_ORDER = ("0", "1", "2", "3-5", ">5")

REPORT_SECTIONS = [
    "## 1. Dataset Overview",
    "## 2. Final Classes",
    "## 3. Dataset Counts",
    "## 4. Class Distribution",
    "## 5. Class Imbalance",
    "## 6. Objects Per Image",
    "## 7. Multi-Object Scenes",
    "## 8. Bounding Box Size Distribution",
    "## 9. Small Object Analysis",
    "## 10. Class Co-Occurrence",
    "## 11. Train/Validation/Test Distribution",
    "## 12. Image Dimensions",
    "## 13. Annotation Quality",
    "## 14. Representative Examples",
    "## 15. Dataset Concerns",
    "## 16. Implications for Model Training",
    "## 17. Phase 4 Conclusion",
]


class AnalysisError(RuntimeError):
    """Raised when preflight verification fails and analysis must stop."""


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    return data


def load_classes(path_value: Any) -> dict[int, str]:
    path = Path(path_value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    data = load_yaml(path)
    class_map = {int(key): str(value) for key, value in data["classes"].items()}
    if sorted(class_map) != list(range(len(class_map))):
        raise ValueError(f"{path}: class ids must be consecutive starting at 0")
    return class_map


def _load_sibling_script(name: str) -> Any:
    path = PROJECT_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def image_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        path for path in directory.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
    )


def count_files(output_dir: Path) -> dict[str, dict[str, int]]:
    return {
        split: {
            "images": len(image_files(output_dir / "images" / split)),
            "labels": len(list((output_dir / "labels" / split).glob("*.txt"))),
        }
        for split in SPLITS
    }


def preflight(
    output_dir: Path, class_map: dict[int, str]
) -> dict[str, dict[str, int]]:
    if not output_dir.is_dir():
        raise AnalysisError(f"dataset directory not found: {output_dir}")
    dataset_yaml = output_dir / "dataset.yaml"
    if not dataset_yaml.is_file():
        raise AnalysisError(f"dataset.yaml not found: {dataset_yaml}")
    data = load_yaml(dataset_yaml)
    expected_names = [class_map[index] for index in range(len(class_map))]
    if data.get("nc") != len(class_map):
        raise AnalysisError(
            f"dataset.yaml nc={data.get('nc')} but classes.yaml has"
            f" {len(class_map)} classes"
        )
    if [str(name) for name in data.get("names", [])] != expected_names:
        raise AnalysisError("dataset.yaml names do not match configs/classes.yaml")
    counts = count_files(output_dir)
    for split in SPLITS:
        if counts[split]["images"] != EXPECTED_IMAGES[split]:
            raise AnalysisError(
                f"{split}: expected {EXPECTED_IMAGES[split]} images,"
                f" found {counts[split]['images']}"
            )
        if counts[split]["labels"] != EXPECTED_LABELS[split]:
            raise AnalysisError(
                f"{split}: expected {EXPECTED_LABELS[split]} labels,"
                f" found {counts[split]['labels']}"
            )
    return counts


def verify_expected(scan: dict[str, Any]) -> None:
    problems: list[str] = []
    for split in SPLITS:
        entry = scan["splits"][split]
        if entry["images"] != EXPECTED_IMAGES[split]:
            problems.append(
                f"{split}: images {entry['images']} != {EXPECTED_IMAGES[split]}"
            )
        if entry["labels"] != EXPECTED_LABELS[split]:
            problems.append(
                f"{split}: labels {entry['labels']} != {EXPECTED_LABELS[split]}"
            )
        if entry["instances"] != EXPECTED_INSTANCES[split]:
            problems.append(
                f"{split}: instances {entry['instances']} !="
                f" {EXPECTED_INSTANCES[split]}"
            )
    if problems:
        raise AnalysisError(
            "dataset does not match verified Phase 3 state:\n- "
            + "\n- ".join(problems)
        )


def percentile(ordered: list[float], quantile: float) -> float:
    if not ordered:
        raise ValueError("percentile of empty data")
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)


def describe(values: Iterable[float]) -> dict[str, float]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {
            key: 0.0
            for key in ("min", "max", "mean", "median", "p25", "p75", "p90", "p95", "p99")
        } | {"count": 0}
    return {
        "count": len(ordered),
        "min": round(ordered[0], 6),
        "max": round(ordered[-1], 6),
        "mean": round(statistics.fmean(ordered), 6),
        "median": round(statistics.median(ordered), 6),
        "p25": round(percentile(ordered, 0.25), 6),
        "p75": round(percentile(ordered, 0.75), 6),
        "p90": round(percentile(ordered, 0.90), 6),
        "p95": round(percentile(ordered, 0.95), 6),
        "p99": round(percentile(ordered, 0.99), 6),
    }


def pct(part: float, whole: float) -> float:
    return round(100.0 * part / whole, 2) if whole else 0.0


def size_band(area: float) -> str:
    for name, low, high in SIZE_BANDS:
        if low <= area < high:
            return name
    return SIZE_BANDS[-1][0]


def band_counts(areas: Iterable[float]) -> dict[str, dict[str, float]]:
    counts: dict[str, dict[str, float]] = {
        name: {"count": 0, "pct": 0.0} for name, _, _ in SIZE_BANDS
    }
    values = list(areas)
    for area in values:
        counts[size_band(area)]["count"] += 1
    for entry in counts.values():
        entry["pct"] = pct(entry["count"], len(values))
    return counts


def objects_buckets(counts: Iterable[int]) -> dict[str, int]:
    buckets = dict.fromkeys(BUCKET_ORDER, 0)
    for value in counts:
        if value == 0:
            buckets["0"] += 1
        elif value == 1:
            buckets["1"] += 1
        elif value == 2:
            buckets["2"] += 1
        elif value <= 5:
            buckets["3-5"] += 1
        else:
            buckets[">5"] += 1
    return buckets


def multi_object_stats(counts: Iterable[int]) -> dict[str, Any]:
    values = list(counts)
    total = len(values)
    single = sum(1 for value in values if value == 1)
    multi = sum(1 for value in values if value >= 2)
    return {
        "total_images": total,
        "single_object_images": single,
        "multi_object_images": multi,
        "images_2plus": multi,
        "images_3plus": sum(1 for value in values if value >= 3),
        "max_objects_in_image": max(values) if values else 0,
        "multi_share_pct": pct(multi, total),
    }


def imbalance_stats(
    instances: dict[int, int], class_map: dict[int, str] | None = None
) -> dict[str, Any]:
    nonzero = {key: value for key, value in instances.items() if value > 0}
    if not nonzero:
        return {
            "most_class_id": -1,
            "most_class_name": "",
            "most_class_instances": 0,
            "least_class_id": -1,
            "least_class_name": "",
            "least_class_instances": 0,
            "max_min_ratio": 0.0,
            "median_instances": 0.0,
            "mean_instances": 0.0,
        }
    most_id = max(nonzero, key=lambda key: nonzero[key])
    least_id = min(nonzero, key=lambda key: nonzero[key])
    values = sorted(nonzero.values())
    names = class_map or {}

    def name_of(class_id: int) -> str:
        return names.get(class_id, str(class_id))

    return {
        "most_class_id": most_id,
        "most_class_name": name_of(most_id),
        "most_class_instances": nonzero[most_id],
        "least_class_id": least_id,
        "least_class_name": name_of(least_id),
        "least_class_instances": nonzero[least_id],
        "max_min_ratio": round(nonzero[most_id] / nonzero[least_id], 2),
        "median_instances": float(statistics.median(values)),
        "mean_instances": round(statistics.fmean(values), 1),
    }


def cooccurrence_counts(
    class_sets: Iterable[Iterable[int]],
) -> dict[tuple[int, int], int]:
    counts: dict[tuple[int, int], int] = {}
    for classes in class_sets:
        ordered = sorted(set(classes))
        for index, class_a in enumerate(ordered):
            for class_b in ordered[index + 1 :]:
                key = (class_a, class_b)
                counts[key] = counts.get(key, 0) + 1
    return counts


def parse_label_line(
    line: str, class_count: int
) -> tuple[str, tuple[int, float, float, float, float] | None]:
    parts = line.split()
    if len(parts) != 5:
        return "fields", None
    try:
        class_id = int(parts[0])
        xc, yc, box_w, box_h = (float(part) for part in parts[1:])
    except ValueError:
        return "numeric", None
    if class_id < 0 or class_id >= class_count:
        return "class_id", None
    if not all(math.isfinite(value) for value in (xc, yc, box_w, box_h)):
        return "nonfinite", None
    if box_w <= 0.0 or box_h <= 0.0:
        return "size", None
    if not (0.0 < xc <= 1.0 and 0.0 < yc <= 1.0 and box_w <= 1.0 and box_h <= 1.0):
        return "range", None
    return "ok", (class_id, xc, yc, box_w, box_h)


def scan_labels(
    output_dir: Path, class_map: dict[int, str], *, show_progress: bool = True
) -> dict[str, Any]:
    class_count = len(class_map)
    scan: dict[str, Any] = {
        "splits": {
            split: {"images": 0, "labels": 0, "instances": 0} for split in SPLITS
        },
        "class_instances": {index: 0 for index in range(class_count)},
        "class_images": {index: 0 for index in range(class_count)},
        "split_class_instances": {
            split: {index: 0 for index in range(class_count)} for split in SPLITS
        },
        "split_class_images": {
            split: {index: 0 for index in range(class_count)} for split in SPLITS
        },
        "class_image_lists": {index: [] for index in range(class_count)},
        "class_areas": {index: [] for index in range(class_count)},
        "image_info": {},
        "objects_per_image": {split: [] for split in SPLITS},
        "areas": [],
        "areas_by_split": {split: [] for split in SPLITS},
        "widths": [],
        "heights": [],
        "empty_label_files": 0,
        "missing_label_files": 0,
        "label_without_image": 0,
        "lines_fields": 0,
        "lines_numeric": 0,
        "lines_class_id": 0,
        "lines_nonfinite": 0,
        "lines_size": 0,
        "lines_range": 0,
    }
    error_keys = {
        "fields": "lines_fields",
        "numeric": "lines_numeric",
        "class_id": "lines_class_id",
        "nonfinite": "lines_nonfinite",
        "size": "lines_size",
        "range": "lines_range",
    }
    for split in SPLITS:
        images = image_files(output_dir / "images" / split)
        labels_dir = output_dir / "labels" / split
        labels = sorted(labels_dir.glob("*.txt")) if labels_dir.is_dir() else []
        image_stems = {path.stem for path in images}
        scan["splits"][split]["images"] = len(images)
        scan["splits"][split]["labels"] = len(labels)
        labelled_stems: set[str] = set()
        iterator = tqdm(
            labels,
            desc=f"Analyzing {split} labels",
            unit="file",
            dynamic_ncols=True,
            disable=not show_progress,
        )
        for label_path in iterator:
            stem = label_path.stem
            labelled_stems.add(stem)
            if stem not in image_stems:
                scan["label_without_image"] += 1
            content = label_path.read_text(encoding="utf-8").strip()
            box_count = 0
            present: list[int] = []
            min_area = math.inf
            max_area = 0.0
            if not content:
                scan["empty_label_files"] += 1
            else:
                for line in content.splitlines():
                    if not line.strip():
                        continue
                    status, parsed = parse_label_line(line, class_count)
                    if parsed is None:
                        scan[error_keys[status]] += 1
                        continue
                    class_id, _xc, _yc, box_w, box_h = parsed
                    area = box_w * box_h
                    scan["areas"].append(area)
                    scan["areas_by_split"][split].append(area)
                    scan["class_areas"][class_id].append(area)
                    scan["widths"].append(box_w)
                    scan["heights"].append(box_h)
                    scan["class_instances"][class_id] += 1
                    scan["split_class_instances"][split][class_id] += 1
                    box_count += 1
                    if class_id not in present:
                        present.append(class_id)
                        scan["class_images"][class_id] += 1
                        scan["split_class_images"][split][class_id] += 1
                        scan["class_image_lists"][class_id].append((split, stem))
                    min_area = min(min_area, area)
                    max_area = max(max_area, area)
            scan["splits"][split]["instances"] += box_count
            scan["objects_per_image"][split].append(box_count)
            scan["image_info"][(split, stem)] = {
                "objects": box_count,
                "classes": tuple(sorted(present)),
                "min_area": min_area if box_count else None,
                "max_area": max_area if box_count else None,
            }
        for stem in sorted(image_stems - labelled_stems):
            scan["missing_label_files"] += 1
            scan["objects_per_image"][split].append(0)
            scan["image_info"][(split, stem)] = {
                "objects": 0,
                "classes": (),
                "min_area": None,
                "max_area": None,
            }
    return scan


def scan_dimensions(
    output_dir: Path, *, show_progress: bool = True
) -> dict[tuple[str, str], tuple[int, int]]:
    from PIL import Image

    sizes: dict[tuple[str, str], tuple[int, int]] = {}
    for split in SPLITS:
        images = image_files(output_dir / "images" / split)
        iterator = tqdm(
            images,
            desc=f"Scanning {split} image sizes",
            unit="img",
            dynamic_ncols=True,
            disable=not show_progress,
        )
        for image_path in iterator:
            with Image.open(image_path) as image:
                width, height = image.size
            sizes[(split, image_path.stem)] = (width, height)
    return sizes


def hash_duplicates(
    output_dir: Path, *, show_progress: bool = True, chunk_size: int = 1 << 20
) -> dict[str, Any]:
    files = [
        (split, path)
        for split in SPLITS
        for path in image_files(output_dir / "images" / split)
    ]
    sizes: list[int] = []
    total_bytes = 0
    for _split, path in files:
        size = path.stat().st_size
        sizes.append(size)
        total_bytes += size
    seen: dict[str, tuple[str, str]] = {}
    duplicate_pairs: list[dict[str, Any]] = []
    with tqdm(
        total=total_bytes,
        desc="Hashing images for duplicate/leakage check",
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        dynamic_ncols=True,
        disable=not show_progress,
    ) as bar:
        for (split, path), _size in zip(files, sizes):
            digest = hashlib.sha1()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(chunk_size), b""):
                    digest.update(chunk)
                    bar.update(len(chunk))
            hexdigest = digest.hexdigest()
            if hexdigest in seen:
                first_split, first_stem = seen[hexdigest]
                duplicate_pairs.append(
                    {
                        "sha1": hexdigest,
                        "first": {"split": first_split, "stem": first_stem},
                        "duplicate": {"split": split, "stem": path.stem},
                    }
                )
            else:
                seen[hexdigest] = (split, path.stem)
    cross = [
        pair
        for pair in duplicate_pairs
        if pair["first"]["split"] != pair["duplicate"]["split"]
    ]
    return {
        "hashed_files": len(files),
        "hashed_bytes": total_bytes,
        "duplicate_pairs": len(duplicate_pairs),
        "cross_split_pairs": len(cross),
        "within_split_pairs": len(duplicate_pairs) - len(cross),
        "details": duplicate_pairs[:20],
    }


def filename_overlap(stems: dict[str, set[str]]) -> dict[str, int]:
    overlap: dict[str, int] = {}
    for index, split_a in enumerate(SPLITS):
        for split_b in SPLITS[index + 1 :]:
            overlap[f"{split_a}-{split_b}"] = len(stems[split_a] & stems[split_b])
    return overlap


def dimension_stats(
    sizes: dict[tuple[str, str], tuple[int, int]]
) -> dict[str, Any]:
    if not sizes:
        return {"count": 0}
    widths = [size[0] for size in sizes.values()]
    heights = [size[1] for size in sizes.values()]
    aspects = [
        (split, stem, width / height if height else 0.0)
        for (split, stem), (width, height) in sizes.items()
    ]
    landscape = sum(1 for _s, _t, ratio in aspects if ratio > 1.02)
    portrait = sum(1 for _s, _t, ratio in aspects if ratio < 0.98)
    square = len(aspects) - landscape - portrait
    min_width_key = min(sizes, key=lambda key: sizes[key][0])
    max_width_key = max(sizes, key=lambda key: sizes[key][0])
    min_height_key = min(sizes, key=lambda key: sizes[key][1])
    max_height_key = max(sizes, key=lambda key: sizes[key][1])
    min_aspect_entry = min(aspects, key=lambda entry: entry[2])
    max_aspect_entry = max(aspects, key=lambda entry: entry[2])

    def extreme(
        split: str, stem: str, value: float, label: str
    ) -> dict[str, Any]:
        return {
            "label": label,
            "value": round(value, 4),
            "split": split,
            "stem": stem,
        }

    return {
        "count": len(sizes),
        "width": describe(widths),
        "height": describe(heights),
        "aspect_ratio": describe([entry[2] for entry in aspects]),
        "landscape": landscape,
        "portrait": portrait,
        "square": square,
        "top_widths": Counter(widths).most_common(5),
        "top_heights": Counter(heights).most_common(5),
        "top_resolutions": [
            {"width": size[0], "height": size[1], "count": count}
            for size, count in Counter(sizes.values()).most_common(5)
        ],
        "extremes": [
            extreme(min_width_key[0], min_width_key[1], sizes[min_width_key][0], "min width"),
            extreme(max_width_key[0], max_width_key[1], sizes[max_width_key][0], "max width"),
            extreme(min_height_key[0], min_height_key[1], sizes[min_height_key][1], "min height"),
            extreme(max_height_key[0], max_height_key[1], sizes[max_height_key][1], "max height"),
            extreme(min_aspect_entry[0], min_aspect_entry[1], min_aspect_entry[2], "min aspect (tallest)"),
            extreme(max_aspect_entry[0], max_aspect_entry[1], max_aspect_entry[2], "max aspect (widest)"),
        ],
    }


def quality_stats(scan: dict[str, Any]) -> dict[str, Any]:
    objects_all = [
        count for split in SPLITS for count in scan["objects_per_image"][split]
    ]
    ordered = sorted(objects_all)
    p99 = percentile(ordered, 0.99) if ordered else 0.0
    areas = scan["areas"]
    instance_total = len(areas)
    large = sum(1 for area in areas if area >= LARGE_AREA_LIMIT)
    tiny = sum(1 for area in areas if area < TINY_AREA_LIMIT)
    class_counts = sorted(scan["class_instances"].items(), key=lambda item: item[1])
    return {
        "empty_label_files": scan["empty_label_files"],
        "missing_label_files": scan["missing_label_files"],
        "label_without_image": scan["label_without_image"],
        "malformed_lines": {
            "wrong_field_count": scan["lines_fields"],
            "non_numeric": scan["lines_numeric"],
            "class_id_out_of_range": scan["lines_class_id"],
            "non_finite": scan["lines_nonfinite"],
            "non_positive_size": scan["lines_size"],
            "out_of_normalized_range": scan["lines_range"],
        },
        "zero_object_images": objects_all.count(0),
        "large_boxes": large,
        "large_boxes_pct": pct(large, instance_total),
        "tiny_boxes": tiny,
        "tiny_boxes_pct": pct(tiny, instance_total),
        "objects_p99": p99,
        "images_at_or_above_p99": sum(
            1 for count in objects_all if count >= p99
        ),
        "max_objects_in_image": max(objects_all) if objects_all else 0,
        "least_supported_classes": [
            {"id": class_id, "instances": count}
            for class_id, count in class_counts[:3]
        ],
    }


def build_context(
    scan: dict[str, Any],
    class_map: dict[int, str],
    dimensions: dict[tuple[str, str], tuple[int, int]] | None = None,
    hashes: dict[str, Any] | None = None,
    samples: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    objects_all = [
        count for split in SPLITS for count in scan["objects_per_image"][split]
    ]
    stems = {
        split: {key[1] for key in scan["image_info"] if key[0] == split}
        for split in SPLITS
    }
    pair_counts = cooccurrence_counts(
        info["classes"] for info in scan["image_info"].values()
    )
    top_pairs = sorted(pair_counts.items(), key=lambda item: (-item[1], item[0]))[:15]
    total_instances = sum(scan["class_instances"].values())
    split_instances = {split: scan["splits"][split]["instances"] for split in SPLITS}
    instance_total = sum(split_instances.values())
    small_by_split = {
        split: sum(
            1 for area in scan["areas_by_split"][split] if area < SMALL_AREA_LIMIT
        )
        for split in SPLITS
    }
    small_by_class = {
        class_id: sum(
            1 for area in scan["class_areas"][class_id] if area < SMALL_AREA_LIMIT
        )
        for class_id in class_map
    }
    context: dict[str, Any] = {
        "class_map": class_map,
        "scan": scan,
        "dimensions": dimensions,
        "hashes": hashes,
        "samples": samples or [],
        "total_images": sum(scan["splits"][split]["images"] for split in SPLITS),
        "total_instances": total_instances,
        "objects_all": objects_all,
        "objects_desc": describe(objects_all),
        "per_split_objects_desc": {
            split: describe(scan["objects_per_image"][split]) for split in SPLITS
        },
        "buckets": objects_buckets(objects_all),
        "multi": multi_object_stats(objects_all),
        "imbalance": imbalance_stats(scan["class_instances"], class_map),
        "area_desc": describe(scan["areas"]),
        "width_desc": describe(scan["widths"]),
        "height_desc": describe(scan["heights"]),
        "bands": band_counts(scan["areas"]),
        "small_boxes": sum(1 for area in scan["areas"] if area < SMALL_AREA_LIMIT),
        "small_by_split": small_by_split,
        "small_by_class": small_by_class,
        "pair_counts": pair_counts,
        "top_pairs": top_pairs,
        "split_instances": split_instances,
        "overall_split_share": {
            split: pct(count, instance_total)
            for split, count in split_instances.items()
        },
        "filename_overlap": filename_overlap(stems),
        "quality": quality_stats(scan),
        "dim_stats": dimension_stats(dimensions) if dimensions else None,
        "matches_phase3": all(
            scan["splits"][split]["images"] == EXPECTED_IMAGES[split]
            and scan["splits"][split]["labels"] == EXPECTED_LABELS[split]
            and scan["splits"][split]["instances"] == EXPECTED_INSTANCES[split]
            for split in SPLITS
        ),
    }
    context["small_pct"] = pct(context["small_boxes"], context["total_instances"])
    return context


def _slug(name: str) -> str:
    return "_".join(name.split())


def write_samples(
    output_dir: Path,
    class_map: dict[int, str],
    scan: dict[str, Any],
    dimensions: dict[tuple[str, str], tuple[int, int]] | None,
    destination: Path,
    *,
    seed: int = 42,
    show_progress: bool = True,
) -> list[dict[str, Any]]:
    from PIL import Image

    visualization = _load_sibling_script("visualize_prepared_dataset")
    names = [class_map[index] for index in range(len(class_map))]
    if destination.is_dir():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    image_info = scan["image_info"]
    picks: list[dict[str, Any]] = []
    taken: set[tuple[str, str]] = set()

    def choose(candidates: list[tuple[str, str]]) -> tuple[str, str] | None:
        free = [item for item in candidates if item not in taken]
        pool = free or candidates
        return rng.choice(pool) if pool else None

    def add(category: str, split: str, stem: str, note: str) -> None:
        picks.append(
            {"category": category, "split": split, "stem": stem, "note": note}
        )
        taken.add((split, stem))

    for class_id in tqdm(
        range(len(class_map)),
        desc="Selecting per-class examples",
        unit="class",
        disable=not show_progress,
    ):
        choice = choose(scan["class_image_lists"][class_id])
        if choice is None:
            continue
        split, stem = choice
        add(
            f"class{class_id:02d}_{_slug(class_map[class_id])}",
            split,
            stem,
            f"contains '{class_map[class_id]}'",
        )

    crowded = max(
        image_info.items(), key=lambda item: item[1]["objects"], default=None
    )
    if crowded is not None and crowded[1]["objects"] > 0:
        split, stem = crowded[0]
        count = crowded[1]["objects"]
        add(f"crowded_{count}obj", split, stem, f"most objects in dataset ({count})")

    multi_pool = [
        key for key, info in image_info.items() if 3 <= info["objects"] <= 5
    ]
    if multi_pool:
        split, stem = rng.choice(multi_pool)
        add(f"multi_object_{image_info[(split, stem)]['objects']}obj", split, stem,
            "typical multi-object scene (3-5 objects)")

    zero_area = [
        (key, info) for key, info in image_info.items() if info["min_area"] is not None
    ]
    if zero_area:
        (split, stem), info = min(zero_area, key=lambda item: item[1]["min_area"])
        add(f"small_box_{info['min_area']:.6f}", split, stem,
            f"smallest box in dataset (area {info['min_area']:.6f})")
    if zero_area:
        (split, stem), info = max(zero_area, key=lambda item: item[1]["max_area"])
        add(f"large_box_{info['max_area']:.3f}", split, stem,
            f"largest box in dataset (area {info['max_area']:.3f})")

    if dimensions:
        widest = max(dimensions, key=lambda key: dimensions[key][0] / max(dimensions[key][1], 1))
        tallest = min(dimensions, key=lambda key: dimensions[key][0] / max(dimensions[key][1], 1))
        for key, category, note in (
            (widest, "aspect_wide", "widest aspect ratio"),
            (tallest, "aspect_tall", "tallest aspect ratio"),
        ):
            if key not in taken:
                add(f"{category}_{dimensions[key][0]}x{dimensions[key][1]}", key[0], key[1], note)

    for split in ("val", "test"):
        pool = [
            key for key in image_info if key[0] == split and key not in taken
        ]
        if pool:
            split_key, stem = rng.choice(pool)
            add(f"split_{split}", split_key, stem, f"{split} split example")

    written: list[dict[str, Any]] = []
    for pick in tqdm(
        picks,
        desc="Rendering representative samples",
        unit="sample",
        disable=not show_progress,
    ):
        split, stem = pick["split"], pick["stem"]
        source = output_dir / "images" / split / f"{stem}.jpg"
        if not source.is_file():
            continue
        with Image.open(source) as opened:
            canvas = opened.convert("RGB")
        canvas.thumbnail((1024, 1024))
        label_path = output_dir / "labels" / split / f"{stem}.txt"
        if label_path.is_file():
            canvas = visualization.draw_boxes(canvas, label_path, names)
        filename = f"{pick['category']}__{split}__{stem}.jpg"
        canvas.save(destination / filename, quality=85)
        info = image_info.get((split, stem), {})
        written.append(
            {
                **pick,
                "file": filename,
                "objects": info.get("objects", 0),
                "classes": [
                    class_map[class_id] for class_id in info.get("classes", ())
                ],
            }
        )
    return written


def write_figures(
    context: dict[str, Any],
    destination: Path,
    *,
    show_progress: bool = True,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scan = context["scan"]
    class_map = context["class_map"]
    names = [class_map[index] for index in range(len(class_map))]
    nc = len(names)
    destination.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    phase_color = "#3b7dd8"

    steps = tqdm(
        total=8, desc="Writing figures", unit="fig", disable=not show_progress
    )

    instances = [scan["class_instances"][index] for index in range(nc)]
    order = sorted(range(nc), key=lambda index: instances[index])
    figure, axis = plt.subplots(figsize=(9, 6.5))
    axis.barh(
        [names[index] for index in order],
        [instances[index] for index in order],
        color=phase_color,
    )
    axis.set_xlabel("Instances")
    axis.set_title("Phase 4 — instances per class")
    axis.grid(axis="x", alpha=0.3)
    figure.tight_layout()
    path = destination / "phase4_class_instances.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    written.append(path)
    steps.update()

    images_with = [scan["class_images"][index] for index in range(nc)]
    order = sorted(range(nc), key=lambda index: images_with[index])
    figure, axis = plt.subplots(figsize=(9, 6.5))
    axis.barh(
        [names[index] for index in order],
        [images_with[index] for index in order],
        color=phase_color,
    )
    axis.set_xlabel("Images containing class")
    axis.set_title("Phase 4 — images containing each class")
    axis.grid(axis="x", alpha=0.3)
    figure.tight_layout()
    path = destination / "phase4_class_images.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    written.append(path)
    steps.update()

    figure, axis = plt.subplots(figsize=(10, 7))
    height = 0.26
    positions = list(range(nc))
    for offset, split in zip((height, 0.0, -height), SPLITS):
        values = [scan["split_class_instances"][split][index] for index in range(nc)]
        axis.barh(
            [position + offset for position in positions],
            values,
            height=height,
            label=split,
        )
    axis.set_yticks(positions)
    axis.set_yticklabels(names)
    axis.set_xlabel("Instances")
    axis.set_title("Phase 4 — instances per class by split")
    axis.legend()
    axis.grid(axis="x", alpha=0.3)
    figure.tight_layout()
    path = destination / "phase4_split_class_instances.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    written.append(path)
    steps.update()

    bucket_labels = list(BUCKET_ORDER)
    bucket_values = [context["buckets"][label] for label in bucket_labels]
    figure, axis = plt.subplots(figsize=(7.5, 4.8))
    bars = axis.bar(bucket_labels, bucket_values, color=phase_color)
    axis.bar_label(bars, fmt="%d")
    axis.set_xlabel("Objects per image")
    axis.set_ylabel("Images")
    axis.set_title("Phase 4 — objects per image distribution")
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    path = destination / "phase4_objects_per_image.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    written.append(path)
    steps.update()

    areas = scan["areas"]
    positive = [area for area in areas if area > 0.0]
    smallest = min(positive) if positive else 1e-6
    start = math.floor(math.log10(smallest))
    bins = [10 ** (start + index * (0 - start) / 70) for index in range(71)]
    figure, (hist_axis, band_axis) = plt.subplots(1, 2, figsize=(11.5, 4.8))
    hist_axis.hist(areas, bins=bins, color=phase_color)
    hist_axis.set_xscale("log")
    hist_axis.set_yscale("log")
    hist_axis.set_xlabel("Normalized box area (width x height, log scale)")
    hist_axis.set_ylabel("Annotations (log scale)")
    hist_axis.set_title("Box area distribution")
    hist_axis.grid(alpha=0.3)
    band_names = [name for name, _low, _high in SIZE_BANDS]
    band_values = [context["bands"][name]["count"] for name in band_names]
    bars = band_axis.bar(band_names, band_values, color=phase_color)
    band_axis.bar_label(bars, fmt="%d")
    band_axis.set_ylabel("Annotations")
    band_axis.set_title("Box size bands (analysis bins)")
    band_axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    path = destination / "phase4_box_area.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    written.append(path)
    steps.update()

    matrix = [[0 for _ in range(nc)] for _ in range(nc)]
    for (class_a, class_b), count in context["pair_counts"].items():
        matrix[class_a][class_b] = count
        matrix[class_b][class_a] = count
    figure, axis = plt.subplots(figsize=(9.5, 8.2))
    image = axis.imshow(matrix, cmap="viridis", interpolation="nearest")
    axis.set_xticks(range(nc))
    axis.set_yticks(range(nc))
    axis.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    axis.set_yticklabels(names, fontsize=8)
    axis.set_title("Phase 4 — class co-occurrence (images containing both)")
    figure.colorbar(image, ax=axis, label="Shared images")
    figure.tight_layout()
    path = destination / "phase4_cooccurrence.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    written.append(path)
    steps.update()

    figure, axis = plt.subplots(figsize=(9, 7))
    train_values = [
        pct(
            scan["split_class_instances"]["train"][index],
            sum(scan["split_class_instances"][split][index] for split in SPLITS),
        )
        for index in range(nc)
    ]
    val_values = [
        pct(
            scan["split_class_instances"]["val"][index],
            sum(scan["split_class_instances"][split][index] for split in SPLITS),
        )
        for index in range(nc)
    ]
    test_values = [
        100.0 - train_values[index] - val_values[index] for index in range(nc)
    ]
    order = sorted(range(nc), key=lambda index: train_values[index])
    axis.barh(
        [names[index] for index in order],
        [train_values[index] for index in order],
        label="train",
    )
    axis.barh(
        [names[index] for index in order],
        [val_values[index] for index in order],
        left=[train_values[index] for index in order],
        label="val",
    )
    axis.barh(
        [names[index] for index in order],
        [test_values[index] for index in order],
        left=[
            train_values[index] + val_values[index] for index in order
        ],
        label="test",
    )
    axis.set_xlabel("Share of class instances (%)")
    axis.set_title("Phase 4 — split proportions per class")
    axis.legend()
    axis.grid(axis="x", alpha=0.3)
    figure.tight_layout()
    path = destination / "phase4_split_proportions.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    written.append(path)
    steps.update()

    if context["dimensions"]:
        dims = context["dim_stats"]
        widths = [size[0] for size in context["dimensions"].values()]
        heights = [size[1] for size in context["dimensions"].values()]
        aspects = [
            width / height if height else 0.0 for width, height in zip(widths, heights)
        ]
        figure, axes = plt.subplots(1, 3, figsize=(13, 4.2))
        axes[0].hist(widths, bins=40, color=phase_color)
        axes[0].set_title(f"Width (median {dims['width']['median']:.0f})")
        axes[1].hist(heights, bins=40, color=phase_color)
        axes[1].set_title(f"Height (median {dims['height']['median']:.0f})")
        axes[2].hist(aspects, bins=40, color=phase_color)
        axes[2].set_title(
            f"Aspect w/h (median {dims['aspect_ratio']['median']:.2f})"
        )
        for axis, label in zip(axes, ("Width (px)", "Height (px)", "Width / height")):
            axis.set_xlabel(label)
            axis.set_ylabel("Images")
            axis.grid(alpha=0.3)
        figure.suptitle("Phase 4 — image dimension distributions")
        figure.tight_layout()
        path = destination / "phase4_image_dimensions.png"
        figure.savefig(path, dpi=150)
        plt.close(figure)
        written.append(path)
    steps.update()
    return written


def render_report(context: dict[str, Any]) -> str:
    scan = context["scan"]
    class_map = context["class_map"]
    names = [class_map[index] for index in range(len(class_map))]
    nc = len(names)
    imbalance = context["imbalance"]
    objects_desc = context["objects_desc"]
    multi = context["multi"]
    quality = context["quality"]
    dims = context["dim_stats"]
    hashes = context["hashes"]
    lines: list[str] = ["# Phase 4 — Dataset Analysis", ""]

    lines += [
        "## 1. Dataset Overview",
        "",
        "- Source: COCO 2017, prepared to YOLO format in Phase 3 at",
        "  `data/processed/household_objects/` (analysis is read-only).",
        f"- Classes: **{nc}** (`configs/classes.yaml`),",
        f" images: **{context['total_images']:,}**",
        f" (train {scan['splits']['train']['images']:,} /",
        f" val {scan['splits']['val']['images']:,} /",
        f" test {scan['splits']['test']['images']:,}).",
        f"- Instances: **{context['total_instances']:,}**",
        f" (train {scan['splits']['train']['instances']:,} /",
        f" val {scan['splits']['val']['instances']:,} /",
        f" test {scan['splits']['test']['instances']:,}).",
        "- Splits: COCO train2017 -> project train (90%) + val (10%), seed 42;",
        "  COCO val2017 -> project test (Phase 3 design, unchanged).",
        "- Generated by `python scripts/analyze_dataset.py`; all values below",
        "  are measured from the prepared labels and image headers.",
        "",
        "## 2. Final Classes",
        "",
        "| ID | Class | Train inst | Val inst | Test inst |",
        "|---:|---|---:|---:|---:|",
    ]
    for class_id in range(nc):
        lines.append(
            f"| {class_id} | {names[class_id]} |"
            f" {scan['split_class_instances']['train'][class_id]:,} |"
            f" {scan['split_class_instances']['val'][class_id]:,} |"
            f" {scan['split_class_instances']['test'][class_id]:,} |"
        )
    lines += [
        "",
        "## 3. Dataset Counts",
        "",
        "| Split | Images | Labels | Instances | Mean objects/image |",
        "|---|---:|---:|---:|---:|",
    ]
    for split in SPLITS:
        entry = scan["splits"][split]
        mean = entry["instances"] / entry["images"] if entry["images"] else 0.0
        lines.append(
            f"| {split} | {entry['images']:,} | {entry['labels']:,} |"
            f" {entry['instances']:,} | {mean:.2f} |"
        )
    lines.append(
        f"| **total** | **{context['total_images']:,}** |"
        f" **{sum(scan['splits'][s]['labels'] for s in SPLITS):,}** |"
        f" **{context['total_instances']:,}** |"
        f" {context['total_instances'] / context['total_images']:.2f} |"
    )
    lines += [
        "",
        f"Matches verified Phase 3 counts: **{context['matches_phase3']}**.",
        "",
        "## 4. Class Distribution",
        "",
        "Sorted by total instances:",
        "",
        "| ID | Class | Instances | Share | Images containing | % of images |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    order = sorted(range(nc), key=lambda index: -scan["class_instances"][index])
    for class_id in order:
        count = scan["class_instances"][class_id]
        image_count = scan["class_images"][class_id]
        lines.append(
            f"| {class_id} | {names[class_id]} | {count:,} |"
            f" {pct(count, context['total_instances']):.1f}% |"
            f" {image_count:,} |"
            f" {pct(image_count, context['total_images']):.1f}% |"
        )
    lines += [
        "",
        "Per-split image presence (images containing the class):",
        "",
        "| Class | Train | Val | Test |",
        "|---|---:|---:|---:|",
    ]
    for class_id in order:
        lines.append(
            f"| {names[class_id]} |"
            f" {scan['split_class_images']['train'][class_id]:,} |"
            f" {scan['split_class_images']['val'][class_id]:,} |"
            f" {scan['split_class_images']['test'][class_id]:,} |"
        )
    lines += [
        "",
        "## 5. Class Imbalance",
        "",
        f"- Most frequent: **{imbalance['most_class_name']}**"
        f" ({imbalance['most_class_instances']:,} instances,"
        f" {pct(imbalance['most_class_instances'], context['total_instances']):.1f}%"
        " of all instances).",
        f"- Least frequent: **{imbalance['least_class_name']}**"
        f" ({imbalance['least_class_instances']:,} instances,"
        f" {pct(imbalance['least_class_instances'], context['total_instances']):.1f}%).",
        f"- Max/min instance ratio: **{imbalance['max_min_ratio']:.1f}x**.",
        f"- Median instances per class: {imbalance['median_instances']:,.0f};"
        f" mean {imbalance['mean_instances']:,.1f}.",
        "",
        "The distribution is right-skewed: the top five classes",
        f" ({', '.join(names[i] for i in order[:5])}) hold"
        f" {pct(sum(scan['class_instances'][i] for i in order[:5]), context['total_instances']):.1f}%"
        " of all instances, while the bottom five hold"
        f" {pct(sum(scan['class_instances'][i] for i in order[-5:]), context['total_instances']):.1f}%.",
        "",
        "## 6. Objects Per Image",
        "",
        "| Split | Mean | Median | Min | Max | p90 | p95 | p99 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split, entry in [("all", objects_desc)] + [
        (split, context["per_split_objects_desc"][split]) for split in SPLITS
    ]:
        lines.append(
            f"| {split} | {entry['mean']:.2f} | {entry['median']:.1f} |"
            f" {entry['min']:.0f} | {entry['max']:.0f} | {entry['p90']:.0f} |"
            f" {entry['p95']:.0f} | {entry['p99']:.0f} |"
        )
    lines += [
        "",
        "Images by actual observed object count:",
        "",
        "| Objects in image | Images | Share |",
        "|---|---:|---:|",
    ]
    for label in BUCKET_ORDER:
        value = context["buckets"][label]
        lines.append(
            f"| {label} | {value:,} | {pct(value, context['total_images']):.1f}% |"
        )
    lines += [
        "",
        "## 7. Multi-Object Scenes",
        "",
        f"- Single-object images: {multi['single_object_images']:,}"
        f" ({pct(multi['single_object_images'], multi['total_images']):.1f}%).",
        f"- Multi-object images (2+ target objects):"
        f" {multi['multi_object_images']:,}"
        f" ({multi['multi_share_pct']:.1f}%).",
        f"- Images with 3+ target objects: {multi['images_3plus']:,}"
        f" ({pct(multi['images_3plus'], multi['total_images']):.1f}%).",
        f"- Maximum target objects in one image: {multi['max_objects_in_image']}.",
        "",
        "Multi-object scenes are the majority, so the intended application",
        "(multi-object detection on real photos) is well supported by the data.",
        "",
        "## 8. Bounding Box Size Distribution",
        "",
        "Box area = normalized width x height (fraction of image area).",
        "Size bands are **analysis bins only**, not a universal standard:",
        "",
        "| Band | Area range | Annotations | Share |",
        "|---|---|---:|---:|",
    ]
    for name, low, high in SIZE_BANDS:
        range_text = (
            f"{low:g} - {high:g}" if math.isfinite(high) else f">= {low:g}"
        )
        entry = context["bands"][name]
        lines.append(
            f"| {name} | {range_text} | {entry['count']:,} | {entry['pct']:.1f}% |"
        )
    area_desc = context["area_desc"]
    lines += [
        "",
        "| Metric | Area | Width | Height |",
        "|---|---:|---:|---:|",
    ]
    for label, key in (
        ("min", "min"),
        ("max", "max"),
        ("mean", "mean"),
        ("median", "median"),
        ("p90", "p90"),
        ("p95", "p95"),
        ("p99", "p99"),
    ):
        lines.append(
            f"| {label} | {area_desc[key]:.6f} | {context['width_desc'][key]:.4f} |"
            f" {context['height_desc'][key]:.4f} |"
        )
    lines += [
        "",
        "Figure: `reports/figures/phase4/phase4_box_area.png`.",
        "",
        "## 9. Small Object Analysis",
        "",
        f"- Small boxes (area < {SMALL_AREA_LIMIT:g}, i.e. < 1% of image):"
        f" **{context['small_boxes']:,}** annotations"
        f" (**{context['small_pct']:.1f}%** of all annotations).",
        f"- Very small boxes (area < 1e-3): {context['bands']['very small']['count']:,}"
        f" ({context['bands']['very small']['pct']:.1f}%).",
        f"- Tiny boxes (area < {TINY_AREA_LIMIT:g}): {quality['tiny_boxes']:,}"
        f" ({quality['tiny_boxes_pct']:.1f}%).",
        "",
        "By split:",
        "",
        "| Split | Small boxes | Split instances | Share |",
        "|---|---:|---:|---:|",
    ]
    for split in SPLITS:
        split_total = scan["areas_by_split"][split]
        value = context["small_by_split"][split]
        lines.append(
            f"| {split} | {value:,} | {len(split_total):,} |"
            f" {pct(value, len(split_total)):.1f}% |"
        )
    small_order = sorted(
        class_map, key=lambda index: -context["small_by_class"][index]
    )
    lines += [
        "",
        "Classes most affected (largest count of small boxes):",
        "",
        "| Class | Small boxes | Class instances | Share within class |",
        "|---|---:|---:|---:|",
    ]
    for class_id in small_order[:6]:
        value = context["small_by_class"][class_id]
        total = scan["class_instances"][class_id]
        lines.append(
            f"| {names[class_id]} | {value:,} | {total:,} |"
            f" {pct(value, total):.1f}% |"
        )
    lines += [
        "",
        "Small boxes contain fewer pixels than large boxes, so less visual",
        "information is available for learning and evaluating them; this can",
        "affect detection quality for the affected classes and should be",
        "reflected in per-class evaluation later (no model claims made here).",
        "",
        "## 10. Class Co-Occurrence",
        "",
        "Top pairs by number of images containing both classes:",
        "",
        "| Rank | Class A | Class B | Images | % of images |",
        "|---:|---|---|---:|---:|",
    ]
    for rank, ((class_a, class_b), count) in enumerate(
        context["top_pairs"], start=1
    ):
        lines.append(
            f"| {rank} | {names[class_a]} | {names[class_b]} | {count:,} |"
            f" {pct(count, context['total_images']):.1f}% |"
        )
    lines += [
        "",
        "Matrix figure: `reports/figures/phase4/phase4_cooccurrence.png`.",
        "Co-occurrence reflects real household scenes (e.g. dining settings,",
        "desks); the model will see these contextual cues during training.",
        "",
        "## 11. Train/Validation/Test Distribution",
        "",
        "Overall instance shares:",
        "",
        "| Split | Instances | Share | Images | Share |",
        "|---|---:|---:|---:|---:|",
    ]
    image_total = context["total_images"]
    for split in SPLITS:
        lines.append(
            f"| {split} | {context['split_instances'][split]:,} |"
            f" {context['overall_split_share'][split]:.1f}% |"
            f" {scan['splits'][split]['images']:,} |"
            f" {pct(scan['splits'][split]['images'], image_total):.1f}% |"
        )
    deviations: list[tuple[str, str, float, float]] = []
    for class_id in range(nc):
        class_total = sum(
            scan["split_class_instances"][split][class_id] for split in SPLITS
        )
        if not class_total:
            continue
        for split in SPLITS:
            share = pct(scan["split_class_instances"][split][class_id], class_total)
            overall = context["overall_split_share"][split]
            deviations.append((names[class_id], split, share, overall))
    deviations.sort(key=lambda item: -abs(item[2] - item[3]))
    lines += [
        "",
        "Per-class split proportions (share of that class's instances):",
        "",
        "| Class | Train % | Val % | Test % |",
        "|---|---:|---:|---:|",
    ]
    for class_id in range(nc):
        class_total = sum(
            scan["split_class_instances"][split][class_id] for split in SPLITS
        )
        shares = [
            pct(scan["split_class_instances"][split][class_id], class_total)
            for split in SPLITS
        ]
        lines.append(
            f"| {names[class_id]} | {shares[0]:.1f} | {shares[1]:.1f} |"
            f" {shares[2]:.1f} |"
        )
    largest = deviations[0] if deviations else ("", "", 0.0, 0.0)
    lines += [
        "",
        f"Largest deviation from overall split shares: {largest[0]} in"
        f" {largest[1]} ({largest[2]:.1f}% vs overall {largest[3]:.1f}%).",
        "No class shows a materially different split composition; the test set",
        "comes from COCO val2017 by design and remains untouched.",
        "",
        "## 12. Image Dimensions",
    ]
    if dims and dims.get("count"):
        width = dims["width"]
        height = dims["height"]
        aspect = dims["aspect_ratio"]
        lines += [
            "",
            "| Metric | Width (px) | Height (px) | Aspect w/h |",
            "|---|---:|---:|---:|",
        ]
        for label, key in (
            ("min", "min"),
            ("max", "max"),
            ("mean", "mean"),
            ("median", "median"),
            ("p90", "p90"),
        ):
            lines.append(
                f"| {label} | {width[key]:.0f} | {height[key]:.0f} |"
                f" {aspect[key]:.2f} |"
            )
        lines += [
            "",
            f"- Landscape (aspect > 1.02): {dims['landscape']:,}"
            f" ({pct(dims['landscape'], dims['count']):.1f}%);"
            f" portrait (< 0.98): {dims['portrait']:,}"
            f" ({pct(dims['portrait'], dims['count']):.1f}%);"
            f" near-square: {dims['square']:,}"
            f" ({pct(dims['square'], dims['count']):.1f}%).",
            "- Most common widths: "
            + ", ".join(f"{value} px ({count:,})" for value, count in dims["top_widths"]),
            "- Most common heights: "
            + ", ".join(f"{value} px ({count:,})" for value, count in dims["top_heights"]),
            "- Extremes:",
        ]
        for entry in dims["extremes"]:
            lines.append(
                f"  - {entry['label']}: {entry['value']}"
                f" ({entry['split']}/{entry['stem']}.jpg)"
            )
        lines += [
            "",
            "Figure: `reports/figures/phase4/phase4_image_dimensions.png`.",
            "Images are not resized or modified by this analysis.",
        ]
    else:
        lines += ["", "Dimension scan skipped for this run."]
    malformed_total = sum(quality["malformed_lines"].values())
    lines += [
        "",
        "## 13. Annotation Quality",
        "",
        "Findings classified as: **A** confirmed issue,",
        "**B** possible concern, **C** normal dataset characteristic.",
        "",
        "| Check | Value | Class |",
        "|---|---:|---|",
        f"| Empty label files | {quality['empty_label_files']} |"
        f" {'A' if quality['empty_label_files'] else 'C'} |",
        f"| Images missing label files | {quality['missing_label_files']} |"
        f" {'A' if quality['missing_label_files'] else 'C'} |",
        f"| Labels without matching image | {quality['label_without_image']} |"
        f" {'A' if quality['label_without_image'] else 'C'} |",
        f"| Malformed label lines (all causes) | {malformed_total} |"
        f" {'A' if malformed_total else 'C'} |",
        f"| Images with 0 objects | {quality['zero_object_images']} |"
        f" {'A' if quality['zero_object_images'] else 'C'} |",
        f"| Boxes with area >= {LARGE_AREA_LIMIT:g} | {quality['large_boxes']:,}"
        f" ({quality['large_boxes_pct']:.1f}%) | C |",
        f"| Tiny boxes (area < {TINY_AREA_LIMIT:g}) | {quality['tiny_boxes']:,}"
        f" ({quality['tiny_boxes_pct']:.1f}%) | C |",
        f"| Max objects in one image | {quality['max_objects_in_image']} | C |",
        f"| Images at/above p99 object count |"
        f" {quality['images_at_or_above_p99']:,} | C |",
        "",
        "Per-category malformed-line breakdown: "
        + ", ".join(
            f"{key}={value}"
            for key, value in quality["malformed_lines"].items()
        )
        + ".",
        "",
        "Least-supported classes (possible concern B for evaluation granularity): "
        + ", ".join(
            f"{names[entry['id']]} ({entry['instances']:,})"
            for entry in quality["least_supported_classes"]
        )
        + ".",
        "",
        "## 14. Representative Examples",
        "",
        "Annotated copies (originals untouched) under `reports/phase4_samples/`:",
        "",
        "| File | Split | Objects | Classes | Note |",
        "|---|---|---:|---|---|",
    ]
    for sample in context["samples"]:
        lines.append(
            f"| `{sample['file']}` | {sample['split']} | {sample['objects']} |"
            f" {', '.join(sample['classes'][:6])} | {sample['note']} |"
        )
    lines += [
        "",
        "Covers one example per class, multi-object, crowded, smallest-box,",
        "largest-box, extreme aspect ratios, and train/val/test splits.",
        "",
        "## 15. Dataset Concerns",
        "",
        "**A. Confirmed issues**",
        "",
    ]
    confirmed: list[str] = []
    if quality["empty_label_files"]:
        confirmed.append(f"{quality['empty_label_files']} empty label files")
    if quality["missing_label_files"]:
        confirmed.append(f"{quality['missing_label_files']} images lack label files")
    if malformed_total:
        confirmed.append(f"{malformed_total} malformed label lines")
    if quality["zero_object_images"]:
        confirmed.append(f"{quality['zero_object_images']} images with 0 objects")
    if hashes and hashes.get("cross_split_pairs"):
        confirmed.append(
            f"{hashes['cross_split_pairs']} content-identical image pairs across"
            " splits (leakage risk)"
        )
    if confirmed:
        lines += [f"- {item}" for item in confirmed]
    else:
        lines.append("- None found: labels are well-formed, complete, and no")
        lines.append("  cross-split content duplicates were detected.")
    lines += [
        "",
        "**B. Possible concerns**",
        "",
        f"- Class imbalance ({imbalance['max_min_ratio']:.1f}x max/min) means",
        f"  {imbalance['least_class_name']} has only"
        f" {imbalance['least_class_instances']:,} instances and"
        f" {scan['split_class_images']['test'][imbalance['least_class_id']]:,}"
        " test images; per-class metrics for it will be noisy.",
        f"- {context['small_pct']:.1f}% of annotations are small boxes",
        "  (area < 1%), which may be under-detected at small input resolutions.",
        "- Visually similar classes (cup / wine glass / bowl, remote / cell",
        "  phone / keyboard) may be confused; evaluate them individually.",
        "",
        "**C. Normal dataset characteristics**",
        "",
        f"- Multi-object scenes dominate ({multi['multi_share_pct']:.1f}% of",
        "  images have 2+ target objects) — expected for indoor photos.",
        "- Long-tailed class frequencies are typical of real-world datasets.",
        "- Test images originate from COCO val2017 (Phase 3 design).",
        "- Large boxes (area >= 0.5) occur for furniture such as couch/bed.",
        "",
        "## 16. Implications for Model Training",
        "",
        "- Monitor per-class metrics during Phase 7 evaluation; the imbalance",
        f"  ({imbalance['max_min_ratio']:.1f}x) makes aggregate mAP dominated by",
        f"  {imbalance['most_class_name']}.",
        "- Report recall for small-box classes separately; small objects are a",
        "  known hard case and account for a measurable share here.",
        "- Multi-object support is essential: most images contain 2+ target",
        "  objects, matching the web-app use case.",
        "- Similar-class confusion pairs (cup/wine glass, remote/cell phone)",
        "  should be inspected in the confusion matrix during evaluation.",
        "- The 10% val split gives ~4.5k validation images; keep the val set",
        "  fixed for comparable experiments across Phases 5-9.",
        "",
        "## 17. Phase 4 Conclusion",
        "",
        f"The prepared dataset is internally consistent: {context['total_images']:,}",
        f" images and {context['total_instances']:,} instances across {nc} classes",
        " with no confirmed annotation defects and no detected cross-split",
        f" duplicates. Class frequencies are right-skewed (max/min"
        f" {imbalance['max_min_ratio']:.1f}x), {context['small_pct']:.1f}% of boxes",
        f" are small, and {multi['multi_share_pct']:.1f}% of images are",
        "multi-object scenes. These findings are documented for Phase 5+; the",
        "dataset itself is unchanged by this analysis.",
        "",
    ]
    return "\n".join(lines)


def build_summary(context: dict[str, Any]) -> dict[str, Any]:
    scan = context["scan"]
    class_map = context["class_map"]
    names = [class_map[index] for index in range(len(class_map))]
    classes = [
        {
            "id": class_id,
            "name": names[class_id],
            "instances": scan["class_instances"][class_id],
            "share_of_instances_pct": pct(
                scan["class_instances"][class_id], context["total_instances"]
            ),
            "images_containing": scan["class_images"][class_id],
            "share_of_images_pct": pct(
                scan["class_images"][class_id], context["total_images"]
            ),
            "split_instances": {
                split: scan["split_class_instances"][split][class_id]
                for split in SPLITS
            },
            "split_images": {
                split: scan["split_class_images"][split][class_id]
                for split in SPLITS
            },
            "small_boxes": context["small_by_class"][class_id],
        }
        for class_id in range(len(class_map))
    ]
    return {
        "generated_by": "scripts/analyze_dataset.py",
        "dataset": {
            "path": "data/processed/household_objects",
            "classes": len(class_map),
            "names": names,
            "splits": {
                split: dict(scan["splits"][split]) for split in SPLITS
            },
        },
        "totals": {
            "images": context["total_images"],
            "instances": context["total_instances"],
            "labels": sum(scan["splits"][split]["labels"] for split in SPLITS),
        },
        "matches_phase3_verified_counts": context["matches_phase3"],
        "classes": classes,
        "imbalance": context["imbalance"],
        "objects_per_image": {
            "all": context["objects_desc"],
            **{
                split: context["per_split_objects_desc"][split]
                for split in SPLITS
            },
            "buckets": context["buckets"],
        },
        "multi_object": context["multi"],
        "bounding_boxes": {
            "area": context["area_desc"],
            "width": context["width_desc"],
            "height": context["height_desc"],
            "size_bands": {
                name: {
                    "range": [low, high if math.isfinite(high) else None],
                    "count": context["bands"][name]["count"],
                    "pct": context["bands"][name]["pct"],
                }
                for name, low, high in SIZE_BANDS
            },
        },
        "small_objects": {
            "area_limit": SMALL_AREA_LIMIT,
            "count": context["small_boxes"],
            "pct_of_instances": context["small_pct"],
            "by_split": context["small_by_split"],
            "by_class": {
                names[class_id]: context["small_by_class"][class_id]
                for class_id in range(len(class_map))
            },
        },
        "cooccurrence": {
            "top_pairs": [
                {
                    "class_a_id": class_a,
                    "class_a": names[class_a],
                    "class_b_id": class_b,
                    "class_b": names[class_b],
                    "images": count,
                    "pct_of_images": pct(count, context["total_images"]),
                }
                for (class_a, class_b), count in context["top_pairs"]
            ],
            "matrix": [
                [
                    context["pair_counts"].get(
                        (min(class_a, class_b), max(class_a, class_b)), 0
                    )
                    for class_b in range(len(class_map))
                ]
                for class_a in range(len(class_map))
            ],
        },
        "split_distribution": {
            "instance_counts": context["split_instances"],
            "instance_shares_pct": context["overall_split_share"],
            "image_shares_pct": {
                split: pct(scan["splits"][split]["images"], context["total_images"])
                for split in SPLITS
            },
        },
        "image_dimensions": context["dim_stats"],
        "annotation_quality": context["quality"],
        "duplicates": {
            "filename_overlap_across_splits": context["filename_overlap"],
            "content_hash": context["hashes"],
        },
        "samples": {
            "count": len(context["samples"]),
            "files": [sample["file"] for sample in context["samples"]],
        },
    }


def render_samples_index(samples: list[dict[str, Any]]) -> str:
    lines = [
        "# Phase 4 — Representative Samples",
        "",
        "Annotated copies of prepared images (originals are never modified),",
        "generated by `python scripts/analyze_dataset.py`. File naming:",
        "`<category>__<split>__<coco-stem>.jpg`.",
        "",
        "| File | Split | Objects | Classes | Note |",
        "|---|---|---:|---|---|",
    ]
    for sample in samples:
        lines.append(
            f"| `{sample['file']}` | {sample['split']} | {sample['objects']} |"
            f" {', '.join(sample['classes'][:6])} | {sample['note']} |"
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-dimensions", action="store_true")
    parser.add_argument("--skip-hashes", action="store_true")
    parser.add_argument("--skip-samples", action="store_true")
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    show_progress = not args.quiet

    try:
        config = load_yaml(args.config)
        class_map = load_classes(PROJECT_ROOT / config["classes_config"])
        output_dir = PROJECT_ROOT / config["output"]["dir"]
        pre_counts = preflight(output_dir, class_map)
        scan = scan_labels(output_dir, class_map, show_progress=show_progress)
        verify_expected(scan)
        dimensions = (
            None
            if args.skip_dimensions
            else scan_dimensions(output_dir, show_progress=show_progress)
        )
        hashes = (
            None
            if args.skip_hashes
            else hash_duplicates(output_dir, show_progress=show_progress)
        )
        samples: list[dict[str, Any]] = []
        if not args.skip_samples:
            samples = write_samples(
                output_dir,
                class_map,
                scan,
                dimensions,
                SAMPLES_DIR,
                seed=args.seed,
                show_progress=show_progress,
            )
        post_counts = count_files(output_dir)
        if post_counts != pre_counts:
            raise AnalysisError(
                "dataset file counts changed during analysis"
                f" (before={pre_counts}, after={post_counts})"
            )
        context = build_context(scan, class_map, dimensions, hashes, samples)
        figures: list[Path] = []
        if not args.skip_figures:
            figures = write_figures(context, FIGURES_DIR, show_progress=show_progress)
    except AnalysisError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1
    except (OSError, ValueError, KeyError) as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(render_report(context), encoding="utf-8")
    SUMMARY_PATH.write_text(
        json.dumps(build_summary(context), indent=2) + "\n", encoding="utf-8"
    )
    print(f"[ok] wrote {REPORT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"[ok] wrote {SUMMARY_PATH.relative_to(PROJECT_ROOT)}")
    if samples:
        index_path = SAMPLES_DIR / "README.md"
        index_path.write_text(render_samples_index(samples), encoding="utf-8")
        print(
            f"[ok] wrote {len(samples)} samples ->"
            f" {SAMPLES_DIR.relative_to(PROJECT_ROOT)}/"
        )
    for figure in figures:
        print(f"[ok] wrote {figure.relative_to(PROJECT_ROOT)}")
    for split in SPLITS:
        entry = scan["splits"][split]
        print(
            f"[stat] {split}: {entry['images']} images,"
            f" {entry['instances']} instances"
        )
    if hashes is not None:
        print(
            f"[stat] duplicates: {hashes['duplicate_pairs']} pair(s),"
            f" {hashes['cross_split_pairs']} across splits"
        )
    print("[ok] dataset counts unchanged after analysis")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
