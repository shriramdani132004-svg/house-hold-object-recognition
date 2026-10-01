"""Evaluate the COCO-pretrained YOLO26n baseline on the project test split.

Run from the project root:

    python scripts/evaluate_baseline.py [--config configs/baseline.yaml]
        [--recompute] [--limit N]

Performs ONE streamed inference pass over the 1,965-image test split on CPU
(tqdm progress with count, percent, elapsed and ETA), then computes the
Phase 5 baseline metrics, confidence-threshold analysis, speed summary,
failure categorization and sample renderings.

Outputs (project-relative):

    reports/baseline_detections.json          (inference cache, git-ignored)
    reports/baseline_results.json
    reports/baseline_report.md
    reports/baseline_confidence_analysis.md
    reports/baseline_error_examples.md
    reports/baseline_samples/                 (~20 annotated images + index)

--recompute ignores a valid cache and reruns inference; --limit N evaluates
only the first N images (debug use — never for reported numbers).
Exit code 0 on success.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.baseline import (
    CONFIDENCE_THRESHOLDS,
    MATCH_IOU,
    aggregate_matches,
    average_precision,
    box_iou,
    build_class_mapping,
    build_results_json,
    filter_detections,
    load_config,
    load_project_classes,
    match_image,
    mean_average_precision,
    precision_recall_f1,
    resolve_path,
    select_samples,
)

SCRIPT_DIR = PROJECT_ROOT / "scripts"

SIMILAR_CLASS_PAIRS: set[frozenset[str]] = {
    frozenset({"cup", "wine glass"}),
    frozenset({"cup", "bowl"}),
    frozenset({"bottle", "wine glass"}),
    frozenset({"bottle", "vase"}),
    frozenset({"tv", "remote"}),
    frozenset({"tv", "laptop"}),
    frozenset({"laptop", "cell phone"}),
    frozenset({"cell phone", "remote"}),
    frozenset({"couch", "bed"}),
    frozenset({"chair", "couch"}),
    frozenset({"chair", "bed"}),
    frozenset({"mouse", "keyboard"}),
    frozenset({"potted plant", "vase"}),
    frozenset({"clock", "tv"}),
}


def load_sibling(script_filename: str) -> Any:
    """Import another script from this directory as a module."""
    import importlib.util

    script_path = SCRIPT_DIR / script_filename
    name = f"phase5_{script_path.stem}"
    spec = importlib.util.spec_from_file_location(name, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_targets(labels_dir: Path, class_count: int) -> tuple[dict[str, list[dict[str, Any]]], int]:
    """Parse YOLO label files into normalized ``(x1, y1, x2, y2)`` targets.

    Returns the per-image target map (keyed by image stem) and the total
    instance count.
    """
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"labels directory not found: {labels_dir}")
    targets: dict[str, list[dict[str, Any]]] = {}
    total = 0
    for label_path in sorted(labels_dir.glob("*.txt")):
        records: list[dict[str, Any]] = []
        content = label_path.read_text(encoding="utf-8").strip()
        if content:
            for line in content.splitlines():
                parts = line.split()
                if len(parts) != 5:
                    raise ValueError(f"{label_path}: malformed line {line!r}")
                class_id = int(parts[0])
                if not 0 <= class_id < class_count:
                    raise ValueError(
                        f"{label_path}: class id {class_id} outside 0..{class_count - 1}"
                    )
                xc, yc, box_w, box_h = (float(value) for value in parts[1:])
                x1 = xc - box_w / 2.0
                y1 = yc - box_h / 2.0
                x2 = xc + box_w / 2.0
                y2 = yc + box_h / 2.0
                records.append(
                    {
                        "class_id": class_id,
                        "bbox": (x1, y1, x2, y2),
                        "area": box_w * box_h,
                    }
                )
        targets[label_path.stem] = records
        total += len(records)
    return targets, total


def count_images(images_dir: Path) -> int:
    """Count test images (jpg/jpeg/png)."""
    if not images_dir.is_dir():
        raise FileNotFoundError(f"images directory not found: {images_dir}")
    return sum(
        1
        for path in images_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )


def load_model(model_path: Path) -> tuple[Any, float]:
    """Load the baseline weights once; returns (model, load_ms)."""
    from ultralytics import YOLO

    started = time.perf_counter()
    model = YOLO(str(model_path))
    load_ms = (time.perf_counter() - started) * 1000.0
    return model, load_ms


def run_test_pass(
    model: Any,
    image_paths: list[Path],
    mapping: dict[int, int],
    conf_floor: float,
    imgsz: int,
    device: str,
    max_dets: int,
) -> tuple[list[dict[str, Any]], float]:
    """Single streamed CPU pass over the test images.

    Returns per-image records (filtered project-class detections plus
    timing) and the total wall time in milliseconds.
    """
    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    for image_path in tqdm(
        image_paths,
        desc="baseline inference",
        unit="img",
        dynamic_ncols=True,
    ):
        image_started = time.perf_counter()
        result = model.predict(
            source=str(image_path),
            conf=conf_floor,
            imgsz=imgsz,
            device=device,
            max_det=max_dets,
            verbose=False,
            save=False,
        )[0]
        wall_ms = (time.perf_counter() - image_started) * 1000.0
        height, width = result.orig_shape
        raw_detections: list[dict[str, Any]] = []
        if result.boxes is not None and len(result.boxes):
            boxes = result.boxes.xyxy.cpu().tolist()
            confidences = result.boxes.conf.cpu().tolist()
            classes = result.boxes.cls.cpu().tolist()
            for box, score, class_index in zip(boxes, confidences, classes):
                raw_detections.append(
                    {
                        "coco_class": int(class_index),
                        "score": float(score),
                        "bbox": (
                            box[0] / width,
                            box[1] / height,
                            box[2] / width,
                            box[3] / height,
                        ),
                    }
                )
        kept = filter_detections(raw_detections, mapping)
        records.append(
            {
                "image_id": image_path.stem,
                "wall_ms": round(wall_ms, 3),
                "infer_ms": round(float(result.speed.get("inference", 0.0)), 3),
                "raw_count": len(raw_detections),
                "dets": kept,
            }
        )
    total_ms = (time.perf_counter() - started) * 1000.0
    return records, total_ms


def cache_meta(
    model_path: Path,
    conf_floor: float,
    imgsz: int,
    mapping: dict[int, int],
    expected_images: int,
) -> dict[str, Any]:
    """Metadata used to validate a detections cache."""
    return {
        "model": model_path.name,
        "conf_floor": conf_floor,
        "imgsz": imgsz,
        "class_mapping": {str(key): value for key, value in sorted(mapping.items())},
        "image_count": expected_images,
    }


def load_cache(path: Path, meta: dict[str, Any]) -> list[dict[str, Any]] | None:
    """Load a detections cache if its metadata matches; otherwise None."""
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (json.JSONDecodeError, OSError):
        return None
    if payload.get("meta") != meta:
        return None
    images = payload.get("images")
    if not isinstance(images, list) or len(images) != meta["image_count"]:
        return None
    return images


def save_cache(path: Path, meta: dict[str, Any], records: list[dict[str, Any]]) -> None:
    """Persist the inference cache (single JSON document)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": meta,
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "images": records,
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def prediction_dicts(
    records: list[dict[str, Any]], threshold: float
) -> list[dict[str, Any]]:
    """Flatten records into per-image prediction dicts at a threshold."""
    return [
        {
            "image_id": record["image_id"],
            "class_id": detection["class_id"],
            "score": detection["score"],
            "bbox": tuple(detection["bbox"]),
        }
        for record in records
        for detection in record["dets"]
        if detection["score"] >= threshold
    ]


def target_dicts(targets: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Flatten the target map into per-image target dicts."""
    return [
        {"image_id": image_id, "class_id": target["class_id"], "bbox": target["bbox"]}
        for image_id, image_targets in targets.items()
        for target in image_targets
    ]


def restrict_targets(
    targets: dict[str, list[dict[str, Any]]],
    records: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Keep only targets of images present in ``records`` (no-op if equal)."""
    image_ids = {record["image_id"] for record in records}
    if image_ids >= set(targets):
        return targets
    return {
        image_id: image_targets
        for image_id, image_targets in targets.items()
        if image_id in image_ids
    }


def compute_metrics(
    records: list[dict[str, Any]],
    targets: dict[str, list[dict[str, Any]]],
    project_classes: dict[int, str],
    primary_conf: float,
) -> dict[str, Any]:
    """Primary-conf P/R/F1 (greedy matching) and dataset-level mAP."""
    per_image: list[dict[str, Any]] = []
    for record in records:
        predictions = [
            {
                "class_id": detection["class_id"],
                "score": detection["score"],
                "bbox": tuple(detection["bbox"]),
            }
            for detection in record["dets"]
            if detection["score"] >= primary_conf
        ]
        per_image.append(
            match_image(predictions, targets.get(record["image_id"], []), MATCH_IOU)
        )
    aggregate = aggregate_matches(per_image)
    aps = average_precision(
        prediction_dicts(records, 0.0),
        target_dicts(targets),
    )
    maps = mean_average_precision(aps)
    per_class_rows: list[dict[str, Any]] = []
    for class_id, name in sorted(project_classes.items()):
        counts = aggregate["per_class"].get(
            class_id, {"tp": 0, "fp": 0, "fn": 0}
        )
        metrics = precision_recall_f1(counts["tp"], counts["fp"], counts["fn"])
        class_aps = aps.get(class_id, [None] * 10)
        valid_aps = [value for value in class_aps if value is not None]
        per_class_rows.append(
            {
                "class_id": class_id,
                "class_name": name,
                "ground_truth": counts["tp"] + counts["fn"],
                "tp": counts["tp"],
                "fp": counts["fp"],
                "fn": counts["fn"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "ap50": class_aps[0],
                "ap50_95": (
                    sum(valid_aps) / len(valid_aps) if valid_aps else None
                ),
            }
        )
    return {
        "counts": aggregate["counts"],
        "metrics": aggregate["metrics"],
        "map50": maps["map50"],
        "map50_95": maps["map50_95"],
        "per_class": per_class_rows,
        "per_image_matches": per_image,
    }


def confidence_table(
    records: list[dict[str, Any]],
    targets: dict[str, list[dict[str, Any]]],
    thresholds: tuple[float, ...] = CONFIDENCE_THRESHOLDS,
) -> list[dict[str, Any]]:
    """Micro P/R/F1 for each confidence threshold (offline, one pass)."""
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        counts = {"tp": 0, "fp": 0, "fn": 0}
        for record in records:
            predictions = [
                {
                    "class_id": detection["class_id"],
                    "score": detection["score"],
                    "bbox": tuple(detection["bbox"]),
                }
                for detection in record["dets"]
                if detection["score"] >= threshold
            ]
            matched = match_image(
                predictions, targets.get(record["image_id"], []), MATCH_IOU
            )
            for key in counts:
                counts[key] += matched[key]
        metrics = precision_recall_f1(counts["tp"], counts["fp"], counts["fn"])
        rows.append(
            {
                "threshold": threshold,
                **counts,
                **metrics,
                "predicted": counts["tp"] + counts["fp"],
                "ground_truth": counts["tp"] + counts["fn"],
            }
        )
    return rows


def speed_summary(
    records: list[dict[str, Any]],
    total_ms: float,
    model_load_ms: float,
    expected_images: int | None = None,
    warmup_images: int = 20,
) -> dict[str, Any]:
    """CPU speed statistics from a completed pass."""
    count = len(records)
    wall = [record["wall_ms"] for record in records]
    model_times = [record["infer_ms"] for record in records]
    warm = wall[warmup_images:] if count > warmup_images else wall
    first_wall_ms = wall[0] if wall else 0.0
    return {
        "device": "cpu",
        "image_count": count,
        "expected_image_count": expected_images,
        "total_inference_s": round(total_ms / 1000.0, 2),
        "model_load_ms": round(model_load_ms, 1),
        "first_image_ms": round(first_wall_ms, 1),
        "avg_wall_ms_per_image": round(total_ms / count, 1) if count else 0.0,
        "warm_avg_wall_ms_per_image": round(sum(warm) / len(warm), 1) if warm else 0.0,
        "avg_model_ms_per_image": (
            round(sum(model_times) / len(model_times), 1) if model_times else 0.0
        ),
        "images_per_second": round(1000.0 * count / total_ms, 2) if total_ms else 0.0,
        "warmup_excluded_images": min(warmup_images, max(count - len(warm), 0)),
    }


def analyze_failures(
    records: list[dict[str, Any]],
    targets: dict[str, list[dict[str, Any]]],
    project_classes: dict[int, str],
    primary_conf: float,
    max_examples: int = 5,
) -> dict[str, Any]:
    """Categorize baseline errors at the primary confidence threshold.

    Each ground-truth object that is not matched with IoU >= 0.5 is placed
    into exactly one primary category (wrong class, low confidence, poor
    localization, missed); each unmatched prediction becomes a false
    positive unless it duplicates or mis-localizes an existing object.
    Small-object and cluttered-scene tags are counted separately.
    """
    category_counts: Counter[str] = Counter()
    confusion: Counter[tuple[str, str]] = Counter()
    examples: dict[str, list[str]] = {}
    small_object_fn = 0
    total_fn = 0
    images_with_failures = 0

    def note(category: str, image_id: str) -> None:
        category_counts[category] += 1
        bucket = examples.setdefault(category, [])
        if image_id not in bucket and len(bucket) < max_examples:
            bucket.append(image_id)

    for record in records:
        image_id = record["image_id"]
        image_targets = targets.get(image_id, [])
        all_dets = record["dets"]
        primary_dets = [
            det for det in all_dets if det["score"] >= primary_conf
        ]
        matched = match_image(
            [
                {
                    "class_id": det["class_id"],
                    "score": det["score"],
                    "bbox": tuple(det["bbox"]),
                }
                for det in primary_dets
            ],
            image_targets,
            MATCH_IOU,
        )
        matched_targets = {
            detail["target"]
            for detail in matched["details"]
            if detail["verdict"] == "tp"
        }
        matched_preds = {
            detail["prediction"]
            for detail in matched["details"]
            if detail["verdict"] == "tp"
        }
        image_had_failure = False

        for target_index, target in enumerate(image_targets):
            if target_index in matched_targets:
                continue
            total_fn += 1
            target_class = project_classes[target["class_id"]]
            wrong_class_pred: dict[str, Any] | None = None
            for pred_index, det in enumerate(primary_dets):
                if pred_index in matched_preds:
                    continue
                if det["class_id"] == target["class_id"]:
                    continue
                if box_iou(tuple(det["bbox"]), target["bbox"]) >= 0.5:
                    wrong_class_pred = det
                    break
            if wrong_class_pred is not None:
                predicted_class = project_classes[wrong_class_pred["class_id"]]
                category_counts["wrong_class"] += 1
                bucket = examples.setdefault("wrong_class", [])
                if image_id not in bucket and len(bucket) < max_examples:
                    bucket.append(image_id)
                confusion[(target_class, predicted_class)] += 1
                image_had_failure = True
                continue
            low_conf_hit = False
            best_iou = 0.0
            for det in all_dets:
                iou = box_iou(tuple(det["bbox"]), target["bbox"])
                if iou > best_iou:
                    best_iou = iou
                if (
                    det["class_id"] == target["class_id"]
                    and det["score"] < primary_conf
                    and iou >= 0.5
                ):
                    low_conf_hit = True
            small = target["area"] < 0.01
            if small:
                small_object_fn += 1
            if low_conf_hit:
                note("low_confidence", image_id)
            elif best_iou >= 0.1:
                note("poor_localization", image_id)
            else:
                note("missed_detection", image_id)
            image_had_failure = True

        for pred_index, det in enumerate(primary_dets):
            if pred_index in matched_preds:
                continue
            best_iou = 0.0
            best_target: dict[str, Any] | None = None
            for target in image_targets:
                iou = box_iou(tuple(det["bbox"]), target["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_target = target
            if (
                best_target is not None
                and best_iou >= 0.5
                and image_targets.index(best_target) in matched_targets
            ):
                note("false_positive", image_id)
            elif best_target is not None and best_iou >= 0.5:
                continue
            elif best_iou >= 0.1:
                note("poor_localization", image_id)
            else:
                note("false_positive", image_id)
            image_had_failure = True

        if image_had_failure:
            images_with_failures += 1
            if len(image_targets) >= 5:
                category_counts["cluttered_scene"] += 1
                bucket = examples.setdefault("cluttered_scene", [])
                if image_id not in bucket and len(bucket) < max_examples:
                    bucket.append(image_id)

    return {
        "categories": dict(sorted(category_counts.items(), key=lambda kv: -kv[1])),
        "examples": examples,
        "confusion": [
            {"ground_truth": pair[0], "predicted": pair[1], "count": count}
            for pair, count in confusion.most_common()
        ],
        "total_fn": total_fn,
        "small_object_fn": small_object_fn,
        "images_with_failures": images_with_failures,
        "image_count": len(records),
    }


def image_lookup(images_dir: Path) -> dict[str, Path]:
    """Map image stem -> path for the test images directory."""
    return {
        path.stem: path
        for path in images_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    }


def render_sample(
    image_path: Path,
    targets: list[dict[str, Any]],
    detections: list[dict[str, Any]],
    class_names: dict[int, str],
    class_color,
) -> Image.Image:
    """Draw ground-truth (prefix GT) and primary-conf predictions (prefix P)."""
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=13)

    def label_box(text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
        text_y = max(0, y - 15)
        bounds = draw.textbbox((x, text_y), text, font=font)
        draw.rectangle(bounds, fill=color)
        draw.text((x, text_y), text, fill=(255, 255, 255), font=font)

    for target in targets:
        color = class_color(target["class_id"])
        x0, y0, x1, y1 = target["bbox"]
        px = [x0 * width, y0 * height, x1 * width, y1 * height]
        draw.rectangle(px, outline=color, width=3)
        label_box(
            f"GT: {class_names[target['class_id']]}",
            int(px[0]) + 2,
            int(px[1]),
            color,
        )
    for detection in detections:
        color = class_color(detection["class_id"])
        x0, y0, x1, y1 = detection["bbox"]
        px = [x0 * width, y0 * height, x1 * width, y1 * height]
        draw.rectangle(px, outline=color, width=1)
        label_box(
            f"P: {class_names[detection['class_id']]} {detection['score']:.2f}",
            int(px[0]) + 2,
            int(px[1]),
            color,
        )
    return image


def render_samples(
    picks: list[dict[str, str]],
    records_by_id: dict[str, dict[str, Any]],
    targets: dict[str, list[dict[str, Any]]],
    project_classes: dict[int, str],
    images_dir: Path,
    output_dir: Path,
    primary_conf: float,
) -> list[dict[str, Any]]:
    """Render the selected sample images with GT + predictions and index.md."""
    visualize = load_sibling("visualize_prepared_dataset.py")
    class_color = visualize.class_color
    lookup = image_lookup(images_dir)
    if output_dir.exists():
        for stale in output_dir.iterdir():
            if stale.is_file():
                stale.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for number, pick in enumerate(picks, start=1):
        image_id = pick["image_id"]
        image_path = lookup.get(image_id)
        if image_path is None:
            continue
        record = records_by_id[image_id]
        image_targets = targets.get(image_id, [])
        primary_dets = [
            det for det in record["dets"] if det["score"] >= primary_conf
        ]
        rendered = render_sample(
            image_path, image_targets, primary_dets, project_classes, class_color
        )
        filename = f"{number:02d}_{pick['category']}_{image_id}.jpg"
        rendered.save(output_dir / filename, quality=88)
        rows.append(
            {
                **pick,
                "file": filename,
                "ground_truth": len(image_targets),
                "predicted": len(primary_dets),
            }
        )
    lines = [
        "# Phase 5 baseline sample review",
        "",
        "Ground truth: thick box, prefix `GT`. Prediction: thin box, prefix `P`,",
        f"confidence shown for every detection at conf >= {primary_conf}.",
        "Selection is deterministic (seed from `configs/baseline.yaml`).",
        "",
        "| # | File | Category | Image | Note | GT | Pred |",
        "|---|------|----------|-------|------|----|------|",
    ]
    for number, row in enumerate(rows, start=1):
        lines.append(
            f"| {number} | {row['file']} | {row['category']} | {row['image_id']} "
            f"| {row['note']} | {row['ground_truth']} | {row['predicted']} |"
        )
    lines.append("")
    (output_dir / "index.md").write_text("\n".join(lines), encoding="utf-8")
    return rows


def write_results_json(path: Path, document: dict[str, Any]) -> None:
    """Write the machine-readable results document."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2)


def write_confidence_analysis(
    path: Path,
    rows: list[dict[str, Any]],
    primary_conf: float,
    image_count: int,
) -> None:
    """Write reports/baseline_confidence_analysis.md."""
    lines = [
        "# Phase 5 — Baseline Confidence Threshold Analysis",
        "",
        "All thresholds are computed offline from a single inference pass with",
        "detections kept down to a confidence floor of 0.01, so no model",
        "re-run is involved. Matching: greedy one-to-one, same class only,",
        f"IoU >= 0.5, over the {image_count} evaluated test images.",
        "",
        "| Confidence | Precision | Recall | F1 | TP | FP | FN | Predicted |",
        "|------------|-----------|--------|----|----|----|----|-----------|",
    ]
    for row in rows:
        lines.append(
            f"| {row['threshold']:.2f} | {row['precision']:.4f} | "
            f"{row['recall']:.4f} | {row['f1']:.4f} | {row['tp']} | "
            f"{row['fp']} | {row['fn']} | {row['predicted']} |"
        )
    lines.extend(["", f"Primary threshold for all other Phase 5 reports: "
                      f"**{primary_conf:.2f}**.", ""])
    baseline = next(
        (row for row in rows if abs(row["threshold"] - primary_conf) < 1e-9),
        rows[0],
    )
    for row in rows:
        if row is baseline:
            continue
        delta_p = row["precision"] - baseline["precision"]
        delta_r = row["recall"] - baseline["recall"]
        direction_p = "higher" if delta_p >= 0 else "lower"
        direction_r = "higher" if delta_r >= 0 else "lower"
        lines.append(
            f"- At {row['threshold']:.2f}: precision is {abs(delta_p):.4f} "
            f"{direction_p} and recall is {abs(delta_r):.4f} {direction_r} "
            f"than at {primary_conf:.2f} "
            f"(F1 {baseline['f1']:.4f} -> {row['f1']:.4f})."
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_error_examples(
    path: Path,
    failures: dict[str, Any],
    primary_conf: float,
) -> None:
    """Write reports/baseline_error_examples.md."""
    lines = [
        "# Phase 5 — Baseline Error Examples",
        "",
        f"Errors categorized at confidence >= {primary_conf:.2f} with greedy",
        f"same-class matching at IoU >= 0.5 over {failures['image_count']}",
        "evaluated test images.",
        "Each missed ground-truth object lands in exactly one primary",
        "category; duplicate/spurious predictions count as false positives.",
        "",
        "## Error categories",
        "",
        "| Category | Count | Example image IDs |",
        "|----------|-------|-------------------|",
    ]
    categories = failures["categories"]
    for category, count in categories.items():
        example_ids = ", ".join(failures["examples"].get(category, [])) or "-"
        lines.append(f"| {category} | {count} | {example_ids} |")
    if not categories:
        lines.append("| - | 0 | - |")
    small_share = (
        failures["small_object_fn"] / failures["total_fn"]
        if failures["total_fn"]
        else 0.0
    )
    lines.extend(
        [
            "",
            "## Missed-object detail",
            "",
            f"- Ground-truth objects missed at the primary threshold: "
            f"{failures['total_fn']}",
            f"- Of those, small objects (box area < 1% of image): "
            f"{failures['small_object_fn']} ({small_share * 100:.1f}%)",
            f"- Test images with at least one categorized failure: "
            f"{failures['images_with_failures']} / {failures['image_count']}",
            "",
            "## Top class confusions (wrong class at IoU >= 0.5)",
            "",
            "| Ground truth | Predicted | Count |",
            "|--------------|-----------|-------|",
        ]
    )
    confusion = failures["confusion"]
    for entry in confusion[:15]:
        lines.append(
            f"| {entry['ground_truth']} | {entry['predicted']} | {entry['count']} |"
        )
    if not confusion:
        lines.append("| - | - | 0 |")
    lines.extend(
        [
            "",
            "`cluttered_scene` is an image-level tag: images with 5+",
            "ground-truth objects that contain at least one failure (it",
            "overlaps the object-level categories above).",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_report(path: Path, ctx: dict[str, Any]) -> None:
    """Write reports/baseline_report.md (the twelve Phase 5 sections)."""
    counts = ctx["counts"]
    metrics = ctx["metrics"]
    speed = ctx["speed"]
    failures = ctx["failures"]
    lines: list[str] = [
        "# Phase 5 — Baseline Model Report",
        "",
        "Zero-shot evaluation of the COCO-pretrained lightweight YOLO model",
        "on the project test split, before any custom training (Phase 6).",
        f"Generated {ctx['generated']} by `scripts/evaluate_baseline.py`.",
        "",
        "## 1. Overview",
        "",
        "| Item | Value |",
        "|------|-------|",
        f"| Model | {ctx['model_name']} (COCO-pretrained, no fine-tuning) |",
        f"| Test images | {ctx['test_images']} |",
        f"| Ground-truth instances | {ctx['test_instances']} |",
        f"| Classes | {ctx['class_count']} household objects |",
        f"| Confidence threshold | {ctx['primary_conf']:.2f} |",
        f"| Precision | {metrics['precision']:.4f} |",
        f"| Recall | {metrics['recall']:.4f} |",
        f"| F1 | {metrics['f1']:.4f} |",
        f"| mAP@0.5 | {ctx['map50']:.4f} |",
        f"| mAP@0.5:0.95 | {ctx['map50_95']:.4f} |",
        f"| Speed (CPU, warm average) | {speed['warm_avg_wall_ms_per_image']} ms/image |",
        "",
        "## 2. Model",
        "",
        f"- Weights: `{ctx['model_path']}` — {ctx['model_name']} checkpoint,",
        f"  {ctx['model_version']}.",
        "- Architecture: YOLO26-nano detection model from Ultralytics with an",
        "  end-to-end (NMS-free) head; 80 COCO output classes.",
        f"- Runtime: ultralytics {ctx['ultralytics_version']},",
        f"  torch {ctx['pytorch_version']}, device `{ctx['device']}` (CPU only —",
        "  no CUDA GPU on this machine).",
        "- Pretrained on the COCO detection dataset; **not** trained or",
        "  fine-tuned on project data. Results are a true zero-shot baseline.",
        "- Checkpoint license: AGPL-3.0 (Ultralytics).",
        "",
        "## 3. Dataset and Class Mapping",
        "",
        f"- Data: `data/processed/household_objects` test split —",
        f"  {ctx['test_images']} images / {ctx['test_instances']} instances",
        f"  across {ctx['class_count']} classes (train/val were **not** touched).",
        "- Mapping: name-based — each model COCO class name that equals a",
        "  project class name in `configs/classes.yaml` is mapped to that",
        "  project class id.",
        f"- Classes mapped: {ctx['mapping_size']} / {ctx['class_count']} "
        f"project classes present in the model's output space.",
        f"- Detections at conf >= {ctx['conf_floor']:.2f}: {ctx['kept_det_total']}"
        f" of {ctx['raw_det_total']} raw predictions belong to the 19 project",
        f"  classes; {ctx['ignored_det_total']} "
        f"({ctx['ignored_share'] * 100:.1f}%) of other COCO classes were",
        "  ignored (they are not counted as false positives).",
        "",
        "## 4. Evaluation Protocol",
        "",
        f"- Single streamed inference pass over the test split,",
        f"  `imgsz={ctx['imgsz']}`, `conf>={ctx['conf_floor']:.2f}`,",
        f"  `max_det={ctx['max_dets']}`, CPU; all thresholds below are",
        "  computed offline from that one pass.",
        f"- Detection matching: greedy one-to-one per image and class in",
        f"  descending confidence order; a true positive needs same class and",
        f"  IoU >= {ctx['match_iou']:.2f}.",
        "- Precision / recall / F1 are micro-averaged over the whole split at",
        f"  the primary threshold ({ctx['primary_conf']:.2f}).",
        "- mAP is implemented in-repo (`src/evaluation/baseline.py`):",
        "  score-ordered greedy matching, COCO-style 101-point interpolated",
        "  precision envelope; mAP@0.5 = AP at IoU 0.50, mAP@0.5:0.95 = mean",
        "  AP over IoU 0.50:0.05:0.95; per-class AP is `None` when the class",
        "  has no ground truth and is excluded from the mAP mean. Classes",
        "  without ground truth therefore do not lower the reported mAP.",
        "- Evaluation is class-restricted by design: only the 19 project",
        "  classes participate, so COCO-only objects (e.g. person) neither",
        "  help nor hurt the metrics.",
        "",
        "## 5. Overall Metrics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| TP / FP / FN | {counts['tp']} / {counts['fp']} / {counts['fn']} |",
        f"| Precision | {metrics['precision']:.4f} |",
        f"| Recall | {metrics['recall']:.4f} |",
        f"| F1 | {metrics['f1']:.4f} |",
        f"| mAP@0.5 | {ctx['map50']:.4f} |",
        f"| mAP@0.5:0.95 | {ctx['map50_95']:.4f} |",
        "",
        "## 6. Per-Class Metrics",
        "",
        "| Class | GT | TP | FP | FN | Precision | Recall | F1 | AP50 | AP50-95 |",
        "|-------|----|----|----|----|-----------|--------|----|------|---------|",
    ]
    for row in ctx["per_class"]:
        ap50 = f"{row['ap50']:.4f}" if row["ap50"] is not None else "-"
        ap5095 = (
            f"{row['ap50_95']:.4f}" if row["ap50_95"] is not None else "-"
        )
        lines.append(
            f"| {row['class_name']} | {row['ground_truth']} | {row['tp']} | "
            f"{row['fp']} | {row['fn']} | {row['precision']:.4f} | "
            f"{row['recall']:.4f} | {row['f1']:.4f} | {ap50} | {ap5095} |"
        )
    lines.extend(
        [
            "",
            "## 7. Confidence Threshold Analysis",
            "",
            "| Confidence | Precision | Recall | F1 | TP | FP | FN |",
            "|------------|-----------|--------|----|----|----|----|",
        ]
    )
    for row in ctx["conf_rows"]:
        lines.append(
            f"| {row['threshold']:.2f} | {row['precision']:.4f} | "
            f"{row['recall']:.4f} | {row['f1']:.4f} | {row['tp']} | "
            f"{row['fp']} | {row['fn']} |"
        )
    lines.extend(
        [
            "",
            "Full write-up: `reports/baseline_confidence_analysis.md`.",
            "",
            "## 8. Inference Speed",
            "",
            "| Item | Value |",
            "|------|-------|",
            f"| Device | {speed['device']} (no CUDA GPU available) |",
            f"| Images processed | {speed['image_count']} |",
            f"| Total inference time | {speed['total_inference_s']} s |",
            f"| Average wall time | {speed['avg_wall_ms_per_image']} ms/image |",
            f"| Warm average (first {speed['warmup_excluded_images']} excluded) | "
            f"{speed['warm_avg_wall_ms_per_image']} ms/image |",
            f"| Model-reported inference | {speed['avg_model_ms_per_image']} ms/image |",
            f"| Throughput | {speed['images_per_second']} images/s |",
            f"| Model load (one time) | {speed['model_load_ms']} ms |",
            f"| First image (warmup) | {speed['first_image_ms']} ms |",
            "",
            "Wall time includes decode, pre- and post-processing; model-reported",
            "time is the kernel-only figure from Ultralytics. Figures are from",
            "this machine and will differ elsewhere.",
            "",
            "## 9. Error Analysis Summary",
            "",
            "| Category | Count |",
            "|----------|-------|",
        ]
    )
    if failures["categories"]:
        for category, count in failures["categories"].items():
            lines.append(f"| {category} | {count} |")
    else:
        lines.append("| - | 0 |")
    lines.extend(
        [
            "",
            f"- Missed ground-truth objects: {failures['total_fn']} "
            f"(small objects: {failures['small_object_fn']})",
            f"- Images with at least one failure: "
            f"{failures['images_with_failures']} / {failures['image_count']}",
            f"- Top class confusion: "
            + (
                f"{failures['confusion'][0]['ground_truth']} -> "
                f"{failures['confusion'][0]['predicted']} "
                f"({failures['confusion'][0]['count']}x)"
                if failures["confusion"]
                else "none"
            ),
            "",
            "Details and example IDs: `reports/baseline_error_examples.md`.",
            "",
            "## 10. Sample Review",
            "",
            f"{len(ctx['samples'])} annotated test images "
            "(ground truth + predictions, deterministic selection, seed "
            f"{ctx['seed']}) in `reports/baseline_samples/` with `index.md`.",
            "",
        ]
    )
    for number, row in enumerate(ctx["samples"], start=1):
        lines.append(
            f"{number}. `{row['file']}` — {row['category']}: {row['note']} "
            f"(GT {row['ground_truth']}, predicted {row['predicted']})"
        )
    lines.extend(
        [
            "",
            "## 11. Limitations",
            "",
            "- The baseline has never seen the project's curated 19-class data;",
            "  per-class AP mostly reflects COCO coverage of that class name.",
            "- mAP is computed by the in-repo implementation (pycocotools is",
            "  not installed); conventions are documented in section 4.",
            "- Evaluation is class-restricted: detections of non-project COCO",
            "  classes are ignored rather than penalized.",
            "- Timings are CPU-only on one machine; `torch.set_num_threads` was",
            "  left at its default.",
            "- The confidence table reuses one pass, so thresholds below the",
            f"  capture floor ({ctx['conf_floor']:.2f}) are not observable.",
            "",
            "## 12. Reproduction and Next Steps",
            "",
            "Commands (from the project root):",
            "",
            "```powershell",
            r"python scripts\evaluate_baseline.py",
            r"python scripts\baseline_inference.py --source <image> --save",
            "```",
            "",
            "Artifacts: `reports/baseline_results.json` (machine-readable),",
            "`reports/baseline_confidence_analysis.md`,",
            "`reports/baseline_error_examples.md`,",
            "`reports/baseline_samples/`,",
            "`reports/baseline_commands.md`.",
            "",
            "Next: Phase 6 trains a custom model on the train split and",
            "compares against these numbers on the same test split.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_commands_doc(path: Path, config_rel: str, imgsz: int, conf_floor: float) -> None:
    """Write reports/baseline_commands.md (reproduction reference)."""
    lines = [
        "# Phase 5 — Reproduction Commands",
        "",
        "Run from the project root. Use the project virtual environment:",
        "",
        "```powershell",
        r'$env:Path = ".venv\Scripts";$env:Path',
        "```",
        "",
        "## Single-image baseline inference",
        "",
        "```powershell",
        r"python scripts\baseline_inference.py --source data\processed\household_objects\images\test\000000000139.jpg --save",
        r"python scripts\baseline_inference.py --source path\to\image.jpg --conf 0.5 --save",
        "```",
        "",
        "Prints one line per detection (confidence, class, pixel box), the",
        "inference time, and with `--save` stores the annotated copy under",
        "`examples/output/`.",
        "",
        "## Full test-split evaluation (Phase 5 report suite)",
        "",
        "```powershell",
        f"python scripts\\evaluate_baseline.py --config {config_rel}",
        "```",
        "",
        "- Model: `models/baseline/yolo26n.pt` (auto-downloaded when absent)",
        f"- Inference: CPU, imgsz={imgsz}, conf floor {conf_floor}, one pass,",
        "  then all thresholds computed offline.",
        "- Runtime: roughly 10-15 minutes on a laptop CPU (about 0.4 s per",
        "  image including decode); progress is shown as a tqdm bar with",
        "  count, percent, elapsed and ETA.",
        "- The detections cache `reports/baseline_detections.json` is written",
        "  after a full pass; reruns reuse it and skip inference.",
        "",
        "Force a fresh pass:",
        "",
        "```powershell",
        "python scripts\\evaluate_baseline.py --recompute",
        "```",
        "",
        "## Outputs",
        "",
        "```text",
        "reports/baseline_report.md",
        "reports/baseline_results.json",
        "reports/baseline_confidence_analysis.md",
        "reports/baseline_error_examples.md",
        "reports/baseline_samples/",
        "reports/baseline_commands.md",
        "```",
        "",
        "## Tests",
        "",
        "```powershell",
        "python -m pytest tests -q",
        "```",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the COCO-pretrained baseline on the test split."
    )
    parser.add_argument(
        "--config",
        default="configs/baseline.yaml",
        help="baseline YAML config (default: configs/baseline.yaml)",
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="ignore the detections cache and rerun inference",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="debug only: evaluate the first N images (never for reports)",
    )
    return parser.parse_args(argv)


def fail(message: str) -> None:
    """Abort with a clear error message."""
    raise SystemExit(f"error: {message}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    project_classes = load_project_classes(config["data"]["classes_config"])
    class_count = len(project_classes)
    expected = config["expected"]
    if class_count != expected["class_count"]:
        fail(
            f"class count {class_count} does not match config "
            f"{expected['class_count']}"
        )
    images_dir = resolve_path(config["data"]["test_images"])
    labels_dir = resolve_path(config["data"]["test_labels"])
    image_count = count_images(images_dir)
    targets, instance_count = load_targets(labels_dir, class_count)
    if image_count != expected["test_image_count"]:
        fail(
            f"test image count {image_count} != expected "
            f"{expected['test_image_count']}"
        )
    if instance_count != expected["test_instance_count"]:
        fail(
            f"test instance count {instance_count} != expected "
            f"{expected['test_instance_count']}"
        )
    image_stems = set(image_lookup(images_dir))
    if set(targets) != image_stems:
        fail("test label stems do not match test image files")

    inference = config["inference"]
    evaluation = config["evaluation"]
    primary_conf = float(evaluation["primary_conf"])
    conf_floor = float(inference["conf_floor"])
    imgsz = int(inference["imgsz"])
    device = str(inference["device"])
    thresholds = tuple(float(value) for value in evaluation["confidence_thresholds"])

    inference_module = load_sibling("baseline_inference.py")
    model_path = inference_module.ensure_model(resolve_path(config["model"]["path"]))
    model, model_load_ms = load_model(model_path)
    mapping = build_class_mapping(dict(model.names), project_classes)
    if len(mapping) != class_count:
        fail(
            f"only {len(mapping)}/{class_count} project classes found in the "
            "model output space"
        )

    output = config["output"]
    cache_path = resolve_path(output["detections_cache"])
    meta = cache_meta(model_path, conf_floor, imgsz, mapping, image_count)
    use_cache = not args.recompute and args.limit is None
    records: list[dict[str, Any]] | None = None
    cache_used = False
    if use_cache:
        records = load_cache(cache_path, meta)
        cache_used = records is not None
    total_ms = 0.0
    if records is None:
        image_paths = sorted(image_lookup(images_dir).values())
        if args.limit is not None:
            image_paths = image_paths[: max(args.limit, 1)]
            print(
                f"WARNING: --limit {args.limit} partial debug run "
                "(results are not the official baseline)",
                file=sys.stderr,
            )
        records, total_ms = run_test_pass(
            model,
            image_paths,
            mapping,
            conf_floor,
            imgsz,
            device,
            int(inference["max_dets_per_image"]),
        )
        if args.limit is None:
            save_cache(cache_path, meta, records)
    else:
        total_ms = sum(record["wall_ms"] for record in records)
        print(f"reusing detections cache: {cache_path.relative_to(PROJECT_ROOT)}")
    if args.limit is None and len(records) != image_count:
        fail(f"processed {len(records)} images, expected {image_count}")

    records_by_id = {record["image_id"]: record for record in records}
    active_targets = restrict_targets(targets, records)
    metrics = compute_metrics(
        records, active_targets, project_classes, primary_conf
    )
    conf_rows = confidence_table(records, active_targets, thresholds)
    speed = speed_summary(
        records, total_ms, model_load_ms, expected_images=image_count
    )
    failures = analyze_failures(
        records, active_targets, project_classes, primary_conf
    )

    sample_records = [
        {
            "image_id": image_id,
            "gt_boxes": [
                {
                    "class_id": target["class_id"],
                    "class_name": project_classes[target["class_id"]],
                    "area": target["area"],
                }
                for target in image_targets
            ],
            "pred_count": sum(
                1
                for det in records_by_id.get(image_id, {"dets": []})["dets"]
                if det["score"] >= primary_conf
            ),
        }
        for image_id, image_targets in sorted(active_targets.items())
        if image_id in records_by_id
    ]
    picks = select_samples(
        sample_records,
        project_classes,
        count=int(config["samples"]["count"]),
        seed=int(config["samples"]["seed"]),
    )
    sample_rows = render_samples(
        picks,
        records_by_id,
        active_targets,
        project_classes,
        images_dir,
        resolve_path(output["samples_dir"]),
        primary_conf,
    )

    raw_det_total = sum(record["raw_count"] for record in records)
    kept_det_total = sum(len(record["dets"]) for record in records)
    ignored_det_total = raw_det_total - kept_det_total
    ignored_share = ignored_det_total / raw_det_total if raw_det_total else 0.0

    import torch
    import ultralytics

    generated = datetime.now(timezone.utc).isoformat(timespec="seconds")
    document = build_results_json(
        model=str(config["model"]["path"]),
        model_version="checkpoint saved 2025-12-15 (ultralytics 8.3.222)",
        ultralytics_version=ultralytics.__version__,
        pytorch_version=torch.__version__,
        device=device,
        test_image_count=len(records),
        test_instance_count=instance_count,
        metrics={
            "primary_conf": primary_conf,
            "match_iou": MATCH_IOU,
            **metrics["counts"],
            **metrics["metrics"],
            "map50": round(metrics["map50"], 6),
            "map50_95": round(metrics["map50_95"], 6),
        },
        per_class_metrics=metrics["per_class"],
        confidence_threshold_results={
            f"{row['threshold']:.2f}": row for row in conf_rows
        },
        timing=speed,
        class_mapping={
            "mechanism": "name_based_coco_to_project",
            "mapped_classes": len(mapping),
            "coco_to_project": {
                str(key): value for key, value in sorted(mapping.items())
            },
        },
        timestamp=generated,
    )
    document["failure_analysis"] = failures
    document["samples"] = sample_rows
    document["detections"] = {
        "raw_total": raw_det_total,
        "project_class_total": kept_det_total,
        "ignored_non_project_total": ignored_det_total,
        "conf_floor": conf_floor,
        "imgsz": imgsz,
    }
    document["cache_used"] = cache_used

    report_context = {
        "generated": generated,
        "model_name": config["model"]["name"],
        "model_path": config["model"]["path"],
        "model_version": document["model_version"],
        "ultralytics_version": ultralytics.__version__,
        "pytorch_version": torch.__version__,
        "device": device,
        "imgsz": imgsz,
        "conf_floor": conf_floor,
        "primary_conf": primary_conf,
        "match_iou": MATCH_IOU,
        "max_dets": int(inference["max_dets_per_image"]),
        "test_images": image_count,
        "test_instances": instance_count,
        "class_count": class_count,
        "mapping_size": len(mapping),
        "raw_det_total": raw_det_total,
        "kept_det_total": kept_det_total,
        "ignored_det_total": ignored_det_total,
        "ignored_share": ignored_share,
        "counts": metrics["counts"],
        "metrics": metrics["metrics"],
        "map50": metrics["map50"],
        "map50_95": metrics["map50_95"],
        "per_class": metrics["per_class"],
        "conf_rows": conf_rows,
        "speed": speed,
        "failures": failures,
        "samples": sample_rows,
        "seed": int(config["samples"]["seed"]),
    }
    write_results_json(resolve_path(output["results_json"]), document)
    write_confidence_analysis(
        resolve_path(output["confidence_analysis"]),
        conf_rows,
        primary_conf,
        len(records),
    )
    write_error_examples(resolve_path(output["error_examples"]), failures, primary_conf)
    write_report(resolve_path(output["report"]), report_context)
    write_commands_doc(
        resolve_path(output["commands"]), args.config, imgsz, conf_floor
    )

    overall = metrics["metrics"]
    print(
        f"images: {len(records)}  instances: {instance_count}  "
        f"classes: {class_count}"
    )
    print(
        f"precision {overall['precision']:.4f}  recall {overall['recall']:.4f}  "
        f"f1 {overall['f1']:.4f}  map50 {metrics['map50']:.4f}  "
        f"map50-95 {metrics['map50_95']:.4f}"
    )
    print(
        f"speed: {speed['avg_wall_ms_per_image']} ms/image avg, "
        f"{speed['warm_avg_wall_ms_per_image']} ms warm, "
        f"{speed['total_inference_s']} s total, cache_used={cache_used}"
    )
    print("outputs:")
    for key in (
        "report",
        "results_json",
        "confidence_analysis",
        "error_examples",
        "commands",
        "samples_dir",
    ):
        print(f"  {output[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
