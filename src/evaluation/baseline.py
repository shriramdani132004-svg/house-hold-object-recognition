"""Reusable baseline-evaluation helpers for Phase 5.

Boxes are normalized ``(x1, y1, x2, y2)`` coordinates (0-1) so predictions
and YOLO ground truth share one coordinate space. Metrics follow the
conventions documented in ``reports/baseline_report.md``:

- class-restricted evaluation (only the 19 project classes; detections of
  other COCO classes are ignored, not counted as false positives),
- greedy one-to-one matching in descending confidence order per image and
  class,
- AP in COCO style: score-ordered accumulation with a 101-point
  interpolated precision envelope over IoU thresholds 0.50:0.05:0.95.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]

Box = tuple[float, float, float, float]

CONFIDENCE_THRESHOLDS = (0.25, 0.50, 0.75)
MAP_IOU_THRESHOLDS = (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)
MATCH_IOU = 0.5
MAX_DETS = 100


def resolve_path(value: str | Path) -> Path:
    """Resolve a config path relative to the project root."""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a Phase 5 YAML configuration file."""
    config_path = resolve_path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"baseline config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{config_path}: expected a YAML mapping")
    return data


def load_project_classes(classes_path: str | Path) -> dict[int, str]:
    """Load the ordered project class map (IDs 0..n-1) from classes.yaml."""
    config_path = resolve_path(classes_path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    classes = {int(key): str(value) for key, value in data["classes"].items()}
    if sorted(classes) != list(range(len(classes))):
        raise ValueError(f"{config_path}: class ids must be consecutive from 0")
    return classes


def build_class_mapping(
    coco_names: dict[int, str], project_classes: dict[int, str]
) -> dict[int, int]:
    """Map COCO model class IDs to project class IDs by matching names."""
    project_by_name = {name: class_id for class_id, name in project_classes.items()}
    return {
        coco_id: project_by_name[name]
        for coco_id, name in coco_names.items()
        if name in project_by_name
    }


def filter_detections(
    detections: Sequence[dict[str, Any]], mapping: dict[int, int]
) -> list[dict[str, Any]]:
    """Keep only detections of project classes and re-key them by project id.

    Each input detection is a mapping with ``coco_class``, ``score`` and a
    normalized ``bbox``. Detections whose COCO class is not part of the 19
    project classes are dropped entirely.
    """
    kept: list[dict[str, Any]] = []
    for detection in detections:
        project_id = mapping.get(int(detection["coco_class"]))
        if project_id is None:
            continue
        kept.append(
            {
                "class_id": project_id,
                "score": float(detection["score"]),
                "bbox": tuple(float(value) for value in detection["bbox"]),
            }
        )
    return kept


def box_iou(box_a: Box, box_b: Box) -> float:
    """Intersection-over-union of two ``(x1, y1, x2, y2)`` boxes."""
    inter_x1 = max(box_a[0], box_b[0])
    inter_y1 = max(box_a[1], box_b[1])
    inter_x2 = min(box_a[2], box_b[2])
    inter_y2 = min(box_a[3], box_b[3])
    inter = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def precision_recall_f1(
    tp: int, fp: int, fn: int
) -> dict[str, float]:
    """Micro precision/recall/F1 from raw counts (undefined cases -> 0.0)."""
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def match_image(
    predictions: Sequence[dict[str, Any]],
    targets: Sequence[dict[str, Any]],
    iou_threshold: float = MATCH_IOU,
) -> dict[str, Any]:
    """Greedy class-aware matching for one image.

    Predictions are processed in descending confidence order; each target
    can be matched at most once and only with a same-class prediction whose
    IoU reaches ``iou_threshold``.
    """
    order = sorted(range(len(predictions)), key=lambda index: -predictions[index]["score"])
    matched_targets: set[int] = set()
    tp = fp = 0
    per_class: dict[int, dict[str, int]] = {}

    def slot(class_id: int) -> dict[str, int]:
        return per_class.setdefault(class_id, {"tp": 0, "fp": 0, "fn": 0})

    details: list[dict[str, Any]] = []
    for index in order:
        prediction = predictions[index]
        class_id = prediction["class_id"]
        best_iou = 0.0
        best_target = -1
        for target_index, target in enumerate(targets):
            if target_index in matched_targets:
                continue
            if target["class_id"] != class_id:
                continue
            iou = box_iou(prediction["bbox"], target["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_target = target_index
        if best_target >= 0 and best_iou >= iou_threshold:
            matched_targets.add(best_target)
            tp += 1
            slot(class_id)["tp"] += 1
            details.append(
                {
                    "prediction": index,
                    "target": best_target,
                    "iou": round(best_iou, 4),
                    "verdict": "tp",
                }
            )
        else:
            fp += 1
            slot(class_id)["fp"] += 1
            details.append(
                {
                    "prediction": index,
                    "target": None,
                    "iou": round(best_iou, 4) if best_target >= 0 else 0.0,
                    "verdict": "fp",
                }
            )
    fn = len(targets) - len(matched_targets)
    for target_index, target in enumerate(targets):
        if target_index not in matched_targets:
            slot(target["class_id"])["fn"] += 1
    return {"tp": tp, "fp": fp, "fn": fn, "per_class": per_class, "details": details}


def aggregate_matches(per_image: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Sum per-image matching results into overall and per-class counts."""
    overall = {"tp": 0, "fp": 0, "fn": 0}
    per_class: dict[int, dict[str, int]] = {}
    for result in per_image:
        for key in ("tp", "fp", "fn"):
            overall[key] += result[key]
        for class_id, counts in result["per_class"].items():
            slot = per_class.setdefault(class_id, {"tp": 0, "fp": 0, "fn": 0})
            for key in ("tp", "fp", "fn"):
                slot[key] += counts[key]
    metrics = precision_recall_f1(overall["tp"], overall["fp"], overall["fn"])
    per_class_metrics = {
        class_id: counts | precision_recall_f1(**counts)
        for class_id, counts in sorted(per_class.items())
    }
    return {"counts": overall, "metrics": metrics, "per_class": per_class_metrics}


def average_precision(
    detections: Sequence[dict[str, Any]],
    targets: Sequence[dict[str, Any]],
    iou_thresholds: Sequence[float] = MAP_IOU_THRESHOLDS,
    max_dets: int = MAX_DETS,
) -> dict[int, list[float | None]]:
    """COCO-style per-class AP over multiple IoU thresholds.

    Detections are ranked by score across the whole dataset; matching is
    greedy and one-to-one per image, class and IoU threshold. AP uses the
    COCO 101-point interpolated precision envelope. Classes without any
    ground truth get ``None`` (excluded from the mAP mean).
    """
    images = {record["image_id"] for record in detections} | {
        record["image_id"] for record in targets
    }
    targets_by_image: dict[str, list[dict[str, Any]]] = {image: [] for image in images}
    for target in targets:
        targets_by_image[target["image_id"]].append(target)
    detections_by_image: dict[str, list[dict[str, Any]]] = {image: [] for image in images}
    for detection in detections:
        detections_by_image[detection["image_id"]].append(detection)
    for image in images:
        detections_by_image[image].sort(key=lambda record: -record["score"])
        detections_by_image[image] = detections_by_image[image][:max_dets]

    class_ids = sorted(
        {record["class_id"] for record in detections}
        | {record["class_id"] for record in targets}
    )
    gt_counts = {
        class_id: sum(
            1 for target in targets if target["class_id"] == class_id
        )
        for class_id in class_ids
    }
    ranked_ids = {
        id(detection)
        for image in images
        for detection in detections_by_image[image]
    }
    ranked = sorted(
        (detection for detection in detections if id(detection) in ranked_ids),
        key=lambda record: -record["score"],
    )

    results: dict[int, list[float | None]] = {}
    for class_id in class_ids:
        aps: list[float | None] = []
        class_detections = [
            detection for detection in ranked if detection["class_id"] == class_id
        ]
        for threshold in iou_thresholds:
            if gt_counts[class_id] == 0:
                aps.append(None)
                continue
            matched: dict[str, set[int]] = {image: set() for image in images}
            true_positives: list[int] = []
            false_positives: list[int] = []
            for detection in class_detections:
                best_iou = 0.0
                best_index = -1
                for index, target in enumerate(
                    targets_by_image[detection["image_id"]]
                ):
                    if target["class_id"] != class_id or index in matched[
                        detection["image_id"]
                    ]:
                        continue
                    iou = box_iou(detection["bbox"], target["bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        best_index = index
                if best_index >= 0 and best_iou >= threshold:
                    matched[detection["image_id"]].add(best_index)
                    true_positives.append(1)
                    false_positives.append(0)
                else:
                    true_positives.append(0)
                    false_positives.append(1)
            cumulative_tp = 0
            cumulative_fp = 0
            recalls: list[float] = []
            precisions: list[float] = []
            for tp_flag, fp_flag in zip(true_positives, false_positives):
                cumulative_tp += tp_flag
                cumulative_fp += fp_flag
                recalls.append(cumulative_tp / gt_counts[class_id])
                precisions.append(
                    cumulative_tp / (cumulative_tp + cumulative_fp)
                    if cumulative_tp + cumulative_fp
                    else 0.0
                )
            ap = 0.0
            for step in range(101):
                recall_point = step / 100.0
                candidates = [
                    precision
                    for precision, recall in zip(precisions, recalls)
                    if recall >= recall_point
                ]
                ap += max(candidates) if candidates else 0.0
            aps.append(ap / 101.0)
        results[class_id] = aps
    return results


def mean_average_precision(
    aps: dict[int, list[float | None]]
) -> dict[str, float]:
    """Average AP values across classes (None entries excluded)."""
    per_class: list[list[float]] = [
        [value for value in values if value is not None]
        for values in aps.values()
    ]
    per_class = [values for values in per_class if values]
    map50 = (
        sum(values[0] for values in per_class) / len(per_class)
        if per_class
        else 0.0
    )
    map50_95 = (
        sum(sum(values) / len(values) for values in per_class) / len(per_class)
        if per_class
        else 0.0
    )
    return {"map50": map50, "map50_95": map50_95}


def select_samples(
    records: Sequence[dict[str, Any]],
    project_classes: dict[int, str],
    count: int = 20,
    seed: int = 42,
) -> list[dict[str, str]]:
    """Pick a deterministic, diverse set of test images for review.

    Each record needs ``image_id``, ``gt_boxes`` (list of
    ``{"class_id", "area"}``) and is assumed to be a test-set image. The
    selection covers crowded, single-object, smallest-box and largest-box
    cases, one example per reachable class, then seeded random fill.
    """
    rng = random.Random(seed)
    ordered = sorted(records, key=lambda record: record["image_id"])
    picks: list[dict[str, str]] = []
    taken: set[str] = set()

    def add(category: str, record: dict[str, Any], note: str) -> None:
        picks.append(
            {"category": category, "image_id": record["image_id"], "note": note}
        )
        taken.add(record["image_id"])

    def free_records() -> list[dict[str, Any]]:
        return [
            record for record in ordered if record["image_id"] not in taken
        ]

    def first_free(predicate) -> dict[str, Any] | None:
        for record in free_records():
            if predicate(record):
                return record
        return None

    non_empty = [record for record in ordered if record["gt_boxes"]]
    if not non_empty:
        return picks

    crowded = max(non_empty, key=lambda record: len(record["gt_boxes"]))
    add(
        "crowded",
        crowded,
        f"most ground-truth objects ({len(crowded['gt_boxes'])})",
    )

    single = first_free(lambda record: len(record["gt_boxes"]) == 1)
    if single is not None:
        add("single_object", single, "exactly one ground-truth object")

    with_smallest = [
        record
        for record in non_empty
        if record["image_id"] not in taken
    ]
    if with_smallest:
        smallest = min(
            with_smallest,
            key=lambda record: min(box["area"] for box in record["gt_boxes"]),
        )
        area = min(box["area"] for box in smallest["gt_boxes"])
        add("small_object", smallest, f"smallest ground-truth box (area {area:.6f})")
        largest = max(
            with_smallest,
            key=lambda record: max(box["area"] for box in record["gt_boxes"]),
        )
        area = max(box["area"] for box in largest["gt_boxes"])
        add("large_object", largest, f"largest ground-truth box (area {area:.3f})")

    class_budget = max(0, count - 6)
    added_classes = 0
    for _class_id, name in sorted(project_classes.items()):
        if len(picks) >= class_budget:
            break
        record = first_free(
            lambda record, name=name: any(
                box.get("class_name") == name for box in record["gt_boxes"]
            )
        )
        if record is None:
            continue
        add(f"class_{name.replace(' ', '_')}", record, f"contains '{name}'")
        added_classes += 1
    if added_classes == 0 and len(picks) < count:
        for record in free_records():
            if any(box.get("class_name") for box in record["gt_boxes"]):
                add("annotated", record, "ground-truth annotated scene")
                break

    remaining = free_records()
    rng.shuffle(remaining)
    for record in remaining:
        if len(picks) >= count:
            break
        add(
            "random",
            record,
            f"seeded random pick (seed {seed})",
        )
    return picks[:count]


def build_results_json(
    *,
    model: str,
    model_version: str | None,
    ultralytics_version: str,
    pytorch_version: str,
    device: str,
    test_image_count: int,
    test_instance_count: int,
    metrics: dict[str, Any],
    per_class_metrics: list[dict[str, Any]],
    confidence_threshold_results: dict[str, Any],
    timing: dict[str, Any],
    class_mapping: dict[str, Any],
    timestamp: str,
) -> dict[str, Any]:
    """Assemble the machine-readable Phase 5 result document."""
    return {
        "model": model,
        "model_version": model_version,
        "ultralytics_version": ultralytics_version,
        "pytorch_version": pytorch_version,
        "device": device,
        "test_image_count": test_image_count,
        "test_instance_count": test_instance_count,
        "metrics": metrics,
        "per_class_metrics": per_class_metrics,
        "confidence_threshold_results": confidence_threshold_results,
        "timing": timing,
        "timestamp": timestamp,
        "class_mapping": class_mapping,
    }

