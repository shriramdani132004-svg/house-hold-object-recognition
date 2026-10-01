"""Phase 5 tests: baseline metrics, mapping, evaluation helpers, artifacts.

Synthetic fixtures use pytest's tmp_path; the prepared dataset and the
baseline report artifacts are only read, never modified.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = PROJECT_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


from src.evaluation import baseline as metric

evaluate = _load_script("evaluate_baseline")
inference = _load_script("baseline_inference")

CLASS_MAP = {0: "chair", 1: "book", 2: "cup"}


def test_config_loads_with_expected_section() -> None:
    config = metric.load_config("configs/baseline.yaml")
    assert config["model"]["path"] == "models/baseline/yolo26n.pt"
    assert config["expected"] == {
        "test_image_count": 1965,
        "test_instance_count": 9174,
        "class_count": 19,
    }
    assert config["evaluation"]["primary_conf"] == 0.25
    assert config["inference"]["conf_floor"] == 0.01
    resolved = metric.resolve_path(config["data"]["test_images"])
    assert resolved.is_dir()


def test_project_classes_match_dataset_config() -> None:
    classes = metric.load_project_classes("configs/classes.yaml")
    assert len(classes) == 19
    assert classes[0] == "chair"
    with (PROJECT_ROOT / "configs" / "dataset.yaml").open(
        encoding="utf-8"
    ) as handle:
        dataset = yaml.safe_load(handle)
    assert sorted(classes.values()) == sorted(dataset["selection"]["selected_categories"])


def test_build_class_mapping_by_name() -> None:
    coco_names = {0: "person", 56: "chair", 62: "cup", 67: "cup"}
    mapping = metric.build_class_mapping(coco_names, CLASS_MAP)
    assert mapping == {56: 0, 62: 2, 67: 2}


def test_filter_detections_drops_non_project_classes() -> None:
    mapping = {56: 0}
    detections = [
        {"coco_class": 56, "score": 0.9, "bbox": (0.1, 0.1, 0.5, 0.5)},
        {"coco_class": 0, "score": 0.8, "bbox": (0.2, 0.2, 0.4, 0.4)},
    ]
    kept = metric.filter_detections(detections, mapping)
    assert len(kept) == 1
    assert kept[0]["class_id"] == 0
    assert kept[0]["score"] == 0.9


def test_box_iou_basics() -> None:
    box = (0.1, 0.1, 0.5, 0.5)
    assert metric.box_iou(box, box) == pytest.approx(1.0)
    assert metric.box_iou(box, (0.9, 0.9, 0.95, 0.95)) == 0.0
    partial = metric.box_iou(box, (0.3, 0.3, 0.7, 0.7))
    assert 0.0 < partial < 1.0


def test_match_image_counts_and_greedy_order() -> None:
    target = {"class_id": 0, "bbox": (0.1, 0.1, 0.5, 0.5)}
    high = {"class_id": 0, "bbox": (0.1, 0.1, 0.5, 0.5), "score": 0.9}
    low = {"class_id": 0, "bbox": (0.11, 0.11, 0.51, 0.51), "score": 0.4}
    result = metric.match_image([low, high], [target])
    assert result["tp"] == 1 and result["fp"] == 1 and result["fn"] == 0
    wrong_class = {"class_id": 1, "bbox": (0.1, 0.1, 0.5, 0.5), "score": 0.8}
    result2 = metric.match_image([wrong_class], [target])
    assert result2["tp"] == 0 and result2["fp"] == 1 and result2["fn"] == 1
    assert 1 in result2["per_class"] and 0 in result2["per_class"]


def test_aggregate_and_precision_recall_f1() -> None:
    per_image = [metric.match_image([], [{"class_id": 0, "bbox": (0, 0, 1, 1)}])]
    aggregate = metric.aggregate_matches(per_image)
    assert aggregate["counts"] == {"tp": 0, "fp": 0, "fn": 1}
    assert aggregate["metrics"]["recall"] == 0.0
    prf = metric.precision_recall_f1(2, 1, 1)
    assert prf["precision"] == pytest.approx(2 / 3)
    assert prf["recall"] == pytest.approx(2 / 3)
    assert prf["f1"] == pytest.approx(2 / 3)


def test_average_precision_perfect_prediction() -> None:
    detections = [
        {"image_id": "a", "class_id": 0, "bbox": (0.1, 0.1, 0.5, 0.5), "score": 0.9}
    ]
    targets = [{"image_id": "a", "class_id": 0, "bbox": (0.1, 0.1, 0.5, 0.5)}]
    aps = metric.average_precision(detections, targets)
    assert all(value == pytest.approx(1.0) for value in aps[0])
    maps = metric.mean_average_precision(aps)
    assert maps["map50"] == pytest.approx(1.0)
    assert maps["map50_95"] == pytest.approx(1.0)


def test_average_precision_no_ground_truth_is_none() -> None:
    aps = metric.average_precision(
        [{"image_id": "a", "class_id": 4, "bbox": (0, 0, 0.1, 0.1), "score": 0.8}],
        [],
    )
    assert aps[4] == [None] * len(metric.MAP_IOU_THRESHOLDS)
    maps = metric.mean_average_precision(aps)
    assert maps["map50"] == 0.0


def test_select_samples_is_deterministic_and_unique() -> None:
    records = [
        {
            "image_id": f"img{index:04d}",
            "gt_boxes": [
                {
                    "class_id": (index + offset) % 3,
                    "class_name": CLASS_MAP[(index + offset) % 3],
                    "area": 0.01 * (index + offset + 1),
                }
                for offset in range(1 + index % 3)
            ],
        }
        for index in range(10)
    ]
    picks = metric.select_samples(records, CLASS_MAP, count=8, seed=42)
    assert 0 < len(picks) <= 8
    image_ids = [pick["image_id"] for pick in picks]
    assert len(image_ids) == len(set(image_ids))
    assert picks == metric.select_samples(records, CLASS_MAP, count=8, seed=42)


def test_build_results_json_is_serializable() -> None:
    document = metric.build_results_json(
        model="models/baseline/yolo26n.pt",
        model_version=None,
        ultralytics_version="8.4.171",
        pytorch_version="2.14.1+cpu",
        device="cpu",
        test_image_count=1965,
        test_instance_count=9174,
        metrics={"precision": 0.5},
        per_class_metrics=[],
        confidence_threshold_results={},
        timing={"avg_wall_ms_per_image": 180.0},
        class_mapping={"mechanism": "name_based_coco_to_project"},
        timestamp="2026-10-01T00:00:00+00:00",
    )
    payload = json.dumps(document)
    assert json.loads(payload)["device"] == "cpu"


def test_evaluate_load_targets_and_counts(tmp_path: Path) -> None:
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n1 0.3 0.3 0.1 0.1\n")
    (labels / "b.txt").write_text("2 0.4 0.4 0.005 0.005")
    targets, total = evaluate.load_targets(labels, 3)
    assert total == 3
    assert set(targets) == {"a", "b"}
    x1, y1, x2, y2 = targets["a"][0]["bbox"]
    assert (x1, y1, x2, y2) == pytest.approx((0.4, 0.4, 0.6, 0.6))
    assert targets["a"][0]["area"] == pytest.approx(0.04)


def test_evaluate_load_targets_rejects_bad_class(tmp_path: Path) -> None:
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "a.txt").write_text("9 0.5 0.5 0.2 0.2\n")
    with pytest.raises(ValueError, match="class id"):
        evaluate.load_targets(labels, 3)


def test_evaluate_count_images(tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    for name in ("a.jpg", "b.png", "c.txt"):
        (images / name).write_bytes(b"x")
    assert evaluate.count_images(images) == 2


def test_evaluate_cache_roundtrip(tmp_path: Path) -> None:
    meta = evaluate.cache_meta(
        Path("yolo26n.pt"), 0.01, 640, {56: 0}, expected_images=1
    )
    records = [
        {"image_id": "a", "wall_ms": 100.0, "infer_ms": 90.0, "raw_count": 2,
         "dets": [{"class_id": 0, "score": 0.7, "bbox": [0.1, 0.1, 0.5, 0.5]}]}
    ]
    cache = tmp_path / "detections.json"
    evaluate.save_cache(cache, meta, records)
    loaded = evaluate.load_cache(cache, meta)
    assert loaded == records
    stale = dict(meta, imgsz=512)
    assert evaluate.load_cache(cache, stale) is None
    assert evaluate.load_cache(tmp_path / "missing.json", meta) is None


def test_evaluate_confidence_table() -> None:
    records = [
        {
            "image_id": "a",
            "dets": [
                {"class_id": 0, "score": 0.9, "bbox": (0.1, 0.1, 0.5, 0.5)},
                {"class_id": 0, "score": 0.4, "bbox": (0.6, 0.6, 0.9, 0.9)},
            ],
        }
    ]
    targets = {
        "a": [
            {"class_id": 0, "bbox": (0.1, 0.1, 0.5, 0.5)},
            {"class_id": 0, "bbox": (0.6, 0.6, 0.9, 0.9)},
        ]
    }
    rows = evaluate.confidence_table(records, targets, thresholds=(0.25, 0.75))
    assert [row["threshold"] for row in rows] == [0.25, 0.75]
    assert rows[0]["tp"] == 2 and rows[0]["fn"] == 0 and rows[0]["fp"] == 0
    assert rows[1]["tp"] == 1 and rows[1]["fn"] == 1 and rows[1]["fp"] == 0
    assert rows[0]["precision"] == pytest.approx(1.0)


def test_evaluate_analyze_failures_categories() -> None:
    targets = {
        "img1": [
            {"class_id": 0, "bbox": (0.1, 0.1, 0.3, 0.3), "area": 0.04},
            {"class_id": 1, "bbox": (0.6, 0.6, 0.8, 0.8), "area": 0.04},
        ]
    }
    records = [
        {
            "image_id": "img1",
            "dets": [
                {"class_id": 2, "score": 0.9, "bbox": (0.1, 0.1, 0.3, 0.3)},
                {"class_id": 0, "score": 0.5, "bbox": (0.85, 0.85, 0.95, 0.95)},
            ],
        }
    ]
    failures = evaluate.analyze_failures(records, targets, CLASS_MAP, 0.25)
    assert failures["categories"].get("wrong_class") == 1
    assert failures["total_fn"] == 2
    assert failures["confusion"][0] == {
        "ground_truth": "chair",
        "predicted": "cup",
        "count": 1,
    }
    assert failures["examples"]["wrong_class"] == ["img1"]


def test_evaluate_speed_summary_averages() -> None:
    records = [
        {"wall_ms": 100.0, "infer_ms": 90.0},
        {"wall_ms": 200.0, "infer_ms": 180.0},
        {"wall_ms": 300.0, "infer_ms": 270.0},
    ]
    speed = evaluate.speed_summary(records, total_ms=600.0, model_load_ms=50.0)
    assert speed["avg_wall_ms_per_image"] == pytest.approx(200.0)
    assert speed["avg_model_ms_per_image"] == pytest.approx(180.0)
    assert speed["images_per_second"] == pytest.approx(5.0)
    assert speed["warmup_excluded_images"] == 0
    assert speed["model_load_ms"] == pytest.approx(50.0)


def test_inference_helpers() -> None:
    config = inference.load_config()
    assert config["model"]["path"].endswith("yolo26n.pt")
    existing = inference.ensure_model(
        PROJECT_ROOT / "models" / "baseline" / "yolo26n.pt"
    )
    assert existing.is_file()
    line = inference.format_detection("chair", 0.875, (1, 2, 3, 4))
    assert "chair" in line and "0.875" in line and "[1, 2, 3, 4]" in line


def test_inference_rejects_missing_source(tmp_path: Path) -> None:
    code = inference.main(["--source", str(tmp_path / "nope.jpg")])
    assert code == 1


@pytest.mark.skipif(
    not (PROJECT_ROOT / "models" / "baseline" / "yolo26n.pt").is_file(),
    reason="baseline weights not downloaded yet",
)
def test_model_covers_all_project_classes() -> None:
    from ultralytics import YOLO

    model = YOLO(str(PROJECT_ROOT / "models" / "baseline" / "yolo26n.pt"))
    classes = metric.load_project_classes("configs/classes.yaml")
    mapping = metric.build_class_mapping(dict(model.names), classes)
    assert len(mapping) == len(classes)


def test_baseline_report_sections() -> None:
    report = (PROJECT_ROOT / "reports" / "baseline_report.md").read_text(
        encoding="utf-8"
    )
    for number, title in enumerate(
        [
            "Overview",
            "Model",
            "Dataset and Class Mapping",
            "Evaluation Protocol",
            "Overall Metrics",
            "Per-Class Metrics",
            "Confidence Threshold Analysis",
            "Inference Speed",
            "Error Analysis Summary",
            "Sample Review",
            "Limitations",
            "Reproduction and Next Steps",
        ],
        start=1,
    ):
        assert f"## {number}. {title}" in report


def test_baseline_results_json_structure() -> None:
    with (PROJECT_ROOT / "reports" / "baseline_results.json").open(
        encoding="utf-8"
    ) as handle:
        document = json.load(handle)
    assert document["test_image_count"] == 1965
    assert document["test_instance_count"] == 9174
    assert len(document["per_class_metrics"]) == 19
    assert set(document["confidence_threshold_results"]) == {"0.25", "0.50", "0.75"}
    assert document["class_mapping"]["mapped_classes"] == 19
    assert document["device"] == "cpu"
    for key in ("precision", "recall", "f1", "map50", "map50_95"):
        assert key in document["metrics"]


def test_baseline_supporting_documents_exist() -> None:
    reports = PROJECT_ROOT / "reports"
    for name in (
        "baseline_confidence_analysis.md",
        "baseline_error_examples.md",
        "baseline_commands.md",
    ):
        assert (reports / name).is_file()
    samples_dir = reports / "baseline_samples"
    index = samples_dir / "index.md"
    assert index.is_file()
    listed = [
        line.split("|")[2].strip()
        for line in index.read_text(encoding="utf-8").splitlines()
        if line.startswith("| ") and "File" not in line and "---" not in line
    ]
    rendered = [path.name for path in samples_dir.glob("*.jpg")]
    assert rendered
    assert sorted(listed) == sorted(rendered)
