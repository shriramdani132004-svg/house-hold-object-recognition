"""Phase 3 tests: configuration, YOLO conversion, splitting, and pipeline.

Synthetic COCO fixtures are built inside pytest's tmp_path; the raw
project dataset is never touched.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = PROJECT_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


prepare_dataset = _load_script("prepare_dataset")
validate_dataset = _load_script("validate_prepared_dataset")

PHASE3_SCRIPTS = [
    "analyze_categories",
    "prepare_dataset",
    "validate_prepared_dataset",
    "visualize_prepared_dataset",
    "dataset_statistics",
]

COCO_CATEGORIES = [
    {"id": 62, "name": "chair"},
    {"id": 44, "name": "bottle"},
    {"id": 47, "name": "cup"},
    {"id": 1, "name": "person"},
]


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 24), color).save(path, "JPEG")


def _build_raw(root: Path) -> Path:
    raw = root / "raw"
    for split in ("train2017", "val2017"):
        (raw / split).mkdir(parents=True)
    (raw / "annotations").mkdir()

    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    annotation_id = 0
    for image_id in range(1, 11):
        file_name = f"{image_id:012d}.jpg"
        _write_image(raw / "train2017" / file_name, (image_id * 20 % 255, 40, 60))
        images.append({"id": image_id, "file_name": file_name, "width": 32, "height": 24})
        if image_id == 9:
            category_id = 1
        elif image_id == 10:
            category_id = 62
        else:
            category_id = 62 if image_id % 2 else 44
        annotations.append(
            {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [4, 4, 12, 8],
                "iscrowd": 1 if image_id == 10 else 0,
                "area": 96,
            }
        )
        annotation_id += 1
    annotations.append(
        {
            "id": annotation_id,
            "image_id": 8,
            "category_id": 44,
            "bbox": [0, 0, 0, 8],
            "iscrowd": 0,
            "area": 0,
        }
    )

    val_images = [{"id": 100, "file_name": "000000000100.jpg", "width": 32, "height": 24}]
    _write_image(raw / "val2017" / "000000000100.jpg", (10, 200, 10))
    val_annotations = [
        {
            "id": 999,
            "image_id": 100,
            "category_id": 47,
            "bbox": [8, 6, 16, 12],
            "iscrowd": 0,
            "area": 192,
        }
    ]

    (raw / "annotations" / "instances_train2017.json").write_text(
        json.dumps({"images": images, "annotations": annotations, "categories": COCO_CATEGORIES}),
        encoding="utf-8",
    )
    (raw / "annotations" / "instances_val2017.json").write_text(
        json.dumps({"images": val_images, "annotations": val_annotations, "categories": COCO_CATEGORIES}),
        encoding="utf-8",
    )
    return raw


def test_dataset_yaml_config_is_valid() -> None:
    config = yaml.safe_load((PROJECT_ROOT / "configs" / "dataset.yaml").read_text(encoding="utf-8"))
    for key in ("source", "output", "classes_config", "selection", "split", "annotations"):
        assert key in config, f"configs/dataset.yaml missing '{key}'"
    assert not Path(config["source"]["raw_dir"]).is_absolute()
    assert not Path(config["output"]["dir"]).is_absolute()
    assert not Path(config["classes_config"]).is_absolute()
    assert isinstance(config["split"]["seed"], int)
    assert 0.0 < float(config["split"]["val_fraction"]) < 1.0
    assert config["split"]["strategy"] == "coco_official_val_as_test"
    assert config["annotations"]["include_iscrowd"] is False
    candidates = config["selection"]["candidate_categories"]
    selected = config["selection"]["selected_categories"]
    assert set(selected) <= set(candidates)


def test_classes_yaml_is_valid_and_matches_config() -> None:
    classes_path = PROJECT_ROOT / "configs" / "classes.yaml"
    class_map = prepare_dataset.load_classes(classes_path)
    assert sorted(class_map) == list(range(len(class_map)))
    assert 15 <= len(class_map) <= 20
    names = [class_map[index] for index in range(len(class_map))]
    assert len(set(names)) == len(names)
    config = yaml.safe_load((PROJECT_ROOT / "configs" / "dataset.yaml").read_text(encoding="utf-8"))
    assert config["selection"]["selected_categories"] == names


def test_load_classes_rejects_bad_mappings(tmp_path: Path) -> None:
    bad_ids = tmp_path / "bad_ids.yaml"
    bad_ids.write_text("classes:\n  1: chair\n  2: cup\n", encoding="utf-8")
    with pytest.raises(prepare_dataset.PrepareError):
        prepare_dataset.load_classes(bad_ids)
    duplicates = tmp_path / "duplicates.yaml"
    duplicates.write_text("classes:\n  0: chair\n  1: chair\n", encoding="utf-8")
    with pytest.raises(prepare_dataset.PrepareError):
        prepare_dataset.load_classes(duplicates)


def test_coco_to_yolo_math() -> None:
    result = prepare_dataset.coco_to_yolo([10, 20, 30, 40], 100, 200)
    assert result == pytest.approx((0.25, 0.2, 0.3, 0.2))


def test_coco_to_yolo_clips_to_bounds() -> None:
    result = prepare_dataset.coco_to_yolo([-10, -10, 30, 30], 100, 100)
    assert result == pytest.approx((0.1, 0.1, 0.2, 0.2))
    overflow = prepare_dataset.coco_to_yolo([90, 90, 40, 40], 100, 100)
    assert overflow == pytest.approx((0.95, 0.95, 0.1, 0.1))


@pytest.mark.parametrize(
    "bbox",
    [
        [10, 10, 0, 10],
        [10, 10, 10, -5],
        [float("nan"), 0, 10, 10],
        [float("inf"), 0, 10, 10],
        "not-a-box",
        None,
        [10, 10, 0.5, 0.5],
    ],
)
def test_coco_to_yolo_rejects_invalid(bbox: Any) -> None:
    min_size = 1.0
    assert prepare_dataset.coco_to_yolo(bbox, 100, 100, min_size) is None


def test_split_is_deterministic_and_disjoint() -> None:
    ids = list(range(200))
    first = prepare_dataset.split_image_ids(ids, seed=42, val_fraction=0.1)
    second = prepare_dataset.split_image_ids(list(reversed(ids)), seed=42, val_fraction=0.1)
    assert first == second
    assert not set(first["train"]) & set(first["val"])
    assert sorted(first["train"] + first["val"]) == sorted(ids)
    other = prepare_dataset.split_image_ids(ids, seed=7, val_fraction=0.1)
    assert other["val"] != first["val"]
    assert len(first["val"]) == 20


def test_prepare_end_to_end(tmp_path: Path) -> None:
    raw = _build_raw(tmp_path)
    output = tmp_path / "out"
    class_map = {0: "chair", 1: "bottle", 2: "cup"}
    metadata = prepare_dataset.prepare(
        raw, output, class_map, seed=42, val_fraction=0.2, min_box_size=1.0
    )

    assert metadata["splits"]["train"]["images"] == 6
    assert metadata["splits"]["val"]["images"] == 2
    assert metadata["splits"]["test"]["images"] == 1
    assert metadata["rejected_annotations"] == {
        "unselected": 1,
        "crowd": 1,
        "missing_image": 0,
        "invalid_box": 1,
    }
    assert metadata["source_images_excluded"] == 2

    stems: dict[str, set[str]] = {}
    for split in ("train", "val", "test"):
        image_stems = {p.stem for p in (output / "images" / split).iterdir()}
        label_stems = {p.stem for p in (output / "labels" / split).iterdir()}
        assert image_stems == label_stems
        stems[split] = image_stems
    assert not stems["train"] & stems["val"]
    assert not stems["train"] & stems["test"]
    assert not stems["val"] & stems["test"]
    assert stems["test"] == {"000000000100"}

    class_ids: set[int] = set()
    for split in ("train", "val", "test"):
        for label in (output / "labels" / split).iterdir():
            boxes, problems = validate_dataset.parse_label(label, len(class_map))
            assert not problems
            assert boxes
            class_ids.update(box[0] for box in boxes)
    assert class_ids <= {0, 1, 2}

    dataset_yaml = yaml.safe_load((output / "dataset.yaml").read_text(encoding="utf-8"))
    assert dataset_yaml["nc"] == 3
    assert dataset_yaml["names"] == ["chair", "bottle", "cup"]
    assert dataset_yaml["path"] == "."

    problems, _ = validate_dataset.validate(output, class_map, verify_images=True)
    assert problems == []


def test_prepare_is_reproducible(tmp_path: Path) -> None:
    raw = _build_raw(tmp_path)
    class_map = {0: "chair", 1: "bottle", 2: "cup"}

    def snapshot(output: Path) -> dict[str, list[str]]:
        prepare_dataset.prepare(raw, output, class_map, seed=42, val_fraction=0.2)
        return {
            split: sorted(p.name for p in (output / "images" / split).iterdir())
            for split in ("train", "val", "test")
        }

    first = snapshot(tmp_path / "out1")
    second = snapshot(tmp_path / "out2")
    assert first == second


def test_prepare_removes_duplicate_content_across_splits(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    for split in ("train2017", "val2017"):
        (raw / split).mkdir(parents=True)
    (raw / "annotations").mkdir()
    payload = b"identical-image-bytes"
    (raw / "train2017" / "000000000001.jpg").write_bytes(payload)
    (raw / "val2017" / "000000000100.jpg").write_bytes(payload)
    train_images = [{"id": 1, "file_name": "000000000001.jpg", "width": 32, "height": 24}]
    train_annotations = [
        {"id": 1, "image_id": 1, "category_id": 62, "bbox": [4, 4, 12, 8], "iscrowd": 0, "area": 96}
    ]
    val_images = [{"id": 100, "file_name": "000000000100.jpg", "width": 32, "height": 24}]
    val_annotations = [
        {"id": 2, "image_id": 100, "category_id": 44, "bbox": [4, 4, 12, 8], "iscrowd": 0, "area": 96}
    ]
    (raw / "annotations" / "instances_train2017.json").write_text(
        json.dumps(
            {"images": train_images, "annotations": train_annotations, "categories": COCO_CATEGORIES}
        ),
        encoding="utf-8",
    )
    (raw / "annotations" / "instances_val2017.json").write_text(
        json.dumps(
            {"images": val_images, "annotations": val_annotations, "categories": COCO_CATEGORIES}
        ),
        encoding="utf-8",
    )
    output = tmp_path / "out"
    metadata = prepare_dataset.prepare(raw, output, {0: "chair", 1: "bottle"})
    remaining = [
        split
        for split in ("train", "val", "test")
        if list((output / "images" / split).iterdir())
    ]
    assert len(remaining) == 1
    assert remaining[0] == "test"
    assert metadata["duplicates_removed"]["by_hash"]["train"] == 1


@pytest.mark.parametrize(
    "line,expected_ok",
    [
        ("0 0.5 0.5 0.25 0.25", True),
        ("3 0.5 0.5 0.25 0.25", False),
        ("0 1.5 0.5 0.25 0.25", False),
        ("0 0.5 0.5 0.0 0.25", False),
        ("0 0.5 0.5 1.5 0.25", False),
        ("0 0.5 0.5", False),
        ("x 0.5 0.5 0.25 0.25", False),
        ("0 0.5 0.5 nan 0.25", False),
    ],
)
def test_parse_label_rejects_bad_lines(tmp_path: Path, line: str, expected_ok: bool) -> None:
    label = tmp_path / "sample.txt"
    label.write_text(line + "\n", encoding="utf-8")
    boxes, problems = validate_dataset.parse_label(label, 3)
    assert bool(boxes) is expected_ok
    assert bool(problems) is not expected_ok


def test_validate_detects_missing_label(tmp_path: Path) -> None:
    raw = _build_raw(tmp_path)
    output = tmp_path / "out"
    class_map = {0: "chair", 1: "bottle", 2: "cup"}
    prepare_dataset.prepare(raw, output, class_map, seed=42, val_fraction=0.2)
    victim = sorted((output / "labels" / "train").iterdir())[0]
    victim.unlink()
    problems, _ = validate_dataset.validate(output, class_map, verify_images=True)
    assert any("without label" in problem for problem in problems)


def test_validate_detects_orphan_label_and_bad_class(tmp_path: Path) -> None:
    raw = _build_raw(tmp_path)
    output = tmp_path / "out"
    class_map = {0: "chair", 1: "bottle", 2: "cup"}
    prepare_dataset.prepare(raw, output, class_map, seed=42, val_fraction=0.2)
    orphan = output / "labels" / "train" / "000000000999.txt"
    orphan.write_text("7 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    problems, _ = validate_dataset.validate(output, class_map, verify_images=True)
    assert any("label without image" in problem for problem in problems)
    assert any("outside" in problem for problem in problems)


def test_phase3_reports_exist() -> None:
    selection = PROJECT_ROOT / "reports" / "phase3_class_selection.md"
    analysis = PROJECT_ROOT / "reports" / "phase3_category_analysis.md"
    assert selection.is_file()
    assert analysis.is_file()
    text = selection.read_text(encoding="utf-8")
    class_map = prepare_dataset.load_classes(PROJECT_ROOT / "configs" / "classes.yaml")
    for name in class_map.values():
        assert name in text, f"class missing from selection report: {name}"
    assert "Excluded candidates" in text
    assert "Final classes" in text


@pytest.mark.parametrize("script_name", PHASE3_SCRIPTS)
def test_scripts_contain_no_absolute_paths(script_name: str) -> None:
    source = (PROJECT_ROOT / "scripts" / f"{script_name}.py").read_text(encoding="utf-8")
    assert "C:\\" not in source
    assert "C:/" not in source
    assert "/home/" not in source
