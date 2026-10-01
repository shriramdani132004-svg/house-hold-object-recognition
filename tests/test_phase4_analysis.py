"""Phase 4 tests: label parsing, statistics helpers, scans, and report output.

Synthetic fixtures are built inside pytest's tmp_path; the real prepared
dataset is never modified.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

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


analyze = _load_script("analyze_dataset")

CLASS_MAP = {0: "chair", 1: "book", 2: "cup"}


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 24), color).save(path, "JPEG")


def _write_label(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


@pytest.fixture()
def synthetic_dataset(tmp_path: Path) -> Path:
    data = tmp_path / "household_objects"
    _write_label(
        data / "labels" / "train" / "a.txt",
        ["0 0.5 0.5 0.2 0.2", "1 0.3 0.3 0.1 0.1"],
    )
    _write_label(data / "labels" / "train" / "b.txt", ["2 0.4 0.4 0.005 0.005"])
    _write_label(data / "labels" / "train" / "orphan.txt", ["0 0.5 0.5 0.4 0.4"])
    _write_label(data / "labels" / "val" / "c.txt", ["0 0.5 0.5 0.05 0.05"])
    _write_label(data / "labels" / "test" / "d.txt", [])
    _write_image(data / "images" / "train" / "a.jpg", (10, 20, 30))
    _write_image(data / "images" / "train" / "b.jpg", (10, 20, 30))
    _write_image(data / "images" / "train" / "lonely.jpg", (200, 20, 30))
    _write_image(data / "images" / "val" / "c.jpg", (30, 40, 50))
    _write_image(data / "images" / "test" / "d.jpg", (60, 70, 80))
    (data / "dataset.yaml").write_text(
        yaml.safe_dump(
            {
                "nc": 3,
                "names": ["chair", "book", "cup"],
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
            }
        ),
        encoding="utf-8",
    )
    return data


def test_expected_counts_match_phase3() -> None:
    assert analyze.EXPECTED_IMAGES == {"train": 40890, "val": 4544, "test": 1965}
    assert analyze.EXPECTED_LABELS == {"train": 40890, "val": 4544, "test": 1965}
    assert analyze.EXPECTED_INSTANCES == {"train": 184709, "val": 21100, "test": 9174}


def test_parse_label_line_valid() -> None:
    status, parsed = analyze.parse_label_line("1 0.25 0.75 0.5 0.25", 3)
    assert status == "ok"
    assert parsed == (1, 0.25, 0.75, 0.5, 0.25)


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("0 0.5 0.5", "fields"),
        ("0 0.5 0.5 0.2", "fields"),
        ("0 0.5 0.5 0.2 0.2 0.2", "fields"),
        ("x 0.5 0.5 0.2 0.2", "numeric"),
        ("3 0.5 0.5 0.2 0.2", "class_id"),
        ("-1 0.5 0.5 0.2 0.2", "class_id"),
        ("0 nan 0.5 0.2 0.2", "nonfinite"),
        ("0 0.5 0.5 0 0.2", "size"),
        ("0 0.5 0.5 -0.1 0.2", "size"),
        ("0 0.5 0.5 1.5 0.2", "range"),
        ("0 0.0 0.5 0.5 0.5", "range"),
        ("0 1.2 0.5 0.5 0.5", "range"),
    ],
)
def test_parse_label_line_errors(line: str, expected: str) -> None:
    status, parsed = analyze.parse_label_line(line, 3)
    assert status == expected
    assert parsed is None


def test_size_band_boundaries() -> None:
    assert analyze.size_band(0.0) == "very small"
    assert analyze.size_band(9.9e-4) == "very small"
    assert analyze.size_band(1e-3) == "small"
    assert analyze.size_band(9.9e-3) == "small"
    assert analyze.size_band(1e-2) == "medium"
    assert analyze.size_band(0.099) == "medium"
    assert analyze.size_band(0.1) == "large"
    assert analyze.size_band(1.0) == "large"
    assert analyze.size_band(5.0) == "large"


def test_band_counts_percentages() -> None:
    bands = analyze.band_counts([0.0005, 0.005, 0.05, 0.5, 0.5])
    assert bands["very small"]["count"] == 1
    assert bands["small"]["count"] == 1
    assert bands["medium"]["count"] == 1
    assert bands["large"]["count"] == 2
    assert sum(entry["count"] for entry in bands.values()) == 5
    assert sum(entry["pct"] for entry in bands.values()) == pytest.approx(100.0)


def test_objects_buckets() -> None:
    buckets = analyze.objects_buckets([0, 1, 2, 3, 5, 6, 56])
    assert buckets == {"0": 1, "1": 1, "2": 1, "3-5": 2, ">5": 2}


def test_multi_object_stats() -> None:
    stats = analyze.multi_object_stats([1, 1, 3, 7, 0])
    assert stats["total_images"] == 5
    assert stats["single_object_images"] == 2
    assert stats["multi_object_images"] == 2
    assert stats["images_3plus"] == 2
    assert stats["max_objects_in_image"] == 7
    assert stats["multi_share_pct"] == 40.0


def test_describe_and_percentile() -> None:
    summary = analyze.describe([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    assert summary["count"] == 10
    assert summary["mean"] == pytest.approx(5.5)
    assert summary["median"] == pytest.approx(5.5)
    assert summary["p25"] == pytest.approx(3.25)
    assert summary["p90"] == pytest.approx(9.1)
    empty = analyze.describe([])
    assert empty["count"] == 0
    assert empty["max"] == 0.0


def test_pct() -> None:
    assert analyze.pct(1, 3) == 33.33
    assert analyze.pct(2, 3) == 66.67
    assert analyze.pct(0, 0) == 0.0


def test_imbalance_stats() -> None:
    stats = analyze.imbalance_stats({0: 100, 1: 25, 2: 4}, CLASS_MAP)
    assert stats["most_class_name"] == "chair"
    assert stats["least_class_name"] == "cup"
    assert stats["max_min_ratio"] == 25.0
    empty = analyze.imbalance_stats({0: 0, 1: 0, 2: 0}, CLASS_MAP)
    assert empty["most_class_id"] == -1
    assert empty["max_min_ratio"] == 0.0


def test_cooccurrence_counts() -> None:
    counts = analyze.cooccurrence_counts([[0, 1], [0, 1], [0, 2], [1, 2]])
    assert counts == {(0, 1): 2, (0, 2): 1, (1, 2): 1}


def test_filename_overlap() -> None:
    overlap = analyze.filename_overlap(
        {"train": {"a", "b"}, "val": {"b", "c"}, "test": set()}
    )
    assert overlap == {"train-val": 1, "train-test": 0, "val-test": 0}


def test_scan_labels_on_synthetic(synthetic_dataset: Path) -> None:
    scan = analyze.scan_labels(synthetic_dataset, CLASS_MAP, show_progress=False)
    assert scan["splits"]["train"] == {"images": 3, "labels": 3, "instances": 4}
    assert scan["splits"]["val"] == {"images": 1, "labels": 1, "instances": 1}
    assert scan["splits"]["test"] == {"images": 1, "labels": 1, "instances": 0}
    assert scan["empty_label_files"] == 1
    assert scan["missing_label_files"] == 1
    assert scan["label_without_image"] == 1
    assert scan["class_instances"] == {0: 3, 1: 1, 2: 1}
    assert scan["class_images"] == {0: 3, 1: 1, 2: 1}
    assert scan["objects_per_image"]["train"] == [2, 1, 1, 0]
    assert scan["objects_per_image"]["val"] == [1]
    assert scan["objects_per_image"]["test"] == [0]
    assert len(scan["areas"]) == 5
    assert scan["lines_fields"] == 0
    assert scan["lines_numeric"] == 0


def test_scan_dimensions_on_synthetic(synthetic_dataset: Path) -> None:
    sizes = analyze.scan_dimensions(synthetic_dataset, show_progress=False)
    assert len(sizes) == 5
    assert sizes[("train", "a")] == (32, 24)


def test_hash_duplicates_on_synthetic(synthetic_dataset: Path) -> None:
    result = analyze.hash_duplicates(synthetic_dataset, show_progress=False)
    assert result["hashed_files"] == 5
    assert result["duplicate_pairs"] == 1
    assert result["cross_split_pairs"] == 0
    assert result["within_split_pairs"] == 1
    assert result["details"][0]["first"]["stem"] == "a"
    assert result["details"][0]["duplicate"]["stem"] == "b"


def test_preflight_rejects_wrong_counts(synthetic_dataset: Path) -> None:
    with pytest.raises(analyze.AnalysisError, match="expected 40890 images"):
        analyze.preflight(synthetic_dataset, CLASS_MAP)


def test_preflight_rejects_missing_dataset_yaml(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(analyze.AnalysisError, match="dataset.yaml not found"):
        analyze.preflight(empty, CLASS_MAP)


def test_preflight_rejects_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(analyze.AnalysisError, match="dataset directory not found"):
        analyze.preflight(tmp_path / "missing", CLASS_MAP)


def test_preflight_rejects_nc_mismatch(synthetic_dataset: Path) -> None:
    yaml_path = synthetic_dataset / "dataset.yaml"
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    data["nc"] = 2
    yaml_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(analyze.AnalysisError, match="nc=2"):
        analyze.preflight(synthetic_dataset, CLASS_MAP)


def test_preflight_rejects_names_mismatch(synthetic_dataset: Path) -> None:
    yaml_path = synthetic_dataset / "dataset.yaml"
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    data["names"] = ["chair", "book", "fork"]
    yaml_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(analyze.AnalysisError, match="do not match"):
        analyze.preflight(synthetic_dataset, CLASS_MAP)


def test_verify_expected_rejects_synthetic_scan(synthetic_dataset: Path) -> None:
    scan = analyze.scan_labels(synthetic_dataset, CLASS_MAP, show_progress=False)
    with pytest.raises(analyze.AnalysisError, match="verified Phase 3"):
        analyze.verify_expected(scan)


def test_build_context_summary_and_report(synthetic_dataset: Path) -> None:
    scan = analyze.scan_labels(synthetic_dataset, CLASS_MAP, show_progress=False)
    context = analyze.build_context(scan, CLASS_MAP)
    assert context["matches_phase3"] is False

    summary = analyze.build_summary(context)
    assert json.loads(json.dumps(summary, sort_keys=True))["totals"]["images"] == 5
    assert summary["matches_phase3_verified_counts"] is False

    report = analyze.render_report(context)
    positions = [report.find(heading) for heading in analyze.REPORT_SECTIONS]
    assert all(position >= 0 for position in positions)
    assert positions == sorted(positions)

    second = analyze.build_context(scan, CLASS_MAP)
    assert analyze.build_summary(second) == summary
    assert analyze.render_report(second) == report


def test_write_samples_on_synthetic(synthetic_dataset: Path, tmp_path: Path) -> None:
    scan = analyze.scan_labels(synthetic_dataset, CLASS_MAP, show_progress=False)
    destination = tmp_path / "samples"
    samples = analyze.write_samples(
        synthetic_dataset,
        CLASS_MAP,
        scan,
        None,
        destination,
        seed=42,
        show_progress=False,
    )
    assert samples
    for entry in samples:
        assert (destination / entry["file"]).is_file()


def test_write_figures_on_synthetic(synthetic_dataset: Path, tmp_path: Path) -> None:
    scan = analyze.scan_labels(synthetic_dataset, CLASS_MAP, show_progress=False)
    dimensions = analyze.scan_dimensions(synthetic_dataset, show_progress=False)
    context = analyze.build_context(scan, CLASS_MAP, dimensions)
    paths = analyze.write_figures(
        context, tmp_path / "figures", show_progress=False
    )
    assert len(paths) == 8
    for path in paths:
        assert path.is_file()
        assert path.stat().st_size > 0


def test_write_figures_skips_dimensions_without_scan(
    synthetic_dataset: Path, tmp_path: Path
) -> None:
    scan = analyze.scan_labels(synthetic_dataset, CLASS_MAP, show_progress=False)
    context = analyze.build_context(scan, CLASS_MAP)
    paths = analyze.write_figures(
        context, tmp_path / "figures", show_progress=False
    )
    assert len(paths) == 7
