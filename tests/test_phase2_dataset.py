"""Phase 2 tests: dataset documentation and acquisition tooling.

These tests never touch the network; they operate on documentation files
and synthetic fixtures inside pytest's tmp_path.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = PROJECT_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


download_dataset = _load_script("download_dataset")
verify_dataset = _load_script("verify_dataset")


def test_dataset_md_contains_required_sections() -> None:
    text = (PROJECT_ROOT / "DATASET.md").read_text(encoding="utf-8")
    required = [
        "# Dataset",
        "## Selected Dataset",
        "## Official Source",
        "## Dataset Version",
        "## License",
        "## Why This Dataset Was Selected",
        "## Dataset Size",
        "## Relevant Classes",
        "## Annotation Format",
        "## Acquisition Method",
        "## Citation / Attribution",
        "## Limitations",
    ]
    missing = [heading for heading in required if heading not in text]
    assert not missing, f"DATASET.md missing sections: {missing}"


def test_readme_has_dataset_section_linking_docs() -> None:
    text = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    assert "## Dataset" in text
    assert "DATASET.md" in text


def test_dataset_info_exists_with_verified_fields() -> None:
    info = PROJECT_ROOT / "data" / "raw" / "DATASET_INFO.txt"
    assert info.is_file(), "data/raw/DATASET_INFO.txt must exist after Phase 2"
    text = info.read_text(encoding="utf-8")
    for field in (
        "Dataset name",
        "Source URL",
        "Acquisition date",
        "License",
        "Annotation format",
    ):
        assert field in text, f"DATASET_INFO.txt missing field: {field}"


@pytest.mark.parametrize("script_name", ["download_dataset", "verify_dataset"])
def test_scripts_contain_no_absolute_paths(script_name: str) -> None:
    source = (PROJECT_ROOT / "scripts" / f"{script_name}.py").read_text(
        encoding="utf-8"
    )
    assert "C:\\" not in source
    assert "/home/" not in source
    assert "C:/" not in source


def test_md5_and_size_helpers(tmp_path: Path) -> None:
    target = tmp_path / "sample.bin"
    target.write_bytes(b"abc")
    assert download_dataset.md5_of(target) == "900150983cd24fb0d6963f7d28e17f72"
    download_dataset.check_size(target, 3)
    with pytest.raises(download_dataset.DatasetError):
        download_dataset.check_size(target, 4)


def test_check_zip_valid_and_missing_member(tmp_path: Path) -> None:
    archive_path = tmp_path / "good.zip"
    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.writestr("annotations/instances_val2017.json", "{}")
        zf.writestr("annotations/instances_train2017.json", "{}")
    download_dataset.check_zip(
        archive_path,
        required_members=(
            "annotations/instances_val2017.json",
            "annotations/instances_train2017.json",
        ),
    )
    with pytest.raises(download_dataset.DatasetError):
        download_dataset.check_zip(
            archive_path, required_members=("annotations/missing.json",)
        )


def test_check_zip_image_count(tmp_path: Path) -> None:
    archive_path = tmp_path / "images.zip"
    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.writestr("val2017/000000000001.jpg", b"\xff\xd8\xff")
        zf.writestr("val2017/000000000002.jpg", b"\xff\xd8\xff")
    download_dataset.check_zip(
        archive_path, image_prefix="val2017/", image_count=2
    )
    with pytest.raises(download_dataset.DatasetError):
        download_dataset.check_zip(
            archive_path, image_prefix="val2017/", image_count=3
        )


def test_download_skips_already_complete_file(tmp_path: Path) -> None:
    archive = download_dataset.Archive(
        name="tiny.zip",
        relative_path="zips/tiny.zip",
        url_path="/zips/tiny.zip",
        size_bytes=5,
    )
    destination = tmp_path / archive.relative_path
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"12345")
    result = download_dataset.download_archive(archive, tmp_path)
    assert result == destination
    assert destination.read_bytes() == b"12345"


def test_download_fails_clearly_after_network_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = download_dataset.Archive(
        name="missing.zip",
        relative_path="zips/missing.zip",
        url_path="/zips/missing.zip",
        size_bytes=5,
    )

    def broken_stream(*args, **kwargs):
        raise download_dataset.DatasetError("simulated network failure")

    monkeypatch.setattr(download_dataset, "_stream_to_file", broken_stream)
    with pytest.raises(download_dataset.DatasetError, match="simulated network failure"):
        download_dataset.download_archive(
            archive, tmp_path, max_attempts=2, sleep=lambda _seconds: None
        )


def test_extract_produces_expected_layout(tmp_path: Path) -> None:
    archive = download_dataset.Archive(
        name="ann.zip",
        relative_path="zips/ann.zip",
        url_path="/annotations/ann.zip",
        size_bytes=0,
        required_members=(
            "annotations/instances_val2017.json",
            "annotations/instances_train2017.json",
        ),
    )
    source = tmp_path / archive.relative_path
    source.parent.mkdir(parents=True)
    with zipfile.ZipFile(source, "w") as zf:
        zf.writestr("annotations/instances_val2017.json", "{}")
        zf.writestr("annotations/instances_train2017.json", "{}")
    assert not download_dataset.is_extracted(archive, tmp_path)
    download_dataset.extract_archive(archive, tmp_path)
    assert download_dataset.is_extracted(archive, tmp_path)
    download_dataset.extract_archive(archive, tmp_path)


def _build_synthetic_dataset(root: Path) -> tuple[object, object]:
    ann_dir = root / "zips"
    ann_dir.mkdir(parents=True)

    ann_zip = ann_dir / "annotations.zip"
    with zipfile.ZipFile(ann_zip, "w") as zf:
        zf.writestr(
            "annotations/instances_val2017.json",
            json.dumps(
                {
                    "images": [{"id": 1}, {"id": 2}],
                    "annotations": [{"id": 1}, {"id": 2}],
                    "categories": [{"id": 1, "name": "cup"}],
                }
            ),
        )
        zf.writestr(
            "annotations/instances_train2017.json",
            json.dumps(
                {
                    "images": [{"id": 3}],
                    "annotations": [{"id": 3}],
                    "categories": [{"id": 1, "name": "cup"}],
                }
            ),
        )
    ann_archive = download_dataset.Archive(
        name="annotations.zip",
        relative_path="zips/annotations.zip",
        url_path="/annotations/annotations.zip",
        size_bytes=ann_zip.stat().st_size,
        required_members=(
            "annotations/instances_val2017.json",
            "annotations/instances_train2017.json",
        ),
    )

    val_zip = ann_dir / "val.zip"
    with zipfile.ZipFile(val_zip, "w") as zf:
        zf.writestr("val2017/000000000001.jpg", b"\xff\xd8\xff")
        zf.writestr("val2017/000000000002.jpg", b"\xff\xd8\xff")
    val_archive = download_dataset.Archive(
        name="val.zip",
        relative_path="zips/val.zip",
        url_path="/zips/val.zip",
        size_bytes=val_zip.stat().st_size,
        image_prefix="val2017/",
        image_count=2,
    )

    download_dataset.extract_archive(ann_archive, root)
    download_dataset.extract_archive(val_archive, root)
    return ann_archive, val_archive


def test_verify_passes_on_synthetic_dataset(tmp_path: Path) -> None:
    ann_archive, val_archive = _build_synthetic_dataset(tmp_path)
    problems, stats = verify_dataset.verify(
        tmp_path, archives=(ann_archive, val_archive)
    )
    assert problems == []
    assert stats["images_val"] == 2
    assert stats["instances_val"] == 2
    assert stats["categories"] == 1
    assert stats["category_names"] == ["cup"]


def test_verify_reports_missing_components(tmp_path: Path) -> None:
    problems, _stats = verify_dataset.verify(tmp_path)
    assert problems


def test_verify_main_exits_nonzero_for_missing_dir(tmp_path: Path) -> None:
    exit_code = verify_dataset.main(["--dest", str(tmp_path / "does-not-exist")])
    assert exit_code == 1
