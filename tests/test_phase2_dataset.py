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


def _write_zip(path: Path, members: dict[str, bytes]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        for name, payload in members.items():
            zf.writestr(name, payload)
    return path.stat().st_size


def test_download_skips_already_verified_file(tmp_path: Path) -> None:
    destination = tmp_path / "zips" / "tiny.zip"
    size = _write_zip(destination, {"folder/file.txt": b"hello"})
    archive = download_dataset.Archive(
        name="tiny.zip",
        relative_path="zips/tiny.zip",
        url_path="/zips/tiny.zip",
        size_bytes=size,
    )
    result = download_dataset.download_archive(archive, tmp_path)
    assert result == destination
    assert destination.stat().st_size == size


def test_download_redownloads_corrupt_complete_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "zips" / "broken.zip"
    valid_payload = tmp_path / "source.zip"
    size = _write_zip(valid_payload, {"folder/file.txt": b"hello world"})
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"x" * size)

    archive = download_dataset.Archive(
        name="broken.zip",
        relative_path="zips/broken.zip",
        url_path="/zips/broken.zip",
        size_bytes=size,
    )
    valid_bytes = valid_payload.read_bytes()

    def fake_stream(url, dest, offset, expected, **kwargs):
        dest.write_bytes(valid_bytes)

    monkeypatch.setattr(download_dataset, "_stream_to_file", fake_stream)
    result = download_dataset.download_archive(
        archive, tmp_path, max_attempts=2, sleep=lambda _seconds: None
    )
    assert result == destination
    assert destination.read_bytes() == valid_bytes


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
        raise download_dataset.ConnectionInterrupted("simulated network failure")

    monkeypatch.setattr(download_dataset, "_stream_to_file", broken_stream)
    with pytest.raises(download_dataset.DatasetError, match="simulated network failure"):
        download_dataset.download_archive(
            archive, tmp_path, max_attempts=2, sleep=lambda _seconds: None
        )


def test_connection_interrupted_is_retryable_dataset_error() -> None:
    assert issubclass(
        download_dataset.ConnectionInterrupted, download_dataset.DatasetError
    )


def test_train_archive_constants_match_acquisition_spec() -> None:
    train = download_dataset.TRAIN2017
    assert train.size_bytes == 19_336_861_798
    assert train.image_count == 118_287
    assert train.image_prefix == "train2017/"
    assert train.url_path == "/zips/train2017.zip"


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
                    "images": [{"id": 3}, {"id": 4}],
                    "annotations": [{"id": 3}, {"id": 4}],
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


def test_default_archives_detects_train_archive(tmp_path: Path) -> None:
    archives = verify_dataset.default_archives(tmp_path)
    assert download_dataset.TRAIN2017 not in archives
    train_zip = tmp_path / download_dataset.TRAIN2017.relative_path
    train_zip.parent.mkdir(parents=True)
    train_zip.write_bytes(b"placeholder")
    archives = verify_dataset.default_archives(tmp_path)
    assert download_dataset.TRAIN2017 in archives


def test_verify_includes_train_images_when_present(tmp_path: Path) -> None:
    ann_archive, val_archive = _build_synthetic_dataset(tmp_path)
    train_zip = tmp_path / "zips" / "train.zip"
    _write_zip(
        train_zip,
        {
            "train2017/000000000003.jpg": b"\xff\xd8\xff",
            "train2017/000000000004.jpg": b"\xff\xd8\xff",
        },
    )
    train_archive = download_dataset.Archive(
        name="train.zip",
        relative_path="zips/train.zip",
        url_path="/zips/train.zip",
        size_bytes=train_zip.stat().st_size,
        image_prefix="train2017/",
        image_count=2,
    )
    download_dataset.extract_archive(train_archive, tmp_path)

    problems, stats = verify_dataset.verify(
        tmp_path, archives=(ann_archive, val_archive, train_archive)
    )
    assert problems == []
    assert stats["image_files:train2017"] == 2


def test_verify_reports_train_image_count_mismatch(tmp_path: Path) -> None:
    ann_archive, val_archive = _build_synthetic_dataset(tmp_path)
    train_zip = tmp_path / "zips" / "train.zip"
    _write_zip(train_zip, {"train2017/000000000003.jpg": b"\xff\xd8\xff"})
    train_archive = download_dataset.Archive(
        name="train.zip",
        relative_path="zips/train.zip",
        url_path="/zips/train.zip",
        size_bytes=train_zip.stat().st_size,
        image_prefix="train2017/",
        image_count=2,
    )
    train_dir = tmp_path / "train2017"
    train_dir.mkdir(exist_ok=True)
    (train_dir / "000000000003.jpg").write_bytes(b"\xff\xd8\xff")

    problems, _stats = verify_dataset.verify(
        tmp_path, archives=(ann_archive, val_archive, train_archive)
    )
    assert any("train2017" in problem for problem in problems)


def test_verify_main_exits_nonzero_for_missing_dir(tmp_path: Path) -> None:
    exit_code = verify_dataset.main(["--dest", str(tmp_path / "does-not-exist")])
    assert exit_code == 1
