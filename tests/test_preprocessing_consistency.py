"""Cross-pipeline preprocessing consistency tests (shared source of truth).

Proves that training (``ContinualImageDataset``), evaluation
(``EvalCache``), tensor caching, and inference
(``preprocess_image``) all produce byte-identical tensors from the same
image, and that legacy on-disk evaluation caches stay valid. Everything
runs on tiny synthetic images in ``tmp_path`` — no CORe50 data, no
training, no network.
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import replace
from pathlib import Path
from typing import Callable

import numpy as np
import pytest
import torch
from PIL import Image

from src.data import preprocessing as base
from src.data.continual import SampleRecord
from src.evaluation.continual import EvalCache
from src.inference import preprocessing as inference_preprocessing
from src.inference.preprocessing import IMAGE_SIZE, preprocess_image
from src.training.dataset import MEAN, STD, ContinualImageDataset
from src.training.tensor_cache import TensorCache, _decode_one


def _gradient_rgb(width: int, height: int) -> Image.Image:
    cols = np.arange(width, dtype=np.uint32)
    rows = np.arange(height, dtype=np.uint32)
    array = np.zeros((height, width, 3), dtype=np.uint8)
    array[..., 0] = (cols % 256)[None, :].astype(np.uint8)
    array[..., 1] = (rows % 256)[:, None].astype(np.uint8)
    array[..., 2] = ((cols[None, :] + rows[:, None]) % 256).astype(np.uint8)
    return Image.fromarray(array, "RGB")


def _random_rgb(width: int, height: int) -> Image.Image:
    rng = np.random.default_rng(0)
    array = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    return Image.fromarray(array, "RGB")


def _random_rgba(width: int, height: int) -> Image.Image:
    rng = np.random.default_rng(1)
    array = rng.integers(0, 256, size=(height, width, 4), dtype=np.uint8)
    return Image.fromarray(array, "RGBA")


def _random_gray(width: int, height: int) -> Image.Image:
    rng = np.random.default_rng(2)
    array = rng.integers(0, 256, size=(height, width), dtype=np.uint8)
    return Image.fromarray(array, "L")


CASES: tuple[tuple[str, Callable[[], Image.Image]], ...] = (
    ("gradient_64", lambda: _gradient_rgb(64, 64)),
    ("random_128", lambda: _random_rgb(128, 128)),
    ("solid_50x37", lambda: Image.new("RGB", (50, 37), (12, 200, 90))),
    ("rgba_80x60", lambda: _random_rgba(80, 60)),
    ("grayscale_41x73", lambda: _random_gray(41, 73)),
)


def _write_case(root: Path, name: str, image: Image.Image) -> tuple[Path, str]:
    relative = f"s3/o1/{name}.png"
    path = root.joinpath(*relative.split("/"))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path, relative


def _record(relative: str, label: int) -> SampleRecord:
    return SampleRecord(
        relative_path=relative,
        label=label,
        split="test",
        experience_id=0,
        source_filelist="synthetic/test_filelist.txt",
        line_number=1,
        object_id=1,
        session_id=3,
        category_id=0,
        category_name="synthetic",
        object_name="synthetic_object_1",
    )


def _reference_decode(path: Path, image_size: int) -> torch.Tensor:
    with Image.open(path) as handle:
        image = handle.convert("RGB")
    if image.size != (image_size, image_size):
        image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
    tensor = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
    return tensor.reshape(image_size, image_size, 3).permute(2, 0, 1)


def _open_copy(path: Path) -> Image.Image:
    with Image.open(path) as handle:
        return handle.copy()


@pytest.mark.parametrize(("name", "factory"), CASES, ids=[c[0] for c in CASES])
def test_dataset_and_inference_preprocess_identically(
    tmp_path: Path, name: str, factory: Callable[[], Image.Image]
) -> None:
    path, relative = _write_case(tmp_path, name, factory())
    dataset = ContinualImageDataset([_record(relative, 3)], tmp_path, IMAGE_SIZE)
    dataset_tensor, label = dataset[0]
    pil = _open_copy(path)
    inference_tensor = preprocess_image(pil)
    assert dataset_tensor.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
    assert dataset_tensor.dtype == torch.float32
    assert label == 3
    assert inference_tensor.shape == (1, 3, IMAGE_SIZE, IMAGE_SIZE)
    assert torch.equal(dataset_tensor, inference_tensor[0])
    assert torch.equal(dataset_tensor, base.preprocess_pil(pil))


def test_shared_decode_is_bit_exact_with_pre_refactor_transform(
    tmp_path: Path,
) -> None:
    for name, factory in CASES:
        path, _ = _write_case(tmp_path, name, factory())
        expected = _reference_decode(path, IMAGE_SIZE)
        assert torch.equal(base.decode_image_file(path, IMAGE_SIZE), expected), name
        cached = torch.from_numpy(_decode_one(path, IMAGE_SIZE))
        assert torch.equal(cached, expected), name


def test_eval_cache_stores_raw_uint8_and_matches_shared_preprocess(
    tmp_path: Path,
) -> None:
    records: list[SampleRecord] = []
    expected: list[torch.Tensor] = []
    written: list[Path] = []
    for index, (name, factory) in enumerate(CASES[:3]):
        path, relative = _write_case(tmp_path, name, factory())
        records.append(_record(relative, index))
        written.append(path)
        expected.append(base.preprocess_pil(_open_copy(path)))
    progress: list[tuple[int, int]] = []
    cache = EvalCache.from_records(
        records, tmp_path, IMAGE_SIZE, on_progress=lambda cur, tot: progress.append((cur, tot))
    )
    assert len(cache) == len(records)
    assert cache.image_size == IMAGE_SIZE
    assert cache.paths == tuple(record.relative_path for record in records)
    assert cache.images.dtype == torch.uint8
    assert torch.equal(cache.labels, torch.arange(len(records)))
    assert progress and progress[-1] == (len(records), len(records))
    for index, path in enumerate(written):
        assert torch.equal(cache.images[index], base.decode_image_file(path, IMAGE_SIZE))
        batch = cache.float_batch(index, index + 1)
        assert batch.dtype == torch.float32
        assert torch.equal(batch[0], expected[index])


def test_eval_cache_from_records_still_requires_test_split(tmp_path: Path) -> None:
    _, relative = _write_case(tmp_path, "split_case", Image.new("RGB", (16, 16), (1, 2, 3)))
    train_record = replace(_record(relative, 0), split="train")
    with pytest.raises(ValueError, match="split"):
        EvalCache.from_records([train_record], tmp_path, IMAGE_SIZE)


def test_normalize_roundtrip_ranges_and_layout() -> None:
    rng = np.random.default_rng(0)
    raw = torch.from_numpy(rng.integers(0, 256, size=(3, 24, 24), dtype=np.uint8))
    normalized = base.normalize_chw(raw)
    assert normalized.dtype == torch.float32
    assert normalized.shape == (3, 24, 24)
    assert float(normalized.min()) >= -1.0
    assert float(normalized.max()) <= 1.0
    zero_one = raw.to(torch.float32).div(255.0)
    assert float(zero_one.min()) >= 0.0
    assert float(zero_one.max()) <= 1.0
    std = torch.tensor(STD).view(3, 1, 1)
    mean = torch.tensor(MEAN).view(3, 1, 1)
    denormalized = normalized * std + mean
    assert torch.allclose(denormalized, zero_one, atol=1e-6, rtol=0.0)
    black = torch.zeros(3, 1, 1, dtype=torch.uint8)
    white = torch.full((3, 1, 1), 255, dtype=torch.uint8)
    assert torch.equal(base.normalize_chw(black), -torch.ones(3, 1, 1))
    assert torch.equal(base.normalize_chw(white), torch.ones(3, 1, 1))
    batched = base.normalize_uint8_batch(raw.unsqueeze(0))
    assert batched.dtype == torch.float32
    assert torch.equal(batched[0], normalized)


def test_legacy_encoded_eval_cache_loads_and_matches(tmp_path: Path) -> None:
    records: list[SampleRecord] = []
    raws: list[torch.Tensor] = []
    for index, (name, factory) in enumerate(CASES[:3]):
        path, relative = _write_case(tmp_path, name, factory())
        records.append(_record(relative, index))
        raws.append(base.decode_image_file(path, IMAGE_SIZE))
    legacy_images = torch.stack(
        [
            ((base.normalize_chw(raw) + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8)
            for raw in raws
        ]
    )
    assert torch.equal(legacy_images, torch.stack(raws))
    relative_paths = [record.relative_path for record in records]
    cache = EvalCache(legacy_images, torch.arange(len(raws)), relative_paths)
    saved = cache.save(tmp_path / "legacy_eval_cache.pt")
    loaded = EvalCache.load(saved)
    assert torch.equal(loaded.images, legacy_images)
    assert loaded.paths == tuple(relative_paths)
    for index, record in enumerate(records):
        path = tmp_path.joinpath(*record.relative_path.split("/"))
        expected = base.preprocess_pil(_open_copy(path))
        batch = loaded.float_batch(index, index + 1)
        assert batch.dtype == torch.float32
        assert torch.equal(batch[0], expected)


def test_resize_is_bilinear_matching_pil_reference() -> None:
    source = _gradient_rgb(50, 37)
    prepared = base.ensure_rgb(source, IMAGE_SIZE)
    assert prepared.size == (IMAGE_SIZE, IMAGE_SIZE)
    reference = source.convert("RGB").resize(
        (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR
    )
    assert torch.equal(base.pil_to_chw_uint8(prepared), base.pil_to_chw_uint8(reference))
    assert base.INTERPOLATION == Image.Resampling.BILINEAR
    exact = _gradient_rgb(IMAGE_SIZE, IMAGE_SIZE)
    assert base.resize_bilinear(exact, IMAGE_SIZE) is exact


def test_inference_preprocessing_has_no_training_dependency() -> None:
    source = inspect.getsource(inference_preprocessing)
    assert "src.training" not in source
    assert "src.data.preprocessing" in source
    assert IMAGE_SIZE == base.BASE_IMAGE_SIZE == 64


def test_dataset_reexports_shared_mean_and_std() -> None:
    assert MEAN == base.MEAN == (0.5, 0.5, 0.5)
    assert STD == base.STD == (0.5, 0.5, 0.5)


def test_tensor_cache_rebuilds_corrupt_cache_with_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path, relative = _write_case(tmp_path, "cache_case", _random_rgb(32, 32))
    record = _record(relative, 0)
    cache_dir = tmp_path / "tensor_cache"
    first = TensorCache.load_or_build([record], tmp_path, IMAGE_SIZE, cache_dir)
    assert len(first) == 1
    first._mm._mmap.close()
    del first
    bin_path = next(cache_dir.glob("*.bin"))
    bin_path.write_bytes(b"\x00" * 16)
    with caplog.at_level(logging.WARNING, logger="src.training.tensor_cache"):
        rebuilt = TensorCache.load_or_build([record], tmp_path, IMAGE_SIZE, cache_dir)
    messages = [entry.getMessage() for entry in caplog.records]
    assert any("unusable tensor cache" in message for message in messages)
    assert len(rebuilt) == 1
    assert torch.equal(rebuilt.get(relative), base.decode_image_file(path, IMAGE_SIZE))
