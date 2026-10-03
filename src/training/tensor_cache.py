"""Decoded-image tensor cache for fast continual training (CPU-friendly).

Decodes every referenced image ONCE (PIL open -> RGB -> bilinear resize)
into a memory-mapped ``uint8`` file, then serves exact byte-identical
tensors to the training loop. This removes repeated PNG decoding — the
dominant cost of CPU training — without changing any preprocessing math:
the cached bytes are exactly what ``ContinualImageDataset`` would decode,
and normalization is applied downstream exactly as before.

Layout: one ``.bin`` file of shape ``(N, 3, H, W)`` uint8 plus a ``.json``
index mapping project-relative paths to rows. Files are keyed by image
size and by a hash of the path set, so caches are self-validating.
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from src.data.continual import SampleRecord
from src.data.preprocessing import decode_image_file

logger = logging.getLogger(__name__)


class TensorCacheError(ValueError):
    """Raised when a decoded tensor cache is missing or inconsistent."""


def _decode_one(path: Path, image_size: int) -> np.ndarray:
    return decode_image_file(path, image_size).numpy()


class TensorCache:
    """Memory-mapped decoded images addressed by relative path."""

    def __init__(self, bin_path: Path, index_path: Path) -> None:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("format_version") != 1:
            raise TensorCacheError(f"Unsupported tensor cache index: {index_path}")
        self.image_size = int(payload["image_size"])
        self.paths: tuple[str, ...] = tuple(payload["paths"])
        self._offsets = {path: i for i, path in enumerate(self.paths)}
        shape = (len(self.paths), 3, self.image_size, self.image_size)
        expected_bytes = int(np.prod(shape))
        if bin_path.stat().st_size != expected_bytes:
            raise TensorCacheError(
                f"Tensor cache binary size mismatch: {bin_path} "
                f"({bin_path.stat().st_size} != {expected_bytes})"
            )
        self._mm = np.memmap(bin_path, dtype=np.uint8, mode="r", shape=shape)

    def __len__(self) -> int:
        return len(self.paths)

    def __contains__(self, rel_path: str) -> bool:
        return rel_path in self._offsets

    def get(self, rel_path: str) -> torch.Tensor:
        """Return the decoded uint8 image as ``(3, H, W)``."""
        offset = self._offsets.get(rel_path)
        if offset is None:
            raise TensorCacheError(f"Path not in tensor cache: {rel_path}")
        return torch.from_numpy(np.array(self._mm[offset], copy=True))

    @staticmethod
    def cache_key(paths: Sequence[str]) -> str:
        blob = "\n".join(sorted(set(paths))).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:16]

    @classmethod
    def load_or_build(
        cls,
        records: Sequence[SampleRecord],
        images_root: str | Path,
        image_size: int,
        cache_dir: str | Path,
        *,
        workers: int = 8,
        on_progress=None,
    ) -> "TensorCache":
        """Open the cache for ``records``, building it if absent or stale."""
        paths = list(dict.fromkeys(record.relative_path for record in records))
        key = cls.cache_key(paths)
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        stem = f"decoded_{image_size}_{key}"
        bin_path = cache_dir / f"{stem}.bin"
        index_path = cache_dir / f"{stem}.json"
        if bin_path.is_file() and index_path.is_file():
            try:
                cache = cls(bin_path, index_path)
                if len(cache.paths) == len(paths):
                    return cache
            except TensorCacheError as exc:
                logger.warning(
                    "Discarding unusable tensor cache %s (%s); rebuilding",
                    bin_path,
                    exc,
                )
        cls._build(paths, images_root, image_size, bin_path, index_path,
                   workers=workers, on_progress=on_progress)
        return cls(bin_path, index_path)

    @classmethod
    def _build(
        cls,
        paths: list[str],
        images_root: str | Path,
        image_size: int,
        bin_path: Path,
        index_path: Path,
        *,
        workers: int,
        on_progress,
    ) -> None:
        images_root = Path(images_root)
        count = len(paths)
        row_bytes = 3 * image_size * image_size
        tmp_bin = bin_path.with_suffix(".bin.tmp")
        lock = threading.Lock()
        done = 0

        with tmp_bin.open("wb") as handle:
            handle.truncate(count * row_bytes)

            def task(index: int) -> None:
                nonlocal done
                rel = paths[index]
                path = images_root.joinpath(*rel.split("/"))
                array = _decode_one(path, image_size)
                with lock:
                    handle.seek(index * row_bytes)
                    handle.write(np.ascontiguousarray(array).tobytes())
                    done += 1
                    if on_progress is not None and (done % 10000 == 0 or done == count):
                        on_progress(done, count)

            with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
                list(pool.map(task, range(count)))

        tmp_bin.replace(bin_path)
        index = {
            "format_version": 1,
            "image_size": image_size,
            "paths": paths,
        }
        tmp_index = index_path.with_suffix(".json.tmp")
        tmp_index.write_text(json.dumps(index), encoding="utf-8")
        tmp_index.replace(index_path)
