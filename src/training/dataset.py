"""Dataset and loader over Phase-3 ``SampleRecord`` references.

Images are opened lazily from the official ``images_root`` using the
project's existing image-path references — nothing is ever copied,
re-encoded, or written back into the dataset tree. The image transform
itself is the shared one from :mod:`src.data.preprocessing` (the single
source of truth for training, evaluation, and inference); ``MEAN`` and
``STD`` are re-exported here for existing importers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from src.data import preprocessing as base_preprocessing
from src.data.continual import SampleRecord
from src.data.preprocessing import decode_image_file, normalize_chw
from src.training.tensor_cache import TensorCache

MEAN = base_preprocessing.MEAN
STD = base_preprocessing.STD


class ContinualImageDataset(Dataset):
    """Torch dataset over official filelist references.

    ``require_split`` guards the train/evaluation boundary: a dataset built
    with ``require_split="train"`` refuses evaluation (``split="test"``)
    records, so test data can never silently enter a training loop.
    """

    def __init__(
        self,
        records: Sequence[SampleRecord],
        images_root: str | Path,
        image_size: int,
        *,
        require_split: str | None = None,
        cache: TensorCache | None = None,
    ) -> None:
        self.records = tuple(records)
        self.images_root = Path(images_root)
        self.image_size = int(image_size)
        self.cache = cache
        if self.image_size < 8:
            raise ValueError(f"image_size must be >= 8, got {self.image_size}")
        if require_split is not None:
            bad = [r for r in self.records if r.split != require_split]
            if bad:
                raise ValueError(
                    f"Dataset requires split={require_split!r} but contains "
                    f"{len(bad)} record(s) with split={bad[0].split!r} "
                    f"(e.g. {bad[0].relative_path})"
                )

    def __len__(self) -> int:
        return len(self.records)

    def get_uint8(self, index: int) -> torch.Tensor:
        """Decode record ``index`` to its raw post-resize ``uint8`` CHW tensor.

        Served from the decoded tensor cache when present (identical pixels —
        the cache stores the output of the same shared decode pipeline).
        """
        record = self.records[index]
        if self.cache is not None and record.relative_path in self.cache:
            return self.cache.get(record.relative_path)
        path = self.images_root.joinpath(*record.relative_path.split("/"))
        try:
            return decode_image_file(path, self.image_size)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Image referenced by the official filelist is missing: {path}"
            ) from exc

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        tensor = normalize_chw(self.get_uint8(index))
        return tensor, int(self.records[index].label)


def make_loader(
    dataset: ContinualImageDataset,
    *,
    batch_size: int,
    seed: int,
    workers: int = 0,
    shuffle: bool = True,
) -> DataLoader:
    """Deterministic-shuffle DataLoader (explicit generator seed)."""
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=shuffle,
        num_workers=int(workers),
        generator=generator,
        drop_last=False,
    )
