"""Dataset and loader over Phase-3 ``SampleRecord`` references.

Images are opened lazily from the official ``images_root`` using the
project's existing image-path references — nothing is ever copied,
re-encoded, or written back into the dataset tree.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.data.continual import SampleRecord

MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)


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
    ) -> None:
        self.records = tuple(records)
        self.images_root = Path(images_root)
        self.image_size = int(image_size)
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

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        record = self.records[index]
        path = self.images_root.joinpath(*record.relative_path.split("/"))
        try:
            with Image.open(path) as handle:
                image = handle.convert("RGB")
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Image referenced by the official filelist is missing: {path}"
            ) from exc
        if image.size != (self.image_size, self.image_size):
            image = image.resize(
                (self.image_size, self.image_size), Image.Resampling.BILINEAR
            )
        tensor = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
        tensor = tensor.reshape(self.image_size, self.image_size, 3).permute(2, 0, 1)
        tensor = tensor.to(torch.float32) / 255.0
        mean = torch.tensor(MEAN).view(3, 1, 1)
        std = torch.tensor(STD).view(3, 1, 1)
        tensor = (tensor - mean) / std
        return tensor, int(record.label)


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
