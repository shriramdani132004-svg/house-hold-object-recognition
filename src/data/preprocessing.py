"""The ONE authoritative base preprocessing for the project.

Training, evaluation, and inference all share this module for the base
image transform; no other module may re-implement it:

1. decode to PIL and convert to ``RGB`` (grayscale/RGBA/palette inputs),
2. resize to ``(image_size, image_size)`` with
   :data:`INTERPOLATION` (bilinear) only when the size differs,
3. raw RGB bytes -> ``uint8`` CHW tensor,
4. ``float32`` division by ``255.0`` -> ``[0, 1]``,
5. ``(x - MEAN) / STD`` with ``MEAN = STD = (0.5, 0.5, 0.5)`` -> ``[-1, 1]``.

There is no crop. Training-only augmentation (horizontal flip, small
translation, mild brightness/contrast — see ``src.training.improved``) is
applied on top of these outputs and deliberately lives outside this
module.

The decode path is deterministic: every caller goes through the same
functions, so a given file always yields byte-identical tensors.
"""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

BASE_IMAGE_SIZE = 64
INTERPOLATION = Image.Resampling.BILINEAR
MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)

_MEAN = torch.tensor(MEAN, dtype=torch.float32).view(3, 1, 1)
_STD = torch.tensor(STD, dtype=torch.float32).view(3, 1, 1)
_MEAN_BATCH = _MEAN.view(1, 3, 1, 1)
_STD_BATCH = _STD.view(1, 3, 1, 1)


def to_rgb_pil(image: Image.Image) -> Image.Image:
    """Return the RGB version of ``image`` (grayscale/RGBA/palette safe)."""
    return image.convert("RGB")


def resize_bilinear(image: Image.Image, image_size: int) -> Image.Image:
    """Bilinear-resize to ``image_size`` x ``image_size`` when size differs."""
    if image.size == (image_size, image_size):
        return image
    return image.resize((image_size, image_size), INTERPOLATION)


def ensure_rgb(image: Image.Image, image_size: int) -> Image.Image:
    """Canonical PIL prepare step: RGB conversion plus resize-if-needed."""
    return resize_bilinear(to_rgb_pil(image), image_size)


def pil_to_chw_uint8(image: Image.Image) -> torch.Tensor:
    """Raw RGB bytes of an RGB ``image`` as a ``uint8`` ``(3, H, W)`` tensor."""
    width, height = image.size
    buffer = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
    return buffer.reshape(height, width, 3).permute(2, 0, 1)


def normalize_chw(image: torch.Tensor) -> torch.Tensor:
    """``uint8`` CHW -> ``float32`` CHW in ``[-1, 1]``.

    One fixed op order: divide by ``255.0`` first, then subtract
    ``MEAN`` and divide by ``STD``.
    """
    tensor = image.to(torch.float32).div(255.0)
    return (tensor - _MEAN) / _STD


def normalize_uint8_batch(images: torch.Tensor) -> torch.Tensor:
    """``uint8`` NCHW batch -> ``float32`` batch, same op order as
    :func:`normalize_chw`; elementwise math makes the values identical."""
    tensor = images.to(torch.float32).div(255.0)
    return (tensor - _MEAN_BATCH) / _STD_BATCH


def preprocess_pil(
    image: Image.Image, image_size: int = BASE_IMAGE_SIZE
) -> torch.Tensor:
    """Full base transform: RGB + resize -> uint8 CHW -> normalized float32."""
    return normalize_chw(pil_to_chw_uint8(ensure_rgb(image, image_size)))


def decode_image_file(
    path: str | Path, image_size: int = BASE_IMAGE_SIZE
) -> torch.Tensor:
    """Deterministically decode ``path`` to a raw ``uint8`` ``(3, S, S)`` tensor."""
    with Image.open(Path(path)) as handle:
        image = ensure_rgb(handle, image_size)
    return pil_to_chw_uint8(image)
