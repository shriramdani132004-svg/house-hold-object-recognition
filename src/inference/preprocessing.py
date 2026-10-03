"""Preprocessing that matches the Phase-5 training pipeline exactly.

The contract is taken from :class:`src.training.dataset.ContinualImageDataset`
(nothing is re-invented):

1. decode to PIL and ``convert("RGB")`` (grayscale/RGBA/palette inputs),
2. resize to ``image_size`` (64) with ``Image.Resampling.BILINEAR`` when
   the size differs,
3. raw RGB bytes -> ``uint8`` HWC -> ``permute(2, 0, 1)`` CHW,
4. scale to ``float32`` in ``[0, 1]`` via ``/ 255.0``,
5. normalize with the training ``MEAN``/``STD`` of 0.5 per channel.

Any input error raises :class:`src.inference.types.InvalidImageError`
with a user-readable message instead of crashing the caller.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from src.inference.types import InvalidImageError
from src.training.dataset import MEAN, STD

IMAGE_SIZE = 64


def to_pil_rgb(image: object) -> Image.Image:
    """Convert a PIL image / NumPy array / camera frame to an RGB PIL image."""
    if image is None:
        raise InvalidImageError(
            "No image received. Capture a camera frame or upload an image."
        )
    if isinstance(image, Image.Image):
        rgb = image.convert("RGB")
        if rgb.width < 1 or rgb.height < 1:
            raise InvalidImageError(
                f"Image has invalid dimensions {image.size}; "
                "width and height must be at least 1 pixel."
            )
        return rgb
    if isinstance(image, np.ndarray):
        return _array_to_pil(image)
    raise InvalidImageError(
        f"Unsupported input type {type(image).__name__}; "
        "expected a PIL image, NumPy array, or camera frame."
    )


def _array_to_pil(array: np.ndarray) -> Image.Image:
    if array.size == 0:
        raise InvalidImageError("Image is empty (zero pixels).")
    if array.ndim == 2:
        gray = _to_uint8(array, "grayscale")
        return Image.fromarray(gray, mode="L").convert("RGB")
    if array.ndim != 3:
        raise InvalidImageError(
            f"Unexpected array shape {tuple(array.shape)}; "
            "expected (H, W), (H, W, 1), (H, W, 3), or (H, W, 4)."
        )
    channels = array.shape[2]
    if channels == 1:
        gray = _to_uint8(array[:, :, 0], "grayscale")
        return Image.fromarray(gray, mode="L").convert("RGB")
    if channels not in (3, 4):
        raise InvalidImageError(
            f"Unsupported channel count {channels} in shape "
            f"{tuple(array.shape)}; expected 1, 3 (RGB), or 4 (RGBA)."
        )
    data = _to_uint8(array, "color")
    if data.shape[0] < 1 or data.shape[1] < 1:
        raise InvalidImageError(
            f"Image has invalid dimensions {data.shape[:2]}; "
            "width and height must be at least 1 pixel."
        )
    mode = "RGB" if channels == 3 else "RGBA"
    return Image.fromarray(data, mode=mode).convert("RGB")


def _to_uint8(array: np.ndarray, kind: str) -> np.ndarray:
    if array.dtype == np.uint8:
        return array
    if array.dtype in (np.float32, np.float64):
        finite = np.isfinite(array)
        if not finite.all():
            raise InvalidImageError(
                "Image array contains non-finite values (NaN/Inf)."
            )
        low = float(array.min())
        high = float(array.max())
        if low < 0.0 or high > 1.0:
            raise InvalidImageError(
                f"Float image values must be in [0, 1] (got "
                f"[{low:.3f}, {high:.3f}]); use a uint8 array instead."
            )
        return (array * 255.0).round().astype(np.uint8)
    raise InvalidImageError(
        f"Unsupported array dtype {array.dtype} for a {kind} image; "
        "expected uint8 (or float in [0, 1])."
    )


def preprocess_image(
    image: object, image_size: int = IMAGE_SIZE
) -> torch.Tensor:
    """Convert any supported input into a normalized ``(1, 3, S, S)`` batch."""
    pil = to_pil_rgb(image)
    if pil.size != (image_size, image_size):
        pil = pil.resize((image_size, image_size), Image.Resampling.BILINEAR)
    tensor = torch.frombuffer(bytearray(pil.tobytes()), dtype=torch.uint8)
    tensor = tensor.reshape(image_size, image_size, 3).permute(2, 0, 1)
    tensor = tensor.to(torch.float32) / 255.0
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor.unsqueeze(0)
