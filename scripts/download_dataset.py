"""Acquire the official COCO 2017 archives selected for this project.

Run from the project root:

    python scripts/download_dataset.py [--include-train]

Downloads resumable archives into data/raw/coco/zips/, verifies them
(byte size, MD5 where available, ZIP CRC), and extracts them into
data/raw/coco/ without ever modifying the originals.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "coco"

BASE_URLS: tuple[str, ...] = (
    "https://s3.amazonaws.com/images.cocodataset.org",
    "http://images.cocodataset.org",
)

USER_AGENT = "household-object-recognition/1.0 (dataset acquisition script)"

MB = 1024 * 1024


class DatasetError(RuntimeError):
    """Raised when acquisition or verification of dataset files fails."""


@dataclass(frozen=True)
class Archive:
    name: str
    relative_path: str
    url_path: str
    size_bytes: int
    md5: str | None = None
    required_members: tuple[str, ...] = field(default_factory=tuple)
    image_prefix: str | None = None
    image_count: int | None = None


ANNOTATIONS = Archive(
    name="annotations_trainval2017.zip",
    relative_path="zips/annotations_trainval2017.zip",
    url_path="/annotations/annotations_trainval2017.zip",
    size_bytes=252_907_541,
    md5="f4bbac642086de4f52a3fdda2de5fa2c",
    required_members=(
        "annotations/instances_train2017.json",
        "annotations/instances_val2017.json",
    ),
)

VAL2017 = Archive(
    name="val2017.zip",
    relative_path="zips/val2017.zip",
    url_path="/zips/val2017.zip",
    size_bytes=815_585_330,
    image_prefix="val2017/",
    image_count=5_000,
)

TRAIN2017 = Archive(
    name="train2017.zip",
    relative_path="zips/train2017.zip",
    url_path="/zips/train2017.zip",
    size_bytes=19_336_861_798,
    image_prefix="train2017/",
)


def human_bytes(count: int) -> str:
    value = float(count)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024 or unit == "GiB":
            return f"{value:.2f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.2f} GiB"


def md5_of(path: Path, chunk_size: int = MB) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def check_size(path: Path, expected: int) -> None:
    if not path.is_file():
        raise DatasetError(f"missing file: {path}")
    actual = path.stat().st_size
    if actual != expected:
        raise DatasetError(
            f"size mismatch for {path.name}: expected {expected} bytes, found {actual}"
        )


def _normalised_names(archive: zipfile.ZipFile) -> list[str]:
    return [name.replace("\\", "/") for name in archive.namelist()]


def check_zip(
    path: Path,
    *,
    required_members: Sequence[str] = (),
    image_prefix: str | None = None,
    image_count: int | None = None,
    deep: bool = True,
) -> list[str]:
    try:
        with zipfile.ZipFile(path) as zf:
            if deep:
                bad = zf.testzip()
                if bad is not None:
                    raise DatasetError(f"CRC failure inside {path.name}: {bad}")
            names = _normalised_names(zf)
    except zipfile.BadZipFile as exc:
        raise DatasetError(f"{path.name} is not a valid zip archive: {exc}") from exc

    for member in required_members:
        if not any(name.endswith(member) for name in names):
            raise DatasetError(f"{path.name} does not contain required member {member}")

    if image_count is not None:
        prefix = image_prefix or ""
        images = [
            name
            for name in names
            if name.startswith(prefix)
            and not name.endswith("/")
            and name.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        if len(images) != image_count:
            raise DatasetError(
                f"{path.name}: expected {image_count} images, archive holds {len(images)}"
            )
    return names


def verify_archive(path: Path, archive: Archive, *, deep: bool = True) -> None:
    check_size(path, archive.size_bytes)
    if archive.md5:
        actual = md5_of(path)
        if actual.lower() != archive.md5.lower():
            raise DatasetError(
                f"MD5 mismatch for {path.name}: expected {archive.md5}, found {actual}"
            )
    check_zip(
        path,
        required_members=archive.required_members,
        image_prefix=archive.image_prefix,
        image_count=archive.image_count,
        deep=deep,
    )


def _progress_line(
    name: str, done: int, total: int, started: float, now: float
) -> str:
    elapsed = max(now - started, 1e-6)
    rate = done / MB / elapsed
    percent = 100.0 * done / total
    remaining = (total - done) / (done / elapsed) if done else 0.0
    return (
        f"  {name}: {done / MB:,.1f} / {total / MB:,.1f} MiB "
        f"({percent:5.1f}%)  {rate:,.2f} MiB/s  ETA {remaining / 60:,.1f} min"
    )


def _stream_to_file(
    url: str,
    dest: Path,
    offset: int,
    expected: int,
    *,
    timeout: float,
    progress_interval: float,
) -> None:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Range": f"bytes={offset}-"},
    )
    try:
        response = urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and dest.exists():
            dest.unlink()
            raise DatasetError(
                "server rejected resume range (HTTP 416); partial file removed"
            ) from exc
        raise

    with response:
        status = getattr(response, "status", None) or response.getcode()
        if status not in (200, 206):
            raise DatasetError(f"unexpected HTTP status {status} from {url}")
        mode = "ab" if (status == 206 and offset > 0) else "wb"
        written = offset if mode == "ab" else 0
        started = time.monotonic()
        last_print = started
        with dest.open(mode) as handle:
            while written < expected:
                chunk = response.read(min(MB, expected - written))
                if not chunk:
                    break
                handle.write(chunk)
                written += len(chunk)
                now = time.monotonic()
                if now - last_print >= progress_interval:
                    print(_progress_line(dest.name, written, expected, started, now))
                    last_print = now

    if written != expected:
        raise DatasetError(
            f"connection ended early for {dest.name}: {written} / {expected} bytes"
        )
    print(_progress_line(dest.name, written, expected, started, time.monotonic()))


def download_archive(
    archive: Archive,
    dest_dir: Path,
    *,
    max_attempts: int = 8,
    timeout: float = 60.0,
    progress_interval: float = 5.0,
    sleep: Callable[[float], None] = time.sleep,
) -> Path:
    dest = dest_dir / archive.relative_path
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.is_file() and dest.stat().st_size == archive.size_bytes:
        print(f"[skip] {archive.name}: already downloaded ({human_bytes(archive.size_bytes)})")
        return dest

    print(f"[get ] {archive.name}: target {human_bytes(archive.size_bytes)}")
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        if dest.is_file():
            size = dest.stat().st_size
            if size > archive.size_bytes:
                dest.unlink()
                size = 0
            offset = size
        else:
            offset = 0

        base = BASE_URLS[0] if attempt <= max_attempts - 3 else BASE_URLS[-1]
        url = base + archive.url_path
        try:
            _stream_to_file(
                url,
                dest,
                offset,
                archive.size_bytes,
                timeout=timeout,
                progress_interval=progress_interval,
            )
            verify_archive(dest, archive, deep=True)
            print(f"[ ok ] {archive.name}: verified ({human_bytes(archive.size_bytes)})")
            return dest
        except (DatasetError, urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            delay = min(30.0, 2.0**attempt)
            print(
                f"[warn] {archive.name}: attempt {attempt}/{max_attempts} failed "
                f"({exc}); retrying in {delay:.0f}s"
            )
            if attempt < max_attempts:
                sleep(delay)

    raise DatasetError(f"could not acquire {archive.name}: {last_error}")


def is_extracted(archive: Archive, dest_dir: Path) -> bool:
    if archive.required_members:
        return all((dest_dir / member).is_file() for member in archive.required_members)
    if archive.image_count is not None and archive.image_prefix:
        folder = dest_dir / archive.image_prefix.rstrip("/")
        if not folder.is_dir():
            return False
        count = sum(
            1 for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        return count == archive.image_count
    return False


def extract_archive(archive: Archive, dest_dir: Path) -> None:
    if is_extracted(archive, dest_dir):
        print(f"[skip] {archive.name}: already extracted")
        return
    source = dest_dir / archive.relative_path
    print(f"[unzp] {archive.name} -> {dest_dir}")
    with zipfile.ZipFile(source) as zf:
        zf.extractall(dest_dir)
    if not is_extracted(archive, dest_dir):
        raise DatasetError(f"extraction of {archive.name} did not produce expected files")
    print(f"[ ok ] {archive.name}: extracted")


def ensure_free_space(dest_dir: Path, archives: Sequence[Archive]) -> None:
    needed = 2 * sum(archive.size_bytes for archive in archives) + 512 * MB
    root = dest_dir if dest_dir.exists() else dest_dir.parent
    free = shutil.disk_usage(root).free
    if free < needed:
        raise DatasetError(
            f"insufficient disk space: need ~{human_bytes(needed)}, have {human_bytes(free)}"
        )
    print(
        f"[info] disk space OK: need ~{human_bytes(needed)}, free {human_bytes(free)}"
    )


def run(include_train: bool = False, dest_dir: Path = RAW_DIR) -> None:
    archives: list[Archive] = [ANNOTATIONS, VAL2017]
    if include_train:
        archives.append(TRAIN2017)
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"Destination: {dest_dir}")
    ensure_free_space(dest_dir, archives)
    for archive in archives:
        download_archive(archive, dest_dir)
        extract_archive(archive, dest_dir)
    print("Dataset acquisition complete.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include-train",
        action="store_true",
        help="also download train2017.zip (18 GiB official training split)",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=RAW_DIR,
        help="destination directory (default: data/raw/coco)",
    )
    args = parser.parse_args(argv)
    try:
        run(include_train=args.include_train, dest_dir=args.dest)
    except DatasetError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
