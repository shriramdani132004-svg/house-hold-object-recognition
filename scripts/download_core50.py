"""Acquire the official CORe50 resources for Phase 2.

Run from the project root:

    .\\.venv\\Scripts\\python.exe scripts/download_core50.py

Downloads, with resume support and a fixed phase-progress header, the
official CORe50 materials listed in ``src.data.core50.RESOURCES`` into
``data/raw/core50/downloads/`` and records a verified manifest in
``data/raw/core50/downloads/MANIFEST.json``.  Nothing outside
``data/raw/core50/`` is written, so the COCO prototype is untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import shutil
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.core50 import (  # noqa: E402
    DOWNLOAD_DIR,
    OFFICIAL_PAGE,
    OFFICIAL_REPO,
    PROJECT_ROOT as ROOT,
    RESOURCES,
    Resource,
)
from src.utils.progress import PhaseProgress, format_duration  # noqa: E402

MB = 1024 * 1024
CHUNK = 1024 * 1024
USER_AGENT = "household-object-recognition/1.0 (CORe50 acquisition script)"

STEP_LABELS = (
    "Project / environment check",
    "Official CORe50 source inspection",
    "Downloading CORe50",
    "Extracting / organizing",
    "Validating dataset structure",
    "Validating sessions / objects / filelists",
    "Documentation + tests",
    "Final verification + git commit",
)


class DownloadError(RuntimeError):
    """Raised when an official resource cannot be acquired."""


class TransferInterrupted(DownloadError):
    """Raised for retryable transfer failures."""


def _human(count: int) -> str:
    value = float(count)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024 or unit == "GiB":
            return f"{value:.2f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.2f} GiB"


def head(url: str, *, timeout: float = 60.0) -> dict[str, str]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return {k.lower(): v for k, v in response.headers.items()}


def resolve_size(resource: Resource) -> int:
    """Confirm the official byte size (fresh HEAD) for this resource."""
    if resource.size_bytes:
        return resource.size_bytes
    headers = head(resource.url)
    return int(headers.get("content-length", "0"))


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def stream_to_file(
    url: str,
    dest: Path,
    offset: int,
    expected: int,
    progress: PhaseProgress,
    *,
    detail: str,
    timeout: float = 120.0,
    update_every: float = 0.4,
) -> None:
    request = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, "Range": f"bytes={offset}-"}
    )
    try:
        response = urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and dest.exists():
            dest.unlink()
            raise TransferInterrupted(
                "server rejected resume range (HTTP 416); partial file removed"
            ) from exc
        raise

    with response:
        status = getattr(response, "status", None) or response.getcode()
        if status not in (200, 206):
            raise TransferInterrupted(f"unexpected HTTP status {status} from {url}")
        mode = "ab" if (status == 206 and offset > 0) else "wb"
        written = offset if mode == "ab" else 0
        started = time.monotonic()
        last_update = 0.0
        with dest.open(mode) as handle:
            while expected <= 0 or written < expected:
                chunk = response.read(CHUNK if expected > 0 else min(CHUNK, 65536))
                if not chunk:
                    break
                handle.write(chunk)
                written += len(chunk)
                now = time.monotonic()
                if now - last_update >= update_every:
                    last_update = now
                    elapsed = max(now - started, 1e-6)
                    speed = (written - offset) / MB / elapsed
                    if expected > 0:
                        eta = (
                            (expected - written) / (speed * MB) if speed > 0 else 0
                        )
                        detail_line = (
                            f"{detail}  {_human(written)} / {_human(expected)}  "
                            f"{speed:,.2f} MiB/s  elapsed "
                            f"{format_duration(elapsed)}  ETA {format_duration(eta)}"
                        )
                        progress.update(written, expected, detail=detail_line)
                    else:
                        progress.update(
                            written,
                            0,
                            detail=(
                                f"{detail}  {_human(written)} downloaded  "
                                f"{speed:,.2f} MiB/s  elapsed "
                                f"{format_duration(elapsed)} (size unknown)"
                            ),
                        )
    if expected > 0 and written != expected:
        raise TransferInterrupted(
            f"connection ended early for {dest.name}: {written} / {expected} bytes"
        )
    if expected <= 0:
        if written <= 0:
            raise TransferInterrupted(f"empty response for {dest.name}")
        progress.update(
            written,
            written,
            detail=f"{detail}  complete ({_human(written)})",
        )
    else:
        progress.update(written, expected, detail=f"{detail}  complete")


def download_resource(
    resource: Resource,
    dest_dir: Path,
    progress: PhaseProgress,
    *,
    index: int,
    count: int,
    max_attempts: int = 8,
    sleep=time.sleep,
) -> dict[str, object]:
    dest = dest_dir / resource.dest
    dest_dir.mkdir(parents=True, exist_ok=True)
    expected = resolve_size(resource)
    detail = f"[{index}/{count}] {resource.name}"

    if dest.is_file() and expected and dest.stat().st_size == expected:
        progress.log(f"[skip] {resource.name}: already present ({_human(expected)})")
        return {
            "name": resource.name,
            "url": resource.url,
            "path": str(dest.relative_to(ROOT).as_posix()),
            "size_bytes": expected,
            "sha256": sha256_of(dest),
            "status": "already-present",
        }

    progress.log(
        f"[get ] {resource.name}: target {_human(expected)} "
        f"({'resuming' if dest.is_file() else 'starting'})"
    )
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        offset = dest.stat().st_size if dest.is_file() else 0
        if offset > expected > 0:
            dest.unlink()
            offset = 0
        try:
            stream_to_file(
                resource.url,
                dest,
                offset,
                expected,
                progress,
                detail=detail,
            )
        except (
            TransferInterrupted,
            urllib.error.URLError,
            http.client.HTTPException,
            TimeoutError,
            OSError,
        ) as exc:
            last_error = exc
            delay = min(30.0, 2.0**attempt)
            progress.log(
                f"[warn] {resource.name}: attempt {attempt}/{max_attempts} failed "
                f"({exc}); retrying in {delay:.0f}s"
            )
            if attempt < max_attempts:
                sleep(delay)
            continue

        actual = dest.stat().st_size
        if expected > 0 and actual != expected:
            raise DownloadError(
                f"{resource.name}: expected {expected} bytes, got {actual}"
            )
        if actual <= 0:
            raise DownloadError(f"{resource.name}: downloaded file is empty")
        progress.log(f"[ ok ] {resource.name}: {_human(actual)}")
        return {
            "name": resource.name,
            "url": resource.url,
            "path": str(dest.relative_to(ROOT).as_posix()),
            "size_bytes": actual,
            "sha256": sha256_of(dest),
            "status": "downloaded",
        }

    raise DownloadError(f"could not acquire {resource.name}: {last_error}")


def ensure_free_space(dest_dir: Path, resources: tuple[Resource, ...]) -> None:
    total = 0
    for resource in resources:
        try:
            total += resolve_size(resource)
        except Exception:  # noqa: BLE001 - size resolved again during download
            total += resource.size_bytes
    needed = total + 8 * 1024 * MB  # room for extraction of the archives
    root = dest_dir if dest_dir.exists() else dest_dir.parent
    free = shutil.disk_usage(root).free
    if free < needed:
        raise DownloadError(
            f"insufficient disk space: need ~{_human(needed)}, free {_human(free)}"
        )
    print(f"[info] disk space OK: need ~{_human(needed)}, free {_human(free)}")


def run(*, required_only: bool = False, dest_dir: Path = DOWNLOAD_DIR) -> int:
    resources = [r for r in RESOURCES if r.required or not required_only]
    progress = PhaseProgress("PHASE 2 OVERALL", STEP_LABELS)
    progress.set_step(3, "Downloading CORe50")
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"Destination: {dest_dir}")
    print(f"Official page: {OFFICIAL_PAGE}")
    print(f"Official repository: {OFFICIAL_REPO}")
    ensure_free_space(dest_dir, tuple(resources))

    entries: list[dict[str, object]] = []
    for index, resource in enumerate(resources, start=1):
        entry = download_resource(
            resource, dest_dir, progress, index=index, count=len(resources)
        )
        entry["purpose"] = resource.purpose
        entry["required"] = resource.required
        entries.append(entry)

    manifest = {
        "dataset": "CORe50",
        "official_page": OFFICIAL_PAGE,
        "official_repository": OFFICIAL_REPO,
        "acquired_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "resources": entries,
    }
    manifest_path = dest_dir / "MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    progress.update(len(resources), len(resources), detail="all resources acquired")
    progress.finish(
        f"Acquisition complete: {len(entries)} resources, "
        f"{_human(sum(int(e['size_bytes']) for e in entries))} total"
    )
    print(f"Manifest written: {manifest_path.relative_to(ROOT).as_posix()}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--required-only",
        action="store_true",
        help="skip optional resources (bbox, LUP/labels pickles, repo snapshot)",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=DOWNLOAD_DIR,
        help="destination directory (default: data/raw/core50/downloads)",
    )
    args = parser.parse_args(argv)
    try:
        return run(required_only=args.required_only, dest_dir=args.dest)
    except DownloadError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
