"""Extract the official CORe50 archives into the project layout (Phase 2).

Run from the project root:

    .\\.venv\\Scripts\\python.exe scripts/extract_core50.py

Layout produced (official structure preserved — sessions, objects and
frame names are never flattened or renamed):

    data/raw/core50/
    ├── dataset/core50_128x128/s1..s11/o1..o50/C_ss_oo_fff.png
    ├── filelists/{NI,NC,NIC}_{inc,cum}/run0..run9/*.txt
    ├── metadata/dataset_dims/*.tsv
    ├── metadata/bbox/s1..s11/CropC_oNN.txt
    ├── metadata/*.txt|*.pkl
    └── metadata/core50-official/   (official repository snapshot)

Archives are kept in downloads/ until validation has verified them.
"""

from __future__ import annotations

import argparse
import sys
import time
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.core50 import (  # noqa: E402
    DATASET_DIR,
    DOWNLOAD_DIR,
    FILELIST_DIR,
    METADATA_DIR,
    PROJECT_ROOT as ROOT,
)
from src.utils.progress import PhaseProgress, format_duration  # noqa: E402

MB = 1024 * 1024
CHUNK = 1024 * 1024

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

# (archive, destination, prefix to strip from member names)
JOBS: tuple[tuple[Path, Path, str], ...] = (
    (DOWNLOAD_DIR / "core50_128x128.zip", DATASET_DIR, ""),
    (DOWNLOAD_DIR / "batches_filelists.zip", FILELIST_DIR, "batches_filelists/"),
    (DOWNLOAD_DIR / "dataset_dims.zip", METADATA_DIR / "dataset_dims", "dataset_dims/"),
    (DOWNLOAD_DIR / "bbox.zip", METADATA_DIR / "bbox", "bbox/"),
    (
        DOWNLOAD_DIR / "core50-master.zip",
        METADATA_DIR / "core50-official",
        "core50-master/",
    ),
    # NICv2 scenario filelists ship inside the official repository snapshot
    # (second CORe50 paper: "Fine-Grained Continual Learning").
    (
        METADATA_DIR / "core50-official" / "extras" / "batches_filelists_NICv2.zip",
        FILELIST_DIR,
        "",
    ),
)


class ExtractError(RuntimeError):
    """Raised when an archive cannot be extracted safely."""


def _human(count: int) -> str:
    value = float(count)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024 or unit == "GiB":
            return f"{value:.2f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.2f} GiB"


def member_target(dest: Path, prefix: str, member: str) -> Path:
    name = member.replace("\\", "/")
    if prefix:
        if not name.startswith(prefix):
            raise ExtractError(f"member {member!r} does not start with {prefix!r}")
        name = name[len(prefix) :]
    if not name or name.endswith("/"):
        raise ExtractError(f"unexpected directory member {member!r}")
    if ".." in Path(name).parts:
        raise ExtractError(f"unsafe member path {member!r}")
    return dest / name


def extract_job(
    archive: Path,
    dest: Path,
    prefix: str,
    progress: PhaseProgress,
    *,
    label: str,
    counters: dict[str, int],
) -> dict[str, object]:
    if not archive.is_file():
        raise ExtractError(f"missing archive: {archive}")
    with zipfile.ZipFile(archive) as zf:
        infos = [
            info
            for info in zf.infolist()
            if not info.is_dir() and not info.filename.endswith("/")
        ]
        total_bytes = sum(info.file_size for info in infos)
        total_files = len(infos)
        counters["files_total"] += total_files
        counters["bytes_total"] += total_bytes

        skipped = 0
        last_update = time.monotonic()
        for info in infos:
            target = member_target(dest, prefix, info.filename)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.is_file() and target.stat().st_size == info.file_size:
                skipped += 1
            else:
                with zf.open(info) as source, target.open("wb") as handle:
                    while chunk := source.read(CHUNK):
                        handle.write(chunk)
                if target.stat().st_size != info.file_size:
                    raise ExtractError(
                        f"short extract for {info.filename}: "
                        f"{target.stat().st_size} != {info.file_size}"
                    )
            counters["files_done"] += 1
            counters["bytes_done"] += info.file_size
            now = time.monotonic()
            if now - last_update >= 0.3:
                last_update = now
                elapsed = max(now - progress.started, 1e-6)
                rate = counters["files_done"] / elapsed
                remaining = (
                    (counters["files_total"] - counters["files_done"]) / rate
                    if rate > 0
                    else 0
                )
                progress.update(
                    counters["files_done"],
                    counters["files_total"],
                    detail=(
                        f"{label}  {_human(counters['bytes_done'])} / "
                        f"{_human(counters['bytes_total'])}  "
                        f"ETA {format_duration(remaining)}"
                    ),
                )

    return {
        "archive": archive.name,
        "destination": str(dest.relative_to(ROOT).as_posix()),
        "files": total_files,
        "bytes_uncompressed": total_bytes,
        "files_skipped_existing": skipped,
    }


def run(*, dest_dir: Path = DOWNLOAD_DIR) -> int:
    progress = PhaseProgress("PHASE 2 OVERALL", STEP_LABELS)
    progress.set_step(4, "Extracting / organizing")
    counters = {"files_done": 0, "files_total": 0, "bytes_done": 0, "bytes_total": 0}
    results: list[dict[str, object]] = []

    for archive_path, dest, prefix in JOBS:
        archive = archive_path
        if not archive.is_absolute():
            archive = dest_dir / archive
        dest.mkdir(parents=True, exist_ok=True)
        progress.log(f"[unzp] {archive.name} -> {dest.relative_to(ROOT).as_posix()}")
        results.append(
            extract_job(
                archive,
                dest,
                prefix,
                progress,
                label=archive.name,
                counters=counters,
            )
        )
        progress.log(
            f"[ ok ] {archive.name}: "
            f"{results[-1]['files']:,} files "
            f"({_human(int(results[-1]['bytes_uncompressed']))})"
        )

    # Verify no zero-byte file landed anywhere in the extracted tree.
    empty = [
        str(path.relative_to(ROOT).as_posix())
        for root in (DATASET_DIR, FILELIST_DIR, METADATA_DIR)
        for path in root.rglob("*")
        if path.is_file() and path.stat().st_size == 0
    ]
    if empty:
        raise ExtractError(f"zero-byte files after extraction: {empty[:5]}")

    progress.update(counters["files_done"], counters["files_total"], force=True)
    progress.finish(
        f"Extraction complete: {counters['files_done']:,} files, "
        f"{_human(counters['bytes_done'])} written"
    )
    for result in results:
        print(
            f"  {result['archive']}: {result['files']:,} files -> "
            f"{result['destination']}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--downloads",
        type=Path,
        default=DOWNLOAD_DIR,
        help="directory holding the downloaded archives",
    )
    args = parser.parse_args(argv)
    try:
        return run(dest_dir=args.downloads)
    except ExtractError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
