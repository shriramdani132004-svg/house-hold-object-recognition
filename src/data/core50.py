"""Official CORe50 dataset constants and project paths.

Source of truth: the official project page (https://vlomonaco.github.io/core50/)
and the official repository (https://github.com/vlomonaco/core50), both
inspected during Phase 2. All values below are taken from those official
materials or from HTTP HEAD responses of the official hosts — none are
guessed.

The 128x128 RGB archive is the official benchmark image set (the one the
official ``fetch_data_and_setup.sh`` script downloads and the one used by
the official data loader). Depth, full-size 350x350 and TF-record archives
are optional modalities that Phase 2 deliberately does not download.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CORE50_ROOT = PROJECT_ROOT / "data" / "raw" / "core50"
DOWNLOAD_DIR = CORE50_ROOT / "downloads"
DATASET_DIR = CORE50_ROOT / "dataset"
FILELIST_DIR = CORE50_ROOT / "filelists"
METADATA_DIR = CORE50_ROOT / "metadata"

OFFICIAL_PAGE = "https://vlomonaco.github.io/core50/"
OFFICIAL_REPO = "https://github.com/vlomonaco/core50"
IMAGE_HOST = "http://bias.csr.unibo.it/maltoni/download/core50/"
PAGES_HOST = "https://vlomonaco.github.io/core50/data/"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"

SESSION_IDS: tuple[int, ...] = tuple(range(1, 12))
TEST_SESSIONS: tuple[int, ...] = (3, 7, 10)
TRAIN_SESSIONS: tuple[int, ...] = tuple(
    s for s in SESSION_IDS if s not in TEST_SESSIONS
)
OBJECT_IDS: tuple[int, ...] = tuple(range(1, 51))

OFFICIAL_CATEGORY_ORDER: tuple[str, ...] = (
    "plug adapter",
    "mobile phone",
    "scissors",
    "light bulb",
    "can",
    "glass",
    "ball",
    "marker",
    "cup",
    "remote control",
)

OBJECT_CATEGORY: dict[int, str] = {
    object_id: OFFICIAL_CATEGORY_ORDER[(object_id - 1) // 5]
    for object_id in OBJECT_IDS
}

OBJECT_SLUG: dict[int, str] = {
    object_id: OFFICIAL_CATEGORY_ORDER[(object_id - 1) // 5].replace(" ", "_")
    for object_id in OBJECT_IDS
}

SCENARIOS: tuple[str, ...] = ("NI", "NC", "NIC")


@dataclass(frozen=True)
class Resource:
    """An official CORe50 download resource."""

    name: str
    url: str
    size_bytes: int
    dest: str  # relative to data/raw/core50/downloads
    purpose: str
    required: bool = True
    kind: str = "archive"  # archive | text | pickle | repo
    sha256: str | None = None
    etag: str | None = None
    last_modified: str | None = None
    members: tuple[str, ...] = field(default_factory=tuple)


RESOURCES: tuple[Resource, ...] = (
    Resource(
        name="core50_128x128.zip",
        url=f"{IMAGE_HOST}core50_128x128.zip",
        size_bytes=5_892_103_007,
        dest="core50_128x128.zip",
        purpose=(
            "Official 128x128 RGB benchmark images (11 sessions x 50 objects "
            "x ~300 frames = 164,866 PNGs) — the dataset the official "
            "fetch_data_and_setup.sh downloads and the official data loader uses."
        ),
        kind="archive",
        etag='"03beef164c9d21:0"',
        last_modified="Mon, 10 May 2017 08:10:54 GMT",
    ),
    Resource(
        name="batches_filelists.zip",
        url=f"{PAGES_HOST}batches_filelists.zip",
        size_bytes=45_418_248,
        dest="batches_filelists.zip",
        purpose=(
            "Official plain-text batch filelists for the NI, NC and NIC "
            "scenarios (incremental and cumulative variants, 10 runs each) "
            "plus test_filelist.txt per run."
        ),
        kind="archive",
    ),
    Resource(
        name="dataset_dims.zip",
        url=f"{PAGES_HOST}dataset_dims.zip",
        size_bytes=2_578,
        dest="dataset_dims.zip",
        purpose="Official exact frame counts for every object/session pair.",
        kind="archive",
    ),
    Resource(
        name="bbox.zip",
        url=f"{PAGES_HOST}bbox.zip",
        size_bytes=1_390_289,
        dest="bbox.zip",
        purpose=(
            "Official bounding boxes per session/object (relative to the "
            "full-size 350x350 frames; kept as detection metadata)."
        ),
        kind="archive",
        required=False,
    ),
    Resource(
        name="core50_class_names.txt",
        url=f"{PAGES_HOST}core50_class_names.txt",
        size_bytes=183,
        dest="core50_class_names.txt",
        purpose="Official ordered list of the 10 CORe50 category names.",
        kind="text",
    ),
    Resource(
        name="labels2names.pkl",
        url=f"{PAGES_HOST}labels2names.pkl",
        size_bytes=6_940,
        dest="labels2names.pkl",
        purpose=(
            "Official label -> object-name mapping for every scenario and run."
        ),
        kind="pickle",
    ),
    Resource(
        name="labels.pkl",
        url=f"{PAGES_HOST}labels.pkl",
        size_bytes=228_464,
        dest="labels.pkl",
        purpose="Official per-run label vectors for the benchmark runs.",
        kind="pickle",
        required=False,
    ),
    Resource(
        name="LUP.pkl",
        url=f"{PAGES_HOST}LUP.pkl",
        size_bytes=24_947_181,
        dest="LUP.pkl",
        purpose="Official look-up table with the pattern order of each run.",
        kind="pickle",
        required=False,
    ),
    Resource(
        name="core50-master.zip",
        url="https://github.com/vlomonaco/core50/archive/refs/heads/master.zip",
        size_bytes=0,  # GitHub archive size varies; verified by HTTP HEAD
        dest="core50-master.zip",
        purpose=(
            "Snapshot of the official core50 repository (confs/ for the "
            "NI/NC/NIC experiment configurations and the official Python "
            "data loader) kept for reference and reproducibility."
        ),
        kind="repo",
        required=False,
    ),
)


def session_name(session_id: int) -> str:
    return f"s{session_id}"


def object_name(object_id: int) -> str:
    return f"o{object_id}"


def frame_name(session_id: int, object_id: int, frame_id: int) -> str:
    return f"C_{session_id:02d}_{object_id:02d}_{frame_id:03d}.png"
