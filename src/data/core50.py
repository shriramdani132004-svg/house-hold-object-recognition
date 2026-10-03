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

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CORE50_ROOT = PROJECT_ROOT / "data" / "raw" / "core50"
DOWNLOAD_DIR = CORE50_ROOT / "downloads"
DATASET_DIR = CORE50_ROOT / "dataset"
FILELIST_DIR = CORE50_ROOT / "filelists"
METADATA_DIR = CORE50_ROOT / "metadata"
OBJECT_MAPPING_PATH = METADATA_DIR / "object_mapping.json"
IMAGES_ROOT = DATASET_DIR / "core50_128x128"

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


# ---------------------------------------------------------------------------
# Semantic validation of metadata/object_mapping.json
# ---------------------------------------------------------------------------

_MAPPING_TOP_KEYS: tuple[str, ...] = (
    "source_object_names",
    "source_categories",
    "category_order",
    "objects",
    "categories",
)


def _relative(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def validate_core50_object_mapping(
    mapping_path: Path | None = None,
    *,
    images_root: Path | None = None,
    expected_identities: int = 50,
    expected_categories: int = 10,
) -> dict[str, Any]:
    """Semantically validate the official CORe50 object mapping.

    ``metadata/object_mapping.json`` nests the50 object identities under
    the ``objects`` key (plus ``categories`` / ``category_order``), so the
    file's top level deliberately has five keys — a naive
    ``len(mapping) == 50`` check is a false positive. This validator
    resolves the real schema and verifies that:

    * the five documented top-level keys are present;
    * ``category_order`` matches the official category order and
      ``categories`` holds exactly the10 official names;
    * the50 identities each have a unique ``object_id`` (1..50), a unique
      official ``name``, and ``directory == "o<object_id>"``;
    * each identity's ``category`` matches the official5-objects-per-
      category table and its label convention is ``object_id - 1``;
    * per-session image counts sum to the total and the claimed sessions
      match the actual ``s<session>/o<id>`` directories on disk
      (stat-only; no image is opened or counted).

    Returns a dict with ``identity_count``, ``category_count``,
    ``identities`` (sorted by ``object_id``), ``categories`` and ``errors``
    (empty when fully valid).
    """
    path = Path(mapping_path) if mapping_path else OBJECT_MAPPING_PATH
    root = Path(images_root) if images_root else IMAGES_ROOT
    errors: list[str] = []
    identities: list[dict[str, Any]] = []
    categories: list[str] = []

    def result() -> dict[str, Any]:
        return {
            "mapping_path": _relative(path),
            "images_root": _relative(root),
            "identity_count": len(identities),
            "category_count": len(categories),
            "identities": identities,
            "categories": categories,
            "errors": errors,
        }

    if not path.is_file():
        errors.append(f"object mapping not found: {_relative(path)}")
        return result()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"object mapping unreadable: {exc}")
        return result()
    if not isinstance(raw, dict):
        errors.append(
            f"object mapping must be a JSON object, got {type(raw).__name__}"
        )
        return result()

    for key in _MAPPING_TOP_KEYS:
        if key not in raw:
            errors.append(
                f"missing top-level key {key!r} (nested schema expected: "
                "identities live under 'objects')"
            )

    order = raw.get("category_order")
    if order != list(OFFICIAL_CATEGORY_ORDER):
        errors.append("category_order does not match the official category order")

    raw_categories = raw.get("categories")
    if not isinstance(raw_categories, list):
        errors.append("'categories' must be a list")
    else:
        categories = [str(value) for value in raw_categories]
        if len(set(categories)) != len(categories):
            errors.append("categories contains duplicates")
        if set(categories) != set(OFFICIAL_CATEGORY_ORDER):
            missing = sorted(set(OFFICIAL_CATEGORY_ORDER) - set(categories))
            extra = sorted(set(categories) - set(OFFICIAL_CATEGORY_ORDER))
            errors.append(f"categories mismatch (missing={missing}, extra={extra})")
        if len(categories) != expected_categories:
            errors.append(
                f"expected {expected_categories} categories, got {len(categories)}"
            )

    objects = raw.get("objects")
    if not isinstance(objects, list):
        errors.append("'objects' must be the list of50 identities")
        return result()

    seen_ids: set[int] = set()
    seen_names: set[str] = set()
    seen_dirs: set[str] = set()
    session_names = [session_name(sid) for sid in SESSION_IDS]
    root_ok = root.is_dir()
    if not root_ok:
        errors.append(f"images root not found: {_relative(root)}")

    for position, entry in enumerate(objects):
        if not isinstance(entry, dict):
            errors.append(f"objects[{position}] is not a JSON object")
            continue
        object_id = entry.get("object_id")
        name = entry.get("name")
        directory = entry.get("directory")
        category = entry.get("category")
        if not isinstance(object_id, int) or object_id not in OBJECT_IDS:
            errors.append(f"objects[{position}]: invalid object_id {object_id!r}")
            continue
        label = object_id - 1
        where = f"object {object_id} ({name!r})" if isinstance(name, str) else f"object {object_id}"
        if object_id in seen_ids:
            errors.append(f"duplicate object_id {object_id}")
        seen_ids.add(object_id)
        if not isinstance(name, str) or not name:
            errors.append(f"object {object_id}: missing or empty name")
        elif name in seen_names:
            errors.append(f"duplicate object name {name!r}")
        else:
            seen_names.add(name)
        expected_dir = object_name(object_id)
        if directory != expected_dir:
            errors.append(
                f"object {object_id}: directory {directory!r} != {expected_dir!r}"
            )
        elif directory in seen_dirs:
            errors.append(f"duplicate directory {directory!r}")
        else:
            seen_dirs.add(directory)
        if category != OBJECT_CATEGORY[object_id]:
            errors.append(
                f"{where}: category {category!r} != official "
                f"{OBJECT_CATEGORY[object_id]!r}"
            )
        elif isinstance(category, str) and categories and category not in categories:
            errors.append(f"{where}: category {category!r} not in 'categories'")

        sessions = entry.get("available_in_sessions")
        per_session = entry.get("images_per_session")
        images_total = entry.get("images")
        if not isinstance(sessions, list) or not sessions:
            errors.append(f"{where}: 'available_in_sessions' must be a non-empty list")
            sessions = []
        else:
            bad_sessions = [s for s in sessions if s not in session_names]
            if bad_sessions:
                errors.append(f"{where}: unknown sessions {bad_sessions}")
            if len(set(sessions)) != len(sessions):
                errors.append(f"{where}: duplicate sessions in 'available_in_sessions'")
        if not isinstance(images_total, int) or images_total <= 0:
            errors.append(f"{where}: 'images' must be a positive integer")
            images_total = None
        if not isinstance(per_session, dict):
            errors.append(f"{where}: 'images_per_session' must be an object")
        else:
            if set(per_session) != set(sessions):
                errors.append(
                    f"{where}: 'images_per_session' keys != 'available_in_sessions'"
                )
            if all(isinstance(v, int) for v in per_session.values()):
                if images_total is not None and sum(per_session.values()) != images_total:
                    errors.append(
                        f"{where}: per-session counts sum to "
                        f"{sum(per_session.values())}, 'images' says {images_total}"
                    )
        if root_ok and isinstance(directory, str):
            missing = [
                s for s in sessions
                if isinstance(s, str) and not (root / s / directory).is_dir()
            ]
            if missing:
                errors.append(f"{where}: claimed sessions missing on disk: {missing}")
            unclaimed = [
                session_name(sid)
                for sid in SESSION_IDS
                if (root / session_name(sid) / directory).is_dir()
                and session_name(sid) not in sessions
            ]
            if unclaimed:
                errors.append(
                    f"{where}: session directories present but unclaimed: {unclaimed}"
                )

        identities.append(
            {
                "object_id": object_id,
                "label": label,
                "name": name if isinstance(name, str) else None,
                "directory": directory if isinstance(directory, str) else None,
                "category": category if isinstance(category, str) else None,
            }
        )

    identities.sort(key=lambda item: item["object_id"])
    if len(objects) != expected_identities:
        errors.append(
            f"expected {expected_identities} identities, got {len(objects)}"
        )
    else:
        labels = sorted(item["label"] for item in identities)
        if labels != list(range(expected_identities)):
            errors.append(
                "labels are not exactly object_id - 1 over 0.."
                f"{expected_identities - 1}"
            )
    return result()
