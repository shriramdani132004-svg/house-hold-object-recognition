"""Phase-5/6 gate: representative image label proof + identity table.

Deterministically selects up to 3 real CORe50 images per identity
(spread across sessions) and prints for each:

    IMAGE PATH | SESSION | OFFICIAL IDENTITY | NUMERIC LABEL | REVERSE-MAPPED IDENTITY

The official identity comes from the repository's
``extras/core50_labels.txt`` via the image's PATH object token; the
reverse identity comes from the authoritative mapping via the numeric
label. They must be identical for every sample. No model prediction is
involved — this proves DATASET LABEL CORRECTNESS only.

Also prints the stable 0..49 identity table with the mapping
version/checksum (Phase 6).

Usage:
    python scripts/prove_labels.py
    python scripts/prove_labels.py --per-identity 3 --no-files
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.class_mapping import (  # noqa: E402
    MAPPING_VERSION,
    load_class_mapping,
    mapping_checksum,
)
from src.data.continual.label_audit import representative_samples  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-identity", type=int, default=3)
    parser.add_argument(
        "--no-files",
        action="store_true",
        help="skip the on-disk existence check for each sample",
    )
    args = parser.parse_args()

    mapping = load_class_mapping()
    checksum = mapping_checksum()
    print("=== authoritative identity table (stable order) ===")
    for label in range(50):
        print(f"{label:2d} -> {mapping.name_for(label)}")
    print(f"mapping version: {MAPPING_VERSION}")
    print(f"mapping checksum: {checksum}")

    samples = representative_samples(
        per_identity=args.per_identity,
        check_files_exist=not args.no_files,
        mapping=mapping,
    )
    identities = {int(s["label"]) for s in samples}
    print(f"\n=== representative label proof ({len(samples)} samples) ===")
    print(
        f"{'IMAGE PATH':<44} {'SES':>3} {'OFFICIAL':<18} {'LBL':>3} "
        f"{'REVERSE':<18} OK"
    )
    mismatches = 0
    missing_files = 0
    for sample in samples:
        ok = bool(sample["matches"])
        exists = sample["file_exists"]
        if not ok:
            mismatches += 1
        if exists is False:
            missing_files += 1
        print(
            f"{sample['relative_path']:<44} {sample['session']:>3} "
            f"{sample['official_identity']:<18} {sample['label']:>3} "
            f"{sample['reverse_identity']:<18} "
            f"{'YES' if ok else 'NO'}"
            + ("" if exists is not False else "  [MISSING FILE]")
        )

    print(f"\nidentities covered: {len(identities)}/50")
    problems = []
    if identities != set(range(50)):
        problems.append("identity coverage != 0..49")
    if mismatches:
        problems.append(f"{mismatches} official/reverse mismatches")
    if missing_files:
        problems.append(f"{missing_files} referenced images missing on disk")
    if problems:
        print("REPRESENTATIVE LABEL PROOF: FAIL")
        for problem in problems:
            print(f"  {problem}")
        return 1
    print("REPRESENTATIVE LABEL PROOF: PASS (50/50 identities, all reverse matches)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
