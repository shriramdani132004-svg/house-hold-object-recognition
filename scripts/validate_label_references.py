"""Phase-4 gate: exhaustive label-reference consistency check.

Audits EVERY filelist reference used by training and evaluation of the
NIC scenario: existence (optional), parse validity, session/object
resolution, official label rule (``label == object_id - 1``),
path/object agreement, reverse-mapping round-trip, category agreement,
identity coverage (all 50 present, none unexpected), and train/eval
session separation. Any single failure exits non-zero — nothing is
silently skipped.

Usage:
    python scripts/validate_label_references.py
    python scripts/validate_label_references.py --check-files
    python scripts/validate_label_references.py --json reports/model_diagnostics/label_integrity.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.continual.label_audit import audit_references  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-files",
        action="store_true",
        help="also stat every referenced image on disk (slower)",
    )
    parser.add_argument("--json", type=Path, help="write a JSON report to this path")
    args = parser.parse_args()

    result = audit_references(check_files_exist=args.check_files)

    print(f"scenario references audited: {result.total_references}")
    print(f"  train: {result.train_references}   eval: {result.eval_references}")
    print(f"identities observed: {len(result.identity_counts)}/50")
    print(f"sessions observed: {sorted(result.sessions_seen)}")
    print(f"files checked: {result.files_checked}")
    print(f"mapping version: {result.mapping_version}")
    print(f"mapping checksum: {result.mapping_checksum}")
    if result.failures:
        print(f"FAILURES: {len(result.failures)}")
        for failure in result.failures[:20]:
            print(f"  {failure.where}:{failure.line_number}: {failure.problem}")
        print("LABEL REFERENCE AUDIT: FAIL")
        exit_code = 1
    else:
        print("LABEL REFERENCE AUDIT: PASS (0 failures, 50/50 identities, rule enforced)")
        exit_code = 0

    if args.json:
        payload = {
            "passed": result.passed,
            "total_references": result.total_references,
            "train_references": result.train_references,
            "eval_references": result.eval_references,
            "identities_observed": len(result.identity_counts),
            "sessions_observed": sorted(result.sessions_seen),
            "files_checked": result.files_checked,
            "mapping_version": result.mapping_version,
            "mapping_checksum": result.mapping_checksum,
            "label_rule": "label == object_id - 1",
            "failures": [
                {"where": f.where, "line": f.line_number, "problem": f.problem}
                for f in result.failures
            ],
            "identity_table": {
                str(label): count
                for label, count in sorted(result.identity_counts.items())
            },
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
        print(f"report: {args.json}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
