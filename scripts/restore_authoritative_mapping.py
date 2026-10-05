from pathlib import Path
import hashlib
import shutil

EXPECTED = "5d28556368a38f161160773a0210bdad6a9d73ba839d6e110066421104facad2"

ROOT = Path(__file__).resolve().parents[1]

SNAPSHOT = ROOT / "data" / "raw" / "core50" / "metadata" / "object_mapping.render.bin"
TARGET = ROOT / "data" / "raw" / "core50" / "metadata" / "object_mapping.json"

if not SNAPSHOT.exists():
    raise RuntimeError(f"Authoritative mapping snapshot missing: {SNAPSHOT}")

data = SNAPSHOT.read_bytes()
actual = hashlib.sha256(data).hexdigest()

if actual != EXPECTED:
    raise RuntimeError(
        f"Authoritative mapping snapshot SHA256 mismatch: "
        f"expected {EXPECTED}, got {actual}"
    )

TARGET.write_bytes(data)

written = hashlib.sha256(TARGET.read_bytes()).hexdigest()

if written != EXPECTED:
    raise RuntimeError(
        f"Restored authoritative mapping SHA256 mismatch: "
        f"expected {EXPECTED}, got {written}"
    )

print("=" * 70)
print("AUTHORITATIVE CORe50 MAPPING RESTORED")
print(f"SHA256: {written}")
print(f"PATH  : {TARGET}")
print("=" * 70)