"""Download the `josauder/coralscapesV2` *segmentation* config only (data/ + the json metadata).

Skips `instances/` (6.7 GB fish instance masks) and `sequences/` (206 GB video context) --
neither is used by the capstone. Total ~6.8 GB.

    python data/coralscapes/download_v2.py      (from the segmentation/ folder)
"""
import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download


def _env_path(var, default):
    """Env override (CORAL_UNREAL / CORAL_DATA_V2); unset -> default."""
    v = os.environ.get(var)
    return Path(v) if v else Path(default)


ROOT = _env_path("CORAL_UNREAL", Path(__file__).resolve().parents[2])   # .../segmentation
DEST = _env_path("CORAL_DATA_V2", ROOT / "data/coralscapes_v2") / "hf"


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    p = snapshot_download(
        repo_id="josauder/coralscapesV2",
        repo_type="dataset",
        local_dir=str(DEST),
        allow_patterns=["data/*.parquet", "*.json", "README.md"],
        max_workers=4,
    )
    print("snapshot ->", p)
    tot = sum(f.stat().st_size for f in Path(p).rglob("*") if f.is_file())
    print(f"bytes on disk: {tot/2**30:.2f} GiB")
    for f in sorted(Path(p).rglob("*.json")):
        print("  ", f.name, f.stat().st_size)


if __name__ == "__main__":
    sys.exit(main())
