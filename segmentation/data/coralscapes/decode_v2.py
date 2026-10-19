"""Decode the Coralscapes V2 parquet shards into the on-disk layout `coral.data` expects.

Output (mirrors CoralCapstone/data/processed/coralscapes exactly):

    data/coralscapes_v2/processed/<split>/{images,masks,health}/NNNN.png      # 1024x2048
    data/coralscapes_v2/processed/<split>_512/{images,masks,health}/NNNN.png  # 512x1024
    data/coralscapes_v2/metadata/coralscapes_v2_index.csv                     # split,idx,orig_path,site,w,h,c0..c95

`masks`  = raw V2 fine label ids (uint8, 0..95, 0 = unlabeled)
`health` = 4-class superclass ids from configs/health_label_mapping_v2.yaml (255 = IGNORE)

Indices are assigned deterministically by (shard order, row-group order, row order), so a
re-run reproduces the same NNNN.png -> orig_path correspondence. Row groups are decoded in
parallel processes (PNG compression is the bottleneck).

    python data/coralscapes/decode_v2.py [--splits train validation test] [--workers 6]   (from segmentation/)
"""
import argparse
import csv
import io
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml
from PIL import Image

def _env_path(var, default):
    """Env override (CORAL_UNREAL / CORAL_DATA_V2 / CORAL_MAPPING_V2); unset -> default."""
    v = os.environ.get(var)
    return Path(v) if v else Path(default)


ROOT = _env_path("CORAL_UNREAL", Path(__file__).resolve().parents[2])   # .../segmentation
DATA_V2 = _env_path("CORAL_DATA_V2", ROOT / "data/coralscapes_v2")
HF = DATA_V2 / "hf/data"
PROC = DATA_V2 / "processed"
META = DATA_V2 / "metadata"
MAPPING = _env_path("CORAL_MAPPING_V2", Path(__file__).resolve().parent / "health_label_mapping_v2.yaml")
NUM_FINE = 95
SPLITS = ["train", "validation", "test"]
_LUT = None


def health_lut():
    global _LUT
    if _LUT is None:
        m = yaml.safe_load(MAPPING.read_text(encoding="utf-8"))
        ids = {k: v["id"] for k, v in m["superclasses"].items()}
        lut = np.full(256, 255, dtype=np.uint8)
        for fid, sup in m["mapping"].items():
            lut[int(fid)] = ids[sup]
        _LUT = lut
    return _LUT


def _decode(cell):
    if isinstance(cell, dict):
        b, p = cell.get("bytes"), cell.get("path")
        return (Image.open(io.BytesIO(b)) if b is not None else Image.open(p)), p
    return Image.open(io.BytesIO(cell)), None


def do_group(task):
    """Decode one parquet row group -> PNGs; return the index rows."""
    shard, rg, base, split = task
    lut = health_lut()
    pf = pq.ParquetFile(shard)
    tbl = pf.read_row_group(rg, columns=["image", "label"])
    rows = []
    for k, r in enumerate(tbl.to_pylist()):
        n = base + k
        im, ipath = _decode(r["image"])
        lb, _ = _decode(r["label"])
        im = im.convert("RGB")
        if lb.mode != "L":
            lb = lb.convert("L")
        arr = np.array(lb, dtype=np.uint8)
        assert arr.max() <= NUM_FINE, f"{ipath}: fine id {arr.max()} > {NUM_FINE}"
        W, H = im.size
        name = f"{n:04d}.png"
        d1, d2 = PROC / split, PROC / f"{split}_512"
        im.save(d1 / "images" / name, compress_level=3)
        Image.fromarray(arr).save(d1 / "masks" / name)
        Image.fromarray(lut[arr]).save(d1 / "health" / name)
        im.resize((W // 2, H // 2), Image.BILINEAR).save(d2 / "images" / name, compress_level=3)
        m512 = np.array(Image.fromarray(arr).resize((W // 2, H // 2), Image.NEAREST))
        Image.fromarray(m512).save(d2 / "masks" / name)
        Image.fromarray(lut[m512]).save(d2 / "health" / name)
        cnt = np.bincount(arr.ravel(), minlength=NUM_FINE + 1)
        base_name = Path(ipath).name if ipath else name
        rows.append(dict(split=split, idx=n, orig_path=base_name,
                         site=base_name.split("_")[0], width=W, height=H, label_mode=lb.mode,
                         **{f"c{i}": int(cnt[i]) for i in range(NUM_FINE + 1)}))
    pf.close()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=SPLITS)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()

    META.mkdir(parents=True, exist_ok=True)
    tasks, counts = [], {}
    for split in a.splits:
        shards = sorted(HF.glob(f"{split}-*.parquet"))
        assert shards, f"no shards for {split} in {HF}"
        for sub in (split, f"{split}_512"):
            for k in ("images", "masks", "health"):
                (PROC / sub / k).mkdir(parents=True, exist_ok=True)
        base = 0
        for sh in shards:
            md = pq.ParquetFile(sh)
            for rg in range(md.metadata.num_row_groups):
                tasks.append((str(sh), rg, base, split))
                base += md.metadata.row_group(rg).num_rows
            md.close()
        counts[split] = base
        print(f"{split}: {len(shards)} shards, {base} rows", flush=True)

    t0, rows, done = time.time(), [], 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for r in ex.map(do_group, tasks):
            rows.extend(r)
            done += len(r)
            print(f"  {done} images ({time.time()-t0:.0f}s)", flush=True)

    rows.sort(key=lambda r: (SPLITS.index(r["split"]), r["idx"]))
    idx_path = META / "coralscapes_v2_index.csv"
    with open(idx_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("index ->", idx_path, len(rows), "rows;", counts, f"in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
