"""Build the decoded uint8 cache that train/standalone_train.py reads (optional, ~10x faster loading).

Without it every training crop decodes a PNG and the loader, not the GPU, is the bottleneck.

    python data/coralscapes/build_cache.py                 # -> cache_v2/512/{train,validation,test}/
    python data/coralscapes/build_cache.py --out D:/cache_v2

Layout (what the trainer's SplitCache memory-maps):
    <out>/512/<split>/images.u8   (N, H, W, 3) uint8, raw, no header
    <out>/512/<split>/health.u8   (N, H, W)    uint8, 0..3 and 255
    <out>/512/<split>/masks.u8    (N, H, W)    uint8, CoralScapes fine ids
    <out>/512/<split>/index.json  ids, height, width, arrays{...}, complete
Pixels are byte-identical to the PNGs written by decode_v2.py. Point the trainer at it with
CORAL_V2_CACHE=<out> (the run_train scripts default to segmentation/cache_v2).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "train"))
import standalone_train as S  # noqa: E402  (dataset paths and split ids, same as training)


def build(split, out, res="512"):
    ids = S.read_split_ids(split)
    first = np.asarray(Image.open(S.png_dir(split, res, "images") / f"{ids[0]:04d}.png").convert("RGB"))
    h, w = first.shape[:2]
    d = out / res / split
    d.mkdir(parents=True, exist_ok=True)
    (d / "index.json").unlink(missing_ok=True)             # an old index must not survive a partial rebuild
    arrays = {"images": ("images.u8", (len(ids), h, w, 3)),
              "health": ("health.u8", (len(ids), h, w)),
              "masks": ("masks.u8", (len(ids), h, w))}
    mms = {k: np.memmap(d / f, dtype=np.uint8, mode="w+", shape=shape) for k, (f, shape) in arrays.items()}
    for r, i in enumerate(ids):
        for kind, mm in mms.items():
            p = S.png_dir(split, res, kind) / f"{i:04d}.png"
            im = Image.open(p)
            a = np.asarray(im.convert("RGB") if kind == "images" else im, np.uint8)
            if a.shape[:2] != (h, w):
                raise SystemExit(f"{p}: size {a.shape[:2]} != {(h, w)}")
            mm[r] = a
        if (r + 1) % 200 == 0 or r + 1 == len(ids):
            print(f"  {split}: {r + 1}/{len(ids)}", flush=True)
    for mm in mms.values():
        mm.flush()
    meta = {"ids": ids, "height": h, "width": w, "res": res, "complete": True,
            "arrays": {k: {"file": f, "dtype": "uint8", "shape": list(shape)} for k, (f, shape) in arrays.items()}}
    (d / "index.json").write_text(json.dumps(meta), encoding="utf-8")
    assert S.cache_ok(out, split, res, kinds=("images", "health", "masks")), f"cache check failed for {split}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(S.DEFAULT_CACHE))
    ap.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    args = ap.parse_args()
    out = Path(args.out)
    print(f"data : {S.DATA_ROOT}\ncache: {out}", flush=True)
    for split in args.splits:
        build(split, out)
    print("done")


if __name__ == "__main__":
    main()
