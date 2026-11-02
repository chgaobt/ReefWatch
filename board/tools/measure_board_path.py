#!/usr/bin/env python
"""PC-side check of the board's own pre/post-processing against the PC server. Needs torch.

For each input size (the xmodel builds: 512x768, 512x1024) and each 640x360 frame (the bundled
Unreal frame + crops of CoralScapes TEST frames, if present):

  server : runtime TorchBackend - float input, reflect-pad to the model multiple (768x512),
           bilinear output, argmax: what the PC detection server sends
  board  : board TorchBackend(emulate_dpu) - int8 LUT input (fix_point 5), reflect-pad to the
           xmodel input size, full-resolution output, crop + argmax: the board's exact code
           with the DPU replaced by a float forward

Reports pixel agreement (7-class and after the 7->4 collapse) and BLEACHED IoU. What it cannot
measure: the INT8 arithmetic inside the DPU (see the quantizer's INT8-vs-float numbers in
board/models/TENSORS_*.md).

    python board/tools/measure_board_path.py [--frames 20] [--sizes 512x768,512x1024] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

BOARD_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIM_ROOT = os.path.dirname(BOARD_DIR)
sys.path.insert(0, BOARD_DIR)
sys.path.insert(0, os.path.join(SIM_ROOT, "runtime"))
from reefwatch_board import prepost as PP  # noqa: E402
from reefwatch_board.backends import TorchBackend  # noqa: E402

TEST_IMAGES = r"C:\Users\nicho\Projects\CoralUnreal\artifacts_quant\vitis\test_images_v2.npy"
LUT4 = PP.to_health_table(7)


def agree(a, b):
    return round(float((a == b).mean()), 5)


def bleached_iou(a, b, idx):
    x, y = a == idx, b == idx
    u = (x | y).sum()
    return None if u == 0 else round(float((x & y).sum() / u), 4)


def frames(n):
    from PIL import Image
    out = [("auv_frame_640x360", np.asarray(Image.open(os.path.join(
        SIM_ROOT, "runtime", "tests", "data", "auv_frame_640x360.png")).convert("RGB")))]
    if os.path.isfile(TEST_IMAGES) and n > 1:
        imgs = np.load(TEST_IMAGES, mmap_mode="r")
        for i in np.linspace(0, len(imgs) - 1, n - 1).astype(int):
            out.append((f"coralscapes_test_{i}_crop", np.ascontiguousarray(imgs[i, 76:436, 192:832])))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--sizes", default="512x768,512x1024")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    from reefwatch_runtime import config
    ckpt = a.ckpt or str(config.DEFAULT_CKPT)
    fr = frames(a.frames)
    res = dict(ckpt=ckpt, n_frames=len(fr), sizes={})
    server = None
    refs = None
    for spec in a.sizes.split(","):
        hw = tuple(int(v) for v in spec.split("x"))
        board = TorchBackend(ckpt, emulate_dpu=True, input_hw=hw)
        if server is None:
            server = board.rt                              # same weights, the PC server's path
            refs = [server.predict_with_stats(rgb)[0] for _, rgb in fr]
        bidx = board.bleached_idx
        rows = []
        for (name, rgb), ref in zip(fr, refs):
            ids, _ = board.predict_with_stats(rgb)
            rows.append(dict(frame=name, agree_7c=agree(ids, ref), agree_4c=agree(LUT4[ids], LUT4[ref]),
                             bleached_iou=bleached_iou(ids, ref, bidx),
                             bleached_frac_server=round(float((ref == bidx).mean()), 4),
                             bleached_frac_board=round(float((ids == bidx).mean()), 4)))
        summ = dict(mean_7c=round(float(np.mean([r["agree_7c"] for r in rows])), 5),
                    min_7c=round(float(np.min([r["agree_7c"] for r in rows])), 5),
                    mean_4c=round(float(np.mean([r["agree_4c"] for r in rows])), 5),
                    unreal_frame=rows[0])
        res["sizes"][spec] = dict(summary=summ, frames=rows)
        print(f"{spec}: {json.dumps(summ)}", flush=True)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
