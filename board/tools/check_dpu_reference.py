#!/usr/bin/env python3
"""First DPU check on the board: is the DPU's output bit-exact with the PC reference? numpy + vart only.

Feeds the stored int8 input of the bundled Unreal frame (tests/data/dpu_ref_auv_frame_512x768.npz)
to the 512x768 xmodel through the board's own VartBackend runner and compares the DPU's int8 logits
with the reference computed on the PC (VART CPU reference runner on the same compiled subgraph,
bit-exact against vai_q's golden dump and vai_q test mode on 20 frames).

    python3 tools/check_dpu_reference.py [--xmodel models/coral_milestone_7c_512x768.xmodel]

Expect "max |d| 0". A difference is a fact about the DPU vs vai_q's model of it: report the
numbers (max/mean |d|, argmax agreement); a few 1-LSB differences would point at rounding in the
DAPPM concat (3 inputs re-quantised fp 5 -> 4), large ones at a wrong overlay/fingerprint or input.
"""
import argparse
import os
import sys
import time

import numpy as np

BOARD_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BOARD_DIR)
from reefwatch_board.backends import VartBackend  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--xmodel", default=os.path.join(BOARD_DIR, "models", "coral_milestone_7c_512x768.xmodel"))
    ap.add_argument("--ref", default=os.path.join(BOARD_DIR, "tests", "data", "dpu_ref_auv_frame_512x768.npz"))
    ap.add_argument("--repeat", type=int, default=20, help="timed runs after the check")
    a = ap.parse_args(argv)
    ref = np.load(a.ref)
    be = VartBackend(a.xmodel)
    print("backend", be.describe())
    x = ref["int8_input"]
    if x.shape != be.inp.buf.shape:
        raise SystemExit("reference input %s does not fit the xmodel input %s (use the 512x768 build)"
                         % (x.shape, be.inp.buf.shape))
    be.inp.buf[...] = x
    be.runner.wait(be.runner.execute_async([be.inp.buf], [be.out]))
    y, yr = be.out.astype(np.int16), ref["logits"][0].astype(np.int16)[None]
    d = np.abs(y - yr)
    h, w = ref["ids"].shape
    ids = be.out[0, :h, :w].argmax(-1)
    print("DPU vs PC reference: max |d| %d  mean |d| %.5f  values differing %.6f  argmax agreement %.6f"
          % (d.max(), d.mean(), (d > 0).mean(), (ids == ref["ids"]).mean()))
    ts = []
    for _ in range(a.repeat):
        t0 = time.perf_counter()
        be.runner.wait(be.runner.execute_async([be.inp.buf], [be.out]))
        ts.append((time.perf_counter() - t0) * 1e3)
    if ts:
        print("execute_async+wait: median %.1f ms, min %.1f ms over %d runs" % (np.median(ts), min(ts), len(ts)))
    return 0 if d.max() == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
