#!/usr/bin/env python
"""Rebuild what the board saw, as a video: the recorded input frames, the board's DPU masks on
top of them, and the health map growing, on the board's own clock.

Inputs: the .csgrec recording that was replayed (record_stream.py) and the board's output
folder from a `--log-frames` run (frames/masks.bin, frames.jsonl, minimaps.jsonl, minimap/).

    python board/tools/make_run_video.py out/rec/survey_full.csgrec out/board_full_run
        -> out/board_full_run/run_video.mp4   (1280x720, real time, 10 fps)

Every video tick shows the newest mask the board had finished by then and the newest minimap
it had rendered, so the video runs at the board's pace (frames it skipped are simply not shown).
Needs numpy, OpenCV and an ffmpeg on PATH (or --ffmpeg).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import subprocess
import sys
import zlib

import cv2
import numpy as np

BOARD_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BOARD_DIR)
from reefwatch_board.framelog import read_masks  # noqa: E402

REQ_V2 = struct.Struct("<4s7I8f")
RECORD = struct.Struct("<dI")

# = reefwatch_runtime.model SEVEN_NAMES / SEVEN_HEX (copied: model.py imports torch)
SEVEN_NAMES = ("BLEACHED", "DEAD", "ALIVE", "ROCK", "OCEAN", "SAND", "OTHER")
SEVEN_HEX = ("#f4ec75", "#ea7979", "#67d88d", "#6b4f3a", "#68b9ef", "#f38400", "#e591d6")
SEVEN_RGB = np.array([[int(h[i:i + 2], 16) for i in (1, 3, 5)] for h in SEVEN_HEX], np.uint8)
# = healthmap engine PALETTE (the map's colours)
HEALTH_NAMES = ("LIVE", "BLEACHED", "DEGRADED", "OTHER")
HEALTH_RGB = ((0, 200, 80), (255, 255, 255), (200, 60, 40), (60, 60, 200))

W, H = 1280, 720
BG = (24, 22, 20)                                               # BGR
FG = (235, 235, 235)
DIM = (150, 150, 150)


def index_recording(path):
    """frame_id -> (file offset of the blob, blob length) for a .csgrec."""
    idx = {}
    with open(path, "rb") as f:
        f.read(8)
        while True:
            head = f.read(RECORD.size)
            if len(head) < RECORD.size:
                break
            _, n = RECORD.unpack(head)
            off = f.tell()
            blob = f.read(n)
            if len(blob) < n:
                break
            fid = REQ_V2.unpack_from(zlib.decompressobj().decompress(blob, REQ_V2.size), 0)[2]
            idx[fid] = (off, n)
    return idx


def read_rgb(f, entry):
    off, n = entry
    f.seek(off)
    msg = zlib.decompress(f.read(n))
    h = REQ_V2.unpack_from(msg, 0)
    w, hh = h[3], h[4]
    return np.frombuffer(msg, np.uint8, count=w * hh * 3, offset=REQ_V2.size).reshape(hh, w, 3)


def text(img, s, xy, scale=0.5, color=FG, thick=1):
    cv2.putText(img, s, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def bgr(rgb):
    return (int(rgb[2]), int(rgb[1]), int(rgb[0]))


def render(rgb, ids, meta, minimap, mm_meta, clock_s, args):
    canvas = np.full((H, W, 3), BG, np.uint8)
    frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    canvas[0:360, 0:640] = frame
    over = (frame * (1 - args.alpha) + SEVEN_RGB[ids][:, :, ::-1] * args.alpha).astype(np.uint8)
    canvas[0:360, 640:1280] = over
    for x, label in ((8, "INPUT  (what Unreal sent the board)"), (648, "DPU MASK  (what the board saw)")):
        cv2.rectangle(canvas, (x - 4, 6), (x + 300, 28), (0, 0, 0), -1)
        text(canvas, label, (x, 22), 0.5)

    if minimap is not None:
        canvas[360:720, 0:360] = cv2.resize(minimap, (360, 360), interpolation=cv2.INTER_NEAREST)
    cv2.rectangle(canvas, (4, 366), (238, 388), (0, 0, 0), -1)
    text(canvas, "HEALTH MAP  (on the board)", (8, 382), 0.5)

    x0, y = 384, 392
    mm, ss = divmod(int(clock_s), 60)
    text(canvas, f"board clock {mm:02d}:{ss:02d}", (x0, y), 0.8, FG, 2)
    text(canvas, f"frame {meta['frame_id']}   sim time {meta['world_time_s']:.1f} s", (x0 + 300, y), 0.55)
    y += 30
    text(canvas, f"pre {meta['pre_ms']:.0f} ms   DPU {meta['dpu_ms']:.0f} ms   post {meta['post_ms']:.0f} ms"
                 f"   receive->mask {meta['rx_to_done_ms']:.0f} ms", (x0, y), 0.55, DIM)
    y += 30
    frac = np.bincount(ids.ravel(), minlength=7)[:7] / ids.size
    for k, name in enumerate(SEVEN_NAMES):
        yy = y + k * 22
        cv2.rectangle(canvas, (x0, yy - 12), (x0 + 14, yy + 2), bgr(SEVEN_RGB[k]), -1)
        text(canvas, name, (x0 + 22, yy), 0.45)
        cv2.rectangle(canvas, (x0 + 110, yy - 11), (x0 + 110 + int(300 * frac[k]), yy + 1),
                      bgr(SEVEN_RGB[k]), -1)
        text(canvas, f"{frac[k] * 100:4.1f}%", (x0 + 420, yy), 0.45, DIM)

    mx = x0 + 500
    yy = y
    text(canvas, "map", (mx, yy), 0.55)
    if mm_meta is not None:
        rows = (f"coverage {mm_meta['coverage_fraction'] * 100:.1f}%",
                f"surveyed {mm_meta['surveyed_area_m2']:.1f} m2",
                f"bleached {mm_meta['bleached_area_m2']:.2f} m2",
                f"frames folded {mm_meta['frames']}")
        for k, r in enumerate(rows):
            text(canvas, r, (mx, yy + 24 + k * 22), 0.5, DIM if k != 2 else FG)
    yy += 24 + 4 * 22 + 12
    for k, name in enumerate(HEALTH_NAMES):
        cv2.rectangle(canvas, (mx, yy + k * 20 - 11), (mx + 12, yy + k * 20 + 1), bgr(HEALTH_RGB[k]), -1)
        text(canvas, name, (mx + 20, yy + k * 20), 0.42, DIM)
    return canvas


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recording", help=".csgrec that was replayed to the board")
    ap.add_argument("board_out", help="board output folder of a --log-frames run")
    ap.add_argument("--out", default=None, help="video path (default <board_out>/run_video.mp4)")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--alpha", type=float, default=0.45, help="mask opacity")
    ap.add_argument("--crf", type=int, default=23)
    ap.add_argument("--ffmpeg", default=shutil.which("ffmpeg") or "ffmpeg")
    args = ap.parse_args(argv)
    fdir = os.path.join(args.board_out, "frames")
    out = args.out or os.path.join(args.board_out, "run_video.mp4")

    metas = {}
    with open(os.path.join(fdir, "frames.jsonl"), encoding="utf-8") as f:
        for line in f:
            m = json.loads(line)
            metas[m["frame_id"]] = m
    mms = []
    with open(os.path.join(fdir, "minimaps.jsonl"), encoding="utf-8") as f:
        for line in f:
            mms.append(json.loads(line))
    masks = {fid: ids for fid, ids in read_masks(os.path.join(fdir, "masks.bin"))}
    order = sorted((m["t_done"], fid) for fid, m in metas.items() if fid in masks)
    rec = index_recording(args.recording)
    missing = sum(1 for _, fid in order if fid not in rec)
    print(f"[video] {len(order)} board masks, {len(mms)} minimaps, {len(rec)} recorded frames"
          f" ({missing} masks without a recorded frame)", flush=True)
    order = [(t, fid) for t, fid in order if fid in rec]
    t0, t1 = order[0][0], order[-1][0]

    enc = subprocess.Popen([args.ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                            "-s", f"{W}x{H}", "-r", f"{args.fps:g}", "-i", "-", "-c:v", "libx264",
                            "-preset", "veryfast", "-crf", str(args.crf), "-pix_fmt", "yuv420p",
                            "-movflags", "+faststart", out], stdin=subprocess.PIPE)
    k = j = 0
    cur = None
    mm_img = mm_meta = None
    n_ticks = int((t1 - t0) * args.fps) + 1
    with open(args.recording, "rb") as rf:
        for tick in range(n_ticks):
            now = t0 + tick / args.fps
            changed = False
            while k < len(order) and order[k][0] <= now:
                cur = order[k][1]
                k += 1
                changed = True
            while j < len(mms) and mms[j]["t"] <= now:
                mm_meta = mms[j]
                png = os.path.join(fdir, "minimap", f"{mm_meta['serial']:05d}.png")
                mm_img = cv2.imread(png, cv2.IMREAD_COLOR)
                j += 1
                changed = True
            if changed or tick == 0:
                img = render(read_rgb(rf, rec[cur]), masks[cur], metas[cur], mm_img, mm_meta, now - t0, args)
            enc.stdin.write(img.tobytes())
            if tick % 300 == 0:
                print(f"[video] {tick}/{n_ticks} ticks", flush=True)
    enc.stdin.close()
    enc.wait()
    print(f"[video] {n_ticks / args.fps:.0f} s of video -> {out}", flush=True)
    return enc.returncode


if __name__ == "__main__":
    sys.exit(main())
