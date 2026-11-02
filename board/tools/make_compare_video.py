#!/usr/bin/env python
"""Compare a PC session with a board replay of the same recording, frame by frame.

Inputs: a session folder written by the PC detection server with `--record` (tools/play.bat:
out/sessions/<date_time>/ = recording.csgrec + pc/frames/) and the board's output folder from a
`--log-frames` run that was fed that recording (replay_stream.py). Frames are joined by
frame_id, the id Unreal stamped on each capture, so both sides are judged on the same pixels.

    python board/tools/make_compare_video.py out/sessions/20261027_101500 out/board_replay_x
        -> <board_out>/compare_video.mp4   input | PC float mask | board INT8 mask, both health maps
        -> <board_out>/compare.json        per-class agreement and BLEACHED IoU, PC vs board
        -> <board_out>/compare_frames.csv  frame_id, pixel agreement, BLEACHED IoU per frame

The video runs on the PC session's clock (when each frame reached the PC server). For every
recorded frame it shows the PC mask of that frame and the board's mask of the same frame; if
the board skipped it (it keeps only the newest frame when it falls behind), the board panel holds
its last mask with a lower frame_id and says so. Agreement is computed only over frames both
sides processed. The PC mask is the reference ("PC row, board column" in the confusion matrix).
Needs numpy, OpenCV and ffmpeg (on PATH, the WinGet install, or --ffmpeg). Layout and colours
come from make_run_video.py.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import zlib

import cv2
import numpy as np

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
BOARD_DIR = os.path.dirname(TOOLS_DIR)
sys.path.insert(0, BOARD_DIR)
sys.path.insert(0, TOOLS_DIR)
from reefwatch_board.framelog import MASK_MAGIC, MASK_RECORD  # noqa: E402
from reefwatch_board.prepost import SEVEN_TO_HEALTH  # noqa: E402
import make_run_video as RV  # noqa: E402  (layout helpers, SEVEN_* / HEALTH_* colours)

W, H = 1920, 1080
GOOD, BAD = (120, 220, 120), (90, 90, 235)                     # BGR
WINGET_FFMPEG = (r"C:\Users\nicho\AppData\Local\Microsoft\WinGet\Packages"
                 r"\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1-full_build\bin\ffmpeg.exe")


# --------------------------------------------------------------------------- inputs
def index_recording(path):
    """-> {frame_id: (t_s, file offset of the blob, blob length)} for a .csgrec."""
    idx = {}
    with open(path, "rb") as f:
        if f.read(8) != b"CSGREC01":
            raise SystemExit(f"{path}: not a .csgrec file")
        while True:
            head = f.read(RV.RECORD.size)
            if len(head) < RV.RECORD.size:
                break
            t, n = RV.RECORD.unpack(head)
            off = f.tell()
            blob = f.read(n)
            if len(blob) < n:
                break
            fid = RV.REQ_V2.unpack_from(zlib.decompressobj().decompress(blob, RV.REQ_V2.size), 0)[2]
            idx[fid] = (t, off, n)
    return idx


def index_masks(path):
    """-> {frame_id: (offset, h, w, blob length)} for a framelog masks.bin (read lazily: a full
    survey is thousands of 640x360 masks)."""
    idx = {}
    with open(path, "rb") as f:
        if f.read(len(MASK_MAGIC)) != MASK_MAGIC:
            raise SystemExit(f"{path}: not a masks.bin")
        while True:
            head = f.read(MASK_RECORD.size)
            if len(head) < MASK_RECORD.size:
                break
            fid, h, w, n = MASK_RECORD.unpack(head)
            off = f.tell()
            f.seek(n, os.SEEK_CUR)
            if f.tell() > os.fstat(f.fileno()).st_size:
                break
            idx[fid] = (off, h, w, n)
    return idx


def read_mask(f, entry):
    off, h, w, n = entry
    f.seek(off)
    return np.frombuffer(zlib.decompress(f.read(n)), np.uint8).reshape(h, w)


def load_jsonl(path):
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


class Side:
    """One frames/ folder (PC session or board run)."""

    def __init__(self, frames_dir):
        self.dir = frames_dir
        self.masks = index_masks(os.path.join(frames_dir, "masks.bin"))
        self.meta = {m["frame_id"]: m for m in load_jsonl(os.path.join(frames_dir, "frames.jsonl"))}
        self.minimaps = sorted(load_jsonl(os.path.join(frames_dir, "minimaps.jsonl")),
                               key=lambda m: (m["frame_id"], m["serial"]))
        self.fids = sorted(self.masks)
        self.f = open(os.path.join(frames_dir, "masks.bin"), "rb")

    def mask(self, fid):
        return read_mask(self.f, self.masks[fid])

    def newest_at_or_before(self, fid):
        """frame_id of the newest mask with id <= fid, or None."""
        i = np.searchsorted(self.fids, fid, side="right") - 1
        return self.fids[i] if i >= 0 else None

    def minimap_at(self, fid):
        """Newest minimap rendered after folding a frame with id <= fid -> (meta, path) or None."""
        best = None
        for m in self.minimaps:
            if m["frame_id"] > fid:
                break
            best = m
        if best is None:
            return None
        return best, os.path.join(self.dir, "minimap", f"{best['serial']:05d}.png")


# --------------------------------------------------------------------------- agreement
def iou_table(conf, names):
    """Per-class IoU and pixel shares from a confusion matrix (rows PC, columns board)."""
    tot = conf.sum()
    out = {}
    for k, name in enumerate(names):
        inter = float(conf[k, k])
        union = float(conf[k, :].sum() + conf[:, k].sum() - conf[k, k])
        out[name] = dict(iou=round(inter / union, 4) if union else None,
                         pc_fraction=round(float(conf[k, :].sum()) / tot, 4) if tot else 0.0,
                         board_fraction=round(float(conf[:, k].sum()) / tot, 4) if tot else 0.0)
    return out


def frame_scores(pc, bd, bleached):
    agree = float((pc == bd).mean())
    a, b = pc == bleached, bd == bleached
    union = int(np.count_nonzero(a | b))
    biou = (np.count_nonzero(a & b) / union) if union else None
    return agree, biou


def resize_like(ids, shape):
    if ids.shape == shape:
        return ids
    return cv2.resize(ids, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)


# --------------------------------------------------------------------------- render
def colour(ids, palette):
    lut = np.zeros((256, 3), np.uint8)
    lut[:len(palette)] = palette
    return lut[ids][:, :, ::-1]                                  # BGR


def put_map(canvas, entry, x, y, side, label):
    cv2.rectangle(canvas, (x, y), (x + side, y + side), (40, 40, 40), -1)
    if entry is not None:
        img = cv2.imread(entry[1], cv2.IMREAD_COLOR)
        if img is not None:
            canvas[y:y + side, x:x + side] = cv2.resize(img, (side, side), interpolation=cv2.INTER_NEAREST)
    else:
        RV.text(canvas, "no map yet", (x + side // 2 - 50, y + side // 2), 0.6, RV.DIM)
    cv2.rectangle(canvas, (x, y - 30), (x + side, y - 4), (0, 0, 0), -1)
    RV.text(canvas, label, (x + 6, y - 11), 0.55)
    if entry is not None:
        m = entry[0]
        RV.text(canvas, f"coverage {m['coverage_fraction'] * 100:.1f}%   surveyed {m['surveyed_area_m2']:.1f} m2"
                        f"   bleached {m['bleached_area_m2']:.2f} m2", (x, y + side + 22), 0.5, RV.DIM)
        RV.text(canvas, f"after frame {m['frame_id']}   ({m['frames']} folded)", (x, y + side + 44), 0.5, RV.DIM)


def render(rgb, pc_ids, bd_ids, bd_fid, fid, t_s, pc_meta, bd_meta, pc_map, bd_map, score, running,
           names, palette, args):
    canvas = np.full((H, W, 3), RV.BG, np.uint8)
    canvas[0:360, 0:640] = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    canvas[0:360, 640:1280] = colour(pc_ids, palette)
    if bd_ids is not None:
        canvas[0:360, 1280:1920] = colour(bd_ids, palette)
    else:
        RV.text(canvas, "board has no mask yet", (1450, 180), 0.7, RV.DIM)
    labels = ((0, "INPUT  (recorded from Unreal)"), (640, f"PC MODEL  ({args.pc_label})"),
              (1280, f"BOARD  ({args.board_label})"))
    for x, label in labels:
        cv2.rectangle(canvas, (x + 4, 6), (x + 12 + 9 * len(label), 30), (0, 0, 0), -1)
        RV.text(canvas, label, (x + 8, 24), 0.55)
    if bd_ids is not None and bd_fid != fid:
        cv2.rectangle(canvas, (1284, 330), (1916, 356), (0, 0, 0), -1)
        RV.text(canvas, f"board skipped frame {fid}: showing its frame {bd_fid}", (1290, 349), 0.5, BAD)

    put_map(canvas, pc_map, 30, 440, 520, "HEALTH MAP  (PC)")
    put_map(canvas, bd_map, 590, 440, 520, f"HEALTH MAP  ({args.board_label})")

    x0, y = 1160, 430
    mm, ss = divmod(int(t_s), 60)
    RV.text(canvas, f"session clock {mm:02d}:{ss:02d}", (x0, y), 0.8, RV.FG, 2)
    y += 34
    wt = pc_meta.get("world_time_s") if pc_meta else None
    RV.text(canvas, f"frame {fid}" + (f"   sim time {wt:.1f} s" if wt is not None else ""), (x0, y), 0.6)
    y += 28
    if pc_meta:
        RV.text(canvas, f"PC: infer {pc_meta.get('infer_ms', 0):.0f} ms   receive->reply "
                        f"{pc_meta.get('rx_to_done_ms', 0):.0f} ms", (x0, y), 0.5, RV.DIM)
    y += 24
    if bd_meta:
        RV.text(canvas, f"board: pre {bd_meta.get('pre_ms', 0):.0f}  DPU {bd_meta.get('dpu_ms', 0):.0f}  "
                        f"post {bd_meta.get('post_ms', 0):.0f} ms   receive->mask "
                        f"{bd_meta.get('rx_to_done_ms', 0):.0f} ms", (x0, y), 0.5, RV.DIM)
    y += 40
    if score is not None:
        agree, biou = score
        RV.text(canvas, f"this frame: pixel agreement {agree * 100:.1f}%", (x0, y), 0.65,
                GOOD if agree >= 0.9 else BAD)
        y += 28
        RV.text(canvas, "            BLEACHED IoU " + ("n/a (none on either side)" if biou is None
                                                       else f"{biou:.3f}"), (x0, y), 0.65)
    else:
        RV.text(canvas, "this frame: not processed by the board", (x0, y), 0.65, RV.DIM)
        y += 28
    y += 34
    if running["frames"]:
        RV.text(canvas, f"so far ({running['frames']} frames on both): agreement "
                        f"{running['agree'] / running['frames'] * 100:.1f}%", (x0, y), 0.55)
        y += 24
        RV.text(canvas, f"BLEACHED IoU (pooled pixels) {running['b_inter'] / max(running['b_union'], 1):.3f}",
                (x0, y), 0.55)
    y += 40
    pc_frac = np.bincount(pc_ids.ravel(), minlength=len(names))[:len(names)] / pc_ids.size
    bd_frac = (np.bincount(bd_ids.ravel(), minlength=len(names))[:len(names)] / bd_ids.size
               if bd_ids is not None else np.zeros(len(names)))
    RV.text(canvas, "class share   PC | board", (x0, y), 0.5, RV.DIM)
    y += 24
    for k, name in enumerate(names):
        yy = y + k * 26
        cv2.rectangle(canvas, (x0, yy - 12), (x0 + 14, yy + 2), RV.bgr(palette[k]), -1)
        RV.text(canvas, name, (x0 + 22, yy), 0.45)
        cv2.rectangle(canvas, (x0 + 120, yy - 13), (x0 + 120 + int(250 * pc_frac[k]), yy - 5),
                      RV.bgr(palette[k]), -1)
        cv2.rectangle(canvas, (x0 + 120, yy - 4), (x0 + 120 + int(250 * bd_frac[k]), yy + 4),
                      RV.bgr(palette[k]), -1)
        RV.text(canvas, f"{pc_frac[k] * 100:4.1f}% | {bd_frac[k] * 100:4.1f}%", (x0 + 390, yy), 0.45, RV.DIM)
    return canvas


# --------------------------------------------------------------------------- main
def find_ffmpeg(arg):
    if arg:
        return arg
    return shutil.which("ffmpeg") or (WINGET_FFMPEG if os.path.isfile(WINGET_FFMPEG) else "ffmpeg")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session", help="PC session folder (recording.csgrec + pc/frames/)")
    ap.add_argument("board_out", help="board output folder of a --log-frames run fed that recording")
    ap.add_argument("--out", default=None, help="video path (default <board_out>/compare_video.mp4)")
    ap.add_argument("--json", default=None, help="summary path (default <board_out>/compare.json)")
    ap.add_argument("--no-video", action="store_true", help="only the agreement numbers")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--crf", type=int, default=23)
    ap.add_argument("--classes", type=int, choices=(4, 7), default=None,
                    help="model classes (default: from session.json, else 7)")
    ap.add_argument("--pc-label", default="float, PyTorch")
    ap.add_argument("--board-label", default="INT8, DPU",
                    help='what the board side is, e.g. "PC rehearsal: torch --emulate-dpu"')
    ap.add_argument("--ffmpeg", default=None)
    args = ap.parse_args(argv)

    rec_path = os.path.join(args.session, "recording.csgrec")
    pc = Side(os.path.join(args.session, "pc", "frames"))
    bd = Side(os.path.join(args.board_out, "frames"))
    rec = index_recording(rec_path)
    info = load_json(os.path.join(args.session, "session.json")) or {}
    ncls = args.classes or int((info.get("model") or {}).get("num_classes", 7))
    if ncls == 7:
        names, palette = RV.SEVEN_NAMES, RV.SEVEN_RGB
        to_health = np.asarray(SEVEN_TO_HEALTH[(info.get("scheme") or {}).get("seven_to_health", "health")],
                               np.uint8)
    else:
        names, palette = RV.HEALTH_NAMES, np.array(RV.HEALTH_RGB, np.uint8)
        to_health = np.arange(4, dtype=np.uint8)
    bleached = names.index("BLEACHED")
    hb = RV.HEALTH_NAMES.index("BLEACHED")

    common = [fid for fid in pc.fids if fid in bd.masks]
    print(f"[compare] recorded {len(rec)}, PC masks {len(pc.fids)}, board masks {len(bd.fids)}, "
          f"on both {len(common)}", flush=True)
    if not common:
        raise SystemExit("[compare] no frame_id is on both sides: was the board fed this session's recording?")

    # ---- agreement over frames both sides processed
    K = max(ncls, 8)
    conf = np.zeros((K, K), np.int64)
    rows = []
    for fid in common:
        a = pc.mask(fid)
        b = resize_like(bd.mask(fid), a.shape)
        conf += np.bincount((a.astype(np.int64) * K + b).ravel(), minlength=K * K).reshape(K, K)
        agree, biou = frame_scores(a, b, bleached)
        rows.append((fid, agree, biou))
    conf = conf[:ncls, :ncls]
    hconf = np.zeros((4, 4), np.int64)
    np.add.at(hconf, (to_health[:, None].repeat(ncls, 1), to_health[None, :].repeat(ncls, 0)), conf)
    agrees = np.array([r[1] for r in rows])
    bious = [r[2] for r in rows if r[2] is not None]
    per_class = iou_table(conf, names)
    health = iou_table(hconf, RV.HEALTH_NAMES)
    summary = dict(
        session=os.path.abspath(args.session), board_out=os.path.abspath(args.board_out),
        pc_label=args.pc_label, board_label=args.board_label,
        frames=dict(recorded=len(rec), pc=len(pc.fids), board=len(bd.fids), both=len(common),
                    board_skipped=len([f for f in pc.fids if f not in bd.masks]),
                    board_only=len([f for f in bd.fids if f not in pc.masks])),
        classes=list(names), reference="pc",
        pixel_agreement=round(float(np.trace(conf)) / float(conf.sum()), 4),
        bleached_iou=per_class["BLEACHED"]["iou"],
        bleached_iou_frame_mean=round(float(np.mean(bious)), 4) if bious else None,
        frames_with_bleached=len(bious),
        per_frame_agreement=dict(mean=round(float(agrees.mean()), 4), min=round(float(agrees.min()), 4),
                                 p05=round(float(np.percentile(agrees, 5)), 4)),
        per_class=per_class,
        health=dict(classes=list(RV.HEALTH_NAMES),
                    pixel_agreement=round(float(np.trace(hconf)) / float(hconf.sum()), 4),
                    bleached_iou=health["BLEACHED"]["iou"], per_class=health,
                    confusion_pc_rows_board_cols=hconf.tolist()),
        confusion_pc_rows_board_cols=conf.tolist())
    maps = {}
    for key, path in (("pc", os.path.join(args.session, "pc", "map", "metrics.json")),
                      ("board", os.path.join(args.board_out, "metrics.json"))):
        m = load_json(path)
        if m:
            maps[key] = {k: m.get(k) for k in ("frames", "coverage_fraction", "surveyed_area_m2",
                                               "bleached_area_m2", "class_area_m2", "gt") if k in m}
    if maps:
        summary["final_maps"] = maps
    json_path = args.json or os.path.join(args.board_out, "compare.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    with open(os.path.join(os.path.dirname(json_path) or ".", "compare_frames.csv"), "w", newline="",
              encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(("frame_id", "pixel_agreement", "bleached_iou"))
        for fid, agree, biou in rows:
            wr.writerow((fid, f"{agree:.5f}", "" if biou is None else f"{biou:.5f}"))
    print(f"[compare] pixel agreement {summary['pixel_agreement']:.4f}, BLEACHED IoU "
          f"{summary['bleached_iou']}, health agreement {summary['health']['pixel_agreement']:.4f} "
          f"-> {json_path}", flush=True)
    if args.no_video:
        return 0

    # ---- video, on the PC session clock
    out = args.out or os.path.join(args.board_out, "compare_video.mp4")
    order = sorted((t, fid) for fid, (t, _, _) in rec.items() if fid in pc.masks)
    t0, t1 = order[0][0], order[-1][0]
    scores = {fid: (agree, biou) for fid, agree, biou in rows}
    enc = subprocess.Popen([find_ffmpeg(args.ffmpeg), "-y", "-loglevel", "error", "-f", "rawvideo",
                            "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", f"{args.fps:g}", "-i", "-",
                            "-c:v", "libx264", "-preset", "veryfast", "-crf", str(args.crf),
                            "-pix_fmt", "yuv420p", "-movflags", "+faststart", out], stdin=subprocess.PIPE)
    running = dict(frames=0, agree=0.0, b_inter=0, b_union=0)
    k, cur, img = 0, None, None
    n_ticks = int((t1 - t0) * args.fps) + 1
    with open(rec_path, "rb") as rf:
        for tick in range(n_ticks):
            now = t0 + tick / args.fps
            changed = False
            while k < len(order) and order[k][0] <= now:
                cur = order[k][1]
                k += 1
                changed = True
                if cur in scores:
                    a = pc.mask(cur)
                    b = resize_like(bd.mask(cur), a.shape)
                    running["frames"] += 1
                    running["agree"] += scores[cur][0]
                    running["b_inter"] += int(np.count_nonzero((a == bleached) & (b == bleached)))
                    running["b_union"] += int(np.count_nonzero((a == bleached) | (b == bleached)))
            if changed or img is None:
                pc_ids = pc.mask(cur)
                bd_fid = bd.newest_at_or_before(cur)
                bd_ids = resize_like(bd.mask(bd_fid), pc_ids.shape) if bd_fid is not None else None
                img = render(RV.read_rgb(rf, rec[cur][1:]), pc_ids, bd_ids, bd_fid, cur, now - t0,
                             pc.meta.get(cur), bd.meta.get(bd_fid), pc.minimap_at(cur), bd.minimap_at(cur),
                             scores.get(cur), running, names, palette, args)
            enc.stdin.write(img.tobytes())
            if tick % 300 == 0:
                print(f"[compare] video {tick}/{n_ticks} ticks", flush=True)
    enc.stdin.close()
    enc.wait()
    print(f"[compare] {n_ticks / args.fps:.0f} s of video -> {out}", flush=True)
    return enc.returncode


if __name__ == "__main__":
    sys.exit(main())
