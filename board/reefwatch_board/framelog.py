"""Optional per-frame log (`--log-frames`): every mask the pipeline produces and every minimap
the map renders, with timestamps, for replaying a run frame by frame on the PC afterwards.

Writes in `<out>/frames/`, from its own thread behind a bounded queue: a slow SD card drops log
entries (counted in `dropped`), it never slows the pipeline.

    masks.bin       b"RWMASK01", then per inferred frame:
                    <I frame_id> <H height> <H width> <I blob_len> zlib(uint8 class ids, H x W)
                    (the model's own classes, before the 7 -> 4 health collapse)
    frames.jsonl    one line per inferred frame: frame_id, sim capture time, board receive /
                    done times, pre / dpu / post ms, pose, fov, class fractions
    minimaps.jsonl  one line per minimap: serial, time, the last folded frame_id, map metrics
    minimap/NNNNN.png
"""
from __future__ import annotations

import json
import os
import queue
import struct
import threading
import time
import zlib

import numpy as np

MASK_MAGIC = b"RWMASK01"
MASK_RECORD = struct.Struct("<IHHI")


class FrameLog:
    def __init__(self, out_dir, level=1, max_queue=64, log=print):
        self.dir = os.path.join(out_dir, "frames")
        os.makedirs(os.path.join(self.dir, "minimap"), exist_ok=True)
        self.level = int(level)
        self.log = log
        self.q = queue.Queue(maxsize=max_queue)
        self.dropped = 0
        self.masks = 0
        self.minimaps = 0
        self.thread = None

    def start(self):
        self.thread = threading.Thread(target=self._run, daemon=True, name="framelog")
        self.thread.start()
        return self

    def _put(self, item):
        try:
            self.q.put_nowait(item)
        except queue.Full:
            self.dropped += 1

    # called from the pipeline thread
    def frame(self, ids, meta):
        self._put(("mask", np.ascontiguousarray(ids, np.uint8), meta))

    # called from the map thread
    def minimap(self, png, meta):
        self._put(("minimap", png, meta))

    def _run(self):
        with open(os.path.join(self.dir, "masks.bin"), "wb") as fm, \
                open(os.path.join(self.dir, "frames.jsonl"), "w", encoding="utf-8") as fj, \
                open(os.path.join(self.dir, "minimaps.jsonl"), "w", encoding="utf-8") as fmm:
            fm.write(MASK_MAGIC)
            while True:
                item = self.q.get()
                if item is None:
                    break
                kind, data, meta = item
                try:
                    if kind == "mask":
                        blob = zlib.compress(data.tobytes(), self.level)
                        h, w = data.shape
                        fm.write(MASK_RECORD.pack(int(meta["frame_id"]), h, w, len(blob)))
                        fm.write(blob)
                        fj.write(json.dumps(meta) + "\n")
                        self.masks += 1
                    else:
                        self.minimaps += 1
                        meta = dict(meta, serial=self.minimaps)
                        with open(os.path.join(self.dir, "minimap", f"{self.minimaps:05d}.png"), "wb") as f:
                            f.write(data)
                        fmm.write(json.dumps(meta) + "\n")
                except Exception as exc:                        # never take the board down
                    self.log(f"[framelog] write FAILED: {type(exc).__name__}: {exc}")
            for f in (fm, fj, fmm):
                f.flush()

    def close(self, timeout=30.0):
        if self.thread is None:
            return
        self.q.put(None)                                        # blocking: flush what is queued
        self.thread.join(timeout)
        self.log(f"[framelog] {self.masks} masks, {self.minimaps} minimaps, {self.dropped} dropped"
                 f" -> {self.dir}")


def read_masks(path):
    """-> (frame_id, uint8 ids H x W) for each record of a masks.bin (PC side)."""
    with open(path, "rb") as f:
        if f.read(len(MASK_MAGIC)) != MASK_MAGIC:
            raise ValueError(f"{path}: not a masks.bin")
        while True:
            head = f.read(MASK_RECORD.size)
            if len(head) < MASK_RECORD.size:
                return
            fid, h, w, n = MASK_RECORD.unpack(head)
            blob = f.read(n)
            if len(blob) < n:
                return
            yield fid, np.frombuffer(zlib.decompress(blob), np.uint8).reshape(h, w)
