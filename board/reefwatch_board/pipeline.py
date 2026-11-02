"""The board's worker threads: inference pipeline, health-map folds, and saving.

    receiver --(newest-frame slot)--> Pipeline --(newest-item slot, <= map_hz)--> MapWorker
                                         |                                           |
                                    class fractions,                     MAP_METRICS dict, minimap PNG
                                    infer timings                        (read by the status thread)
                                                         Saver: map artifacts every N s, on
                                                         disconnect and on exit
Every hand-off is newest-wins, so no stage ever waits on a slower one.
"""
from __future__ import annotations

import threading
import time
import traceback

import numpy as np

from .slot import NewestSlot


class Window:
    """Per-second accumulator: counts and sums, reset when the status thread reads it."""

    def __init__(self, *keys):
        self.lock = threading.Lock()
        self.keys = keys
        self._reset()

    def _reset(self):
        self.n = 0
        self.sums = {k: 0.0 for k in self.keys}

    def add(self, **vals):
        with self.lock:
            self.n += 1
            for k, v in vals.items():
                self.sums[k] += v

    def take(self):
        with self.lock:
            n, sums = self.n, self.sums
            self._reset()
        return n, {k: (v / n if n else 0.0) for k, v in sums.items()}


class _Worker:
    name = "worker"

    def __init__(self, log=print):
        self.log = log
        self.stop_event = threading.Event()
        self.thread = None
        self._errors = set()

    def start(self):
        self.thread = threading.Thread(target=self._run, daemon=True, name=self.name)
        self.thread.start()
        return self

    def stop(self, timeout=5.0):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout)

    def _error_once(self, what, exc):
        key = (what, type(exc).__name__, str(exc)[:200])
        if key not in self._errors:
            self._errors.add(key)
            self.log(f"[{self.name}] {what} FAILED: {type(exc).__name__}: {exc}")
            traceback.print_exc()

    def _run(self):                                         # pragma: no cover - interface
        raise NotImplementedError


class MapItem:
    __slots__ = ("ids", "pose", "fov_h_deg", "frame_id", "reset_gen")

    def __init__(self, ids, pose, fov_h_deg, frame_id, reset_gen):
        self.ids, self.pose, self.fov_h_deg = ids, pose, fov_h_deg
        self.frame_id, self.reset_gen = frame_id, reset_gen


class Pipeline(_Worker):
    """slot -> backend -> health ids -> class fractions (+ a MapItem for posed frames)."""

    name = "pipeline"

    def __init__(self, backend, slot, to_health, map_worker=None, frame_log=None, log=print):
        super().__init__(log)
        self.backend, self.slot, self.map_worker = backend, slot, map_worker
        self.frame_log = frame_log
        self.to_health = np.asarray(to_health, np.uint8)
        self.identity = bool(np.array_equal(self.to_health, np.arange(len(self.to_health))))
        self.window = Window("infer_ms", "pre_ms", "dpu_ms", "post_ms", "pipe_ms", "latency_ms")
        self.frames_done = 0
        self.errors = 0
        self.last = dict(frame_id=0, class_frac=[0.0] * 4, stats=None, res="-")

    def _run(self):
        nc = len(self.to_health)
        while not self.stop_event.is_set():
            f = self.slot.get(timeout=0.25)
            if f is None:
                continue
            t0 = time.perf_counter()
            try:
                ids, stats = self.backend.predict_with_stats(f.rgb)
            except Exception as exc:                          # keep serving; report once
                self.errors += 1
                self._error_once("inference", exc)
                if f.reply is not None:                       # --reply: the client stays in step
                    f.reply[0].failed(f.reply[1], f"inference failed on the board: "
                                                  f"{type(exc).__name__}: {exc}")
                continue
            t1 = time.perf_counter()
            counts = np.bincount(ids.reshape(-1), minlength=nc)[:nc]
            if not self.identity:
                counts = np.bincount(self.to_health, weights=counts, minlength=4)
            frac = (counts[:4] / float(ids.size)).astype(np.float32)
            if f.reply is not None:          # --reply: hand the ids to the connection's writer
                f.reply[0].done(f.reply[1], ids, frac, stats, (t1 - t0) * 1e3)
            if self.map_worker is not None and f.pose is not None:
                health = ids if self.identity else np.take(self.to_health, ids)
                self.map_worker.post(MapItem(health, f.pose, f.fov_h_deg, f.frame_id, f.reset_gen))
            t2 = time.perf_counter()
            tm = self.backend.timings
            self.frames_done += 1
            self.last = dict(frame_id=int(f.frame_id), class_frac=[round(float(v), 4) for v in frac],
                             stats=None if stats is None else [round(float(v), 4) for v in stats],
                             res=f"{f.width}x{f.height}")
            self.window.add(infer_ms=(t1 - t0) * 1e3, pre_ms=tm.get("pre", 0.0),
                            dpu_ms=tm.get("dpu", 0.0), post_ms=tm.get("post", 0.0),
                            pipe_ms=(t2 - t0) * 1e3, latency_ms=(t2 - f.t_rx) * 1e3)
            if self.frame_log is not None:
                self.frame_log.frame(ids, dict(
                    frame_id=int(f.frame_id), world_time_s=f.world_time_s, t_done=time.time(),
                    rx_to_done_ms=round((t2 - f.t_rx) * 1e3, 2), infer_ms=round((t1 - t0) * 1e3, 2),
                    pre_ms=round(tm.get("pre", 0.0), 2), dpu_ms=round(tm.get("dpu", 0.0), 2),
                    post_ms=round(tm.get("post", 0.0), 2), pose=f.pose, fov_h_deg=f.fov_h_deg,
                    reset_gen=f.reset_gen, class_frac=self.last["class_frac"]))


class MapWorker(_Worker):
    """Folds the newest posed frame into the health map, at most `max_hz` times a second.

    RESET_MAP arrives as `reset_gen` on the items: the first item of a newer generation resets
    the map before it is folded; a late item of an older generation is discarded.
    The minimap PNG is rendered here (after a fold, at most `minimap_hz`), never on the
    pipeline or the status thread.
    """

    name = "map"

    def __init__(self, runner, max_hz=3.0, minimap_hz=1.0, frame_log=None, log=print):
        super().__init__(log)
        self.runner = runner
        self.frame_log = frame_log
        self.max_hz = float(max_hz)
        self.minimap_hz = float(minimap_hz)
        self.slot = NewestSlot()
        self.lock = threading.Lock()
        self.window = Window("map_ms")
        self.applied_gen = 0
        self.folded = 0
        self.stale = 0
        self.resets = 0
        self.metrics = None
        self.minimap_png = None
        self.minimap_serial = 0
        self._last_minimap = -1e9

    def post(self, item):
        self.slot.put(item)

    @property
    def dropped(self):
        return self.slot.dropped

    def _run(self):
        period = 1.0 / self.max_hz if self.max_hz > 0 else 0.0
        next_t = 0.0
        while not self.stop_event.is_set():
            wait = next_t - time.perf_counter()
            if wait > 0:
                self.stop_event.wait(wait)
                continue
            item = self.slot.get(timeout=0.25)
            if item is None:
                continue
            t0 = time.perf_counter()
            next_t = t0 + period
            if item.reset_gen > self.applied_gen:
                self.runner.reset()
                self.applied_gen = item.reset_gen
                self.resets += 1
                with self.lock:
                    self.metrics = None
                self.log(f"[map] reset (generation {item.reset_gen})")
            elif item.reset_gen < self.applied_gen:
                self.stale += 1
                continue
            try:
                self.runner.update(item.ids, item.pose, fov_h_deg=item.fov_h_deg)
                png = None
                if self.minimap_hz > 0 and t0 - self._last_minimap >= 0.95 / self.minimap_hz:
                    self._last_minimap = t0
                    png = self.runner.minimap_png()
                m = self.runner.metrics(frame_id=item.frame_id)
            except Exception as exc:
                self._error_once("map fold", exc)
                continue
            map_ms = (time.perf_counter() - t0) * 1e3
            m["map_ms"] = round(map_ms, 3)
            if png is not None and self.frame_log is not None:
                self.frame_log.minimap(png, dict(
                    t=time.time(), frame_id=int(item.frame_id), map_ms=round(map_ms, 1),
                    **{k: m.get(k) for k in ("frames", "coverage_fraction", "surveyed_area_m2",
                                             "bleached_area_m2", "class_area_m2")}))
            with self.lock:
                self.metrics = m
                if png is not None:
                    self.minimap_png = png
                    self.minimap_serial += 1
            self.folded += 1
            self.window.add(map_ms=map_ms)

    def latest(self):
        """-> (metrics dict | None, minimap png | None, minimap serial)."""
        with self.lock:
            return self.metrics, self.minimap_png, self.minimap_serial


class Saver(_Worker):
    """`MapRunner.save_artifacts` every `every_s`, on request (disconnect) and at `close()`."""

    name = "saver"

    def __init__(self, runner, out_dir, every_s=60.0, extra=None, log=print):
        super().__init__(log)
        self.runner, self.out_dir = runner, out_dir
        self.every_s = float(every_s)
        self.extra = extra or (lambda: {})
        self.event = threading.Event()
        self.reason = None
        self.saves = 0

    def request(self, reason):
        self.reason = reason
        self.event.set()

    def save(self, reason):
        if self.runner.hm.frames == 0:
            return None
        try:
            extra = dict(reason=reason)
            extra.update(self.extra())
            paths = self.runner.save_artifacts(self.out_dir, extra=extra)
            self.saves += 1
            self.log(f"[save] {reason}: {self.runner.hm.frames} map frames -> {self.out_dir}")
            return paths
        except Exception as exc:
            self._error_once(f"save ({reason})", exc)
            return None

    def _run(self):
        next_t = time.perf_counter() + self.every_s
        while not self.stop_event.is_set():
            fired = self.event.wait(max(0.05, min(0.5, next_t - time.perf_counter())))
            if self.stop_event.is_set():
                return
            if fired:
                self.event.clear()
                self.save(self.reason or "requested")
            elif self.every_s > 0 and time.perf_counter() >= next_t:
                next_t = time.perf_counter() + self.every_s
                self.save("autosave")

    def close(self):
        self.stop()
        return self.save("shutdown")
