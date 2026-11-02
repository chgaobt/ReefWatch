"""Reply mode (`--reply`, off by default): the board answers requests on the frame socket with
the responses docs/PROTOCOL.md defines, so Unreal in its NORMAL mode (no `-sendonly`) shows the
board's masks on HUD screen B and the board's health map, exactly as with the PC server.

Design (Unreal keeps one request in flight and gives up after 2 s):

* The receiver's reader thread registers each request in this connection's `Replier` (in
  arrival order) before it hands the frame to the newest-frame slot. Nothing else changes on
  the detection path: same slot, same pipeline thread, same map thread.
* When the pipeline has the class ids of a frame it calls `Replier.done()`: an append under a
  lock, never a socket call. The `Replier`'s own writer thread colours the mask (the 4 health
  colours, as the PC server's default `--mask-classes 4`), packs the response with
  `reefwatch_runtime.protocol` and sends it. So the response to request N goes out as soon as
  frame N's ids exist, about pre + DPU + post + colour + send after the request arrived.
* Answer policy `lockstep` (the one `--reply` uses; what today's Unreal client needs): exactly
  one response per request, in request order. A request whose frame was replaced in the
  newest-frame slot before the pipeline took it (only a client that pipelines requests can
  cause that) is answered with the mask of the next frame the pipeline finishes; counted in
  `reused`. The response's frame_id is always the id of the frame the MASK came from; with
  Unreal's one-in-flight loop that is the request's own id (PROTOCOL.md: "echo of the
  request"). An inference error is answered with an all-OTHER mask plus an ERROR block, so the
  client stays in step.
* Answer policy `processed` (streaming): only frames the pipeline actually ran are answered,
  each with its own frame_id; replaced frames get no response. It applies to every request
  that carries the STREAM_REPLIES flag (Unreal's "Real time" detection display: it sends at
  camera rate and never waits), whatever the context's default policy; requests without the
  flag keep `lockstep`. Writer thread, packing and map blocks are shared; only `_take` differs.
* FRAME_INFO (block 7) on every response to a request with STREAM_REPLIES or WANT_INFO: the
  mask's frame_id and capture world_time_s, receive -> inference start / ids ready, and this
  connection's received / inferred / skipped counts. SERVER_INFO (block 8: the board's name
  and backend, for the HUD's mask-source label) on the first such response of the connection.
  Requests with neither flag get exactly the bytes they got before.
* CSG2 responses carry MASK_RGBA, CLASS_FRAC, STATS, then MAP_METRICS (the latest folded map,
  once the map has applied the request's RESET_MAP generation) and MINIMAP_PNG (a minimap the
  map thread rendered that this client has not had yet, once the client has asked with
  WANT_MINIMAP; each minimap is sent once). The map folds on its own thread at `--map-hz`, so,
  as on the PC server, MAP_METRICS lags the mask. CSEG v1 requests get v1 responses.
* A slow or dead client never blocks the pipeline: in reply mode the socket has a timeout
  (`--reply-timeout`, default 2 s = the Unreal client's own response deadline) that bounds each
  send; a send that times out or fails closes the connection (the stream is no longer framed)
  and the reader thread ends. More than `max_pending` unanswered requests (a client that never
  reads) also closes it. The pipeline only ever appends to a list.

Header timings: infer_ms = the backend call (pre + DPU + post) of the mask's frame; map_ms = the
last map fold; total_ms = the answered request fully received -> its response packed.
"""
from __future__ import annotations

import collections
import json
import socket
import threading
import time

import numpy as np

from reefwatch_runtime import protocol as P

from . import prepost as PP

NO_MAP_ERROR = b"health map disabled on this board (--no-map or DEM missing)"


class Counters:
    """Totals over all connections (read by the status thread)."""

    def __init__(self):
        self.sent = 0           # responses written
        self.reused = 0         # ... of which answered with a later frame's mask (lockstep)
        self.failed = 0         # ... of which carried an inference ERROR
        self.minimaps = 0       # MINIMAP_PNG blocks sent
        self.bytes_tx = 0
        self.closed_slow = 0    # connections closed: a send timed out / the client is not reading
        self.streamed = 0       # responses sent under STREAM_REPLIES (the `processed` policy)
        self.skipped = 0        # streamed requests never answered: replaced in the slot


class ReplyContext:
    """What every connection's Replier shares: palette, map access, limits, counters."""

    POLICIES = ("lockstep", "processed")

    def __init__(self, to_health, map_worker=None, send_timeout=2.0, max_pending=64,
                 policy="lockstep", log=print, server_info=None):
        if policy not in self.POLICIES:
            raise ValueError(f"reply policy must be one of {self.POLICIES}, got {policy!r}")
        self.policy = policy
        lut = np.asarray(to_health, np.intp)
        pal = np.asarray(PP.HEALTH_PALETTE_RGBA, np.uint8)[lut]            # native id -> RGBA
        self.pal32 = np.ascontiguousarray(pal).view(np.uint32).reshape(-1)
        self.num_classes = len(lut)
        self.other_id = int(np.flatnonzero(lut == 3)[-1]) if (lut == 3).any() else len(lut) - 1
        self.map_worker = map_worker
        self.send_timeout = float(send_timeout)
        self.max_pending = int(max_pending)
        self.log = log
        self.counters = Counters()
        #: SERVER_INFO block body (bytes); a generic name if the app gave none
        self.server_info = server_info or P.pack_server_info("board program", "board")

    def colourize(self, ids):
        """Native ids (h, w) -> RGBA bytes view (one uint32 gather per pixel, as the PC server)."""
        return memoryview(self.pal32[ids].view(np.uint8).reshape(-1))


class _Entry:
    """One request as it arrived."""

    __slots__ = ("frame_id", "width", "height", "version", "flags", "reset_gen", "t_rx",
                 "world_time_s", "state", "result", "stream", "info")

    def __init__(self, frame_id, width, height, version, flags, reset_gen, t_rx, world_time_s=0.0):
        self.frame_id, self.width, self.height = int(frame_id), int(width), int(height)
        self.version, self.flags, self.reset_gen = int(version), int(flags), int(reset_gen)
        self.t_rx, self.world_time_s = t_rx, float(world_time_s)
        v2 = self.version == 2
        self.stream = v2 and bool(self.flags & P.FLAG_STREAM_REPLIES)   # `processed` policy
        self.info = v2 and P.wants_info(self.flags)                     # FRAME_INFO / SERVER_INFO
        self.state = "wait"          # wait | superseded | done
        self.result = None           # _Result once the pipeline has run this frame


class _Result:
    """What the pipeline produced for one frame. frame_id / world_time_s / t_rx are THAT
    frame's, whichever request the result ends up answering."""

    __slots__ = ("frame_id", "world_time_s", "t_rx", "t_done", "ids", "frac", "stats", "infer_ms",
                 "error")

    def __init__(self, entry, ids, frac, stats, infer_ms, error=None):
        self.frame_id, self.world_time_s, self.t_rx = entry.frame_id, entry.world_time_s, entry.t_rx
        self.t_done = time.perf_counter()
        self.ids, self.frac, self.stats = ids, frac, stats
        self.infer_ms, self.error = float(infer_ms), error


class Replier:
    """One per connection in reply mode: request bookkeeping + the writer thread."""

    def __init__(self, ctx, conn, addr):
        self.ctx, self.conn, self.addr = ctx, conn, addr
        self.cond = threading.Condition()
        self.pending = collections.deque()
        self.closed = False
        self.want_minimap = False        # sticky until a minimap has been sent
        self.minimap_serial = 0          # serial of the last minimap sent to this client
        self.warned_no_map = False
        self.info_sent = False           # SERVER_INFO: once per connection
        self.n_rx = 0                    # requests registered on this connection
        self.n_done = 0                  # ... whose frame the pipeline ran (or failed)
        self.n_skipped = 0               # ... replaced in the slot before the pipeline took them
        self.fps = 0.0                   # EWMA of 1000 / total_ms, sent as stats[3]
        self._metrics_obj = None
        self._metrics_bytes = None
        self.thread = threading.Thread(target=self._run, daemon=True, name=f"reply-{addr[1]}")
        self.thread.start()

    # ------------------------------------------------------------------ called by the reader
    def expect(self, frame_id, width, height, version, flags, reset_gen, t_rx, world_time_s=0.0):
        """Register a request (arrival order). -> entry, or None if the connection is closing."""
        over = False
        with self.cond:
            if self.closed:
                return None
            if len(self.pending) >= self.ctx.max_pending:
                over = True
            else:
                e = _Entry(frame_id, width, height, version, flags, reset_gen, t_rx, world_time_s)
                self.pending.append(e)
                self.n_rx += 1
                if flags & P.FLAG_WANT_MINIMAP:
                    self.want_minimap = True
        if over:
            self.ctx.counters.closed_slow += 1
            self._abort(f"{self.ctx.max_pending} requests unanswered (the client is not reading)")
            return None
        return e

    def superseded(self, entry):
        """The frame was replaced in the newest-frame slot; the pipeline will never run it."""
        with self.cond:
            if entry.state == "wait":
                entry.state = "superseded"
                self.n_skipped += 1
                self.cond.notify()

    # ------------------------------------------------------------------ called by the pipeline
    def done(self, entry, ids, frac, stats, infer_ms):
        with self.cond:
            entry.result = _Result(entry, ids, frac, stats, infer_ms)
            entry.state = "done"
            self.n_done += 1
            self.cond.notify()

    def failed(self, entry, message):
        ids = np.full((entry.height, entry.width), self.ctx.other_id, np.uint8)
        frac = np.array([0, 0, 0, 1], np.float32)
        with self.cond:
            entry.result = _Result(entry, ids, frac, None, 0.0,
                                   str(message).encode("utf-8", "replace")[:1024])
            entry.state = "done"
            self.n_done += 1
            self.cond.notify()

    # ------------------------------------------------------------------ lifecycle
    def close(self):
        with self.cond:
            self.closed = True
            self.pending.clear()
            self.cond.notify_all()

    def _abort(self, reason):
        if not self.closed:
            self.ctx.log(f"[reply] closing {self.addr[0]}:{self.addr[1]}: {reason}")
        self.close()
        try:
            self.conn.shutdown(socket.SHUT_RDWR)      # ends the reader thread too
        except OSError:
            pass

    # ------------------------------------------------------------------ writer thread
    def _processed(self, entry):
        """True if `entry` is answered under the `processed` policy (it carried STREAM_REPLIES,
        or that is the context's default); False = `lockstep`."""
        return entry.stream or self.ctx.policy == "processed"

    def _take(self):
        """Block until requests can be answered. -> [(entry, result, reused)] in request order,
        or None once the connection is closed."""
        with self.cond:
            while not self.closed:
                # "processed": a replaced frame gets no answer
                while (self.pending and self.pending[0].state == "superseded"
                       and self._processed(self.pending[0])):
                    self.pending.popleft()
                    self.ctx.counters.skipped += 1
                k = next((i for i, e in enumerate(self.pending) if e.state == "done"), None)
                if k is not None:
                    res = self.pending[k].result
                    out = []
                    for i in range(k + 1):
                        e = self.pending.popleft()
                        if i == k or not self._processed(e):
                            out.append((e, res, i < k))
                        else:
                            self.ctx.counters.skipped += 1
                    return out
                self.cond.wait(0.5)
            return None

    def _map_blocks(self, entry):
        blocks = []
        mw = self.ctx.map_worker
        if mw is None:
            if entry.flags & P.FLAG_POSE_VALID and not self.warned_no_map:
                self.warned_no_map = True
                blocks.append((P.BLOCK_ERROR, NO_MAP_ERROR))
            return blocks, 0.0
        if mw.applied_gen < entry.reset_gen:          # RESET_MAP not applied yet: old map
            return blocks, 0.0
        metrics, png, serial = mw.latest()
        map_ms = 0.0
        if metrics is not None:
            if metrics is not self._metrics_obj:
                self._metrics_obj = metrics
                self._metrics_bytes = json.dumps(metrics, separators=(",", ":")).encode("utf-8")
            blocks.append((P.BLOCK_MAP_METRICS, self._metrics_bytes))
            map_ms = float(metrics.get("map_ms", 0.0))
        if png and self.want_minimap and serial != self.minimap_serial:
            self.minimap_serial = serial
            self.want_minimap = False
            blocks.append((P.BLOCK_MINIMAP_PNG, png))
            self.ctx.counters.minimaps += 1
        return blocks, map_ms

    def _info_blocks(self, entry, result, reused):
        """FRAME_INFO (+ SERVER_INFO once per connection) for a request that asked for it."""
        if not entry.info:
            return []
        with self.cond:
            n_rx, n_done, n_skip = self.n_rx, self.n_done, self.n_skipped
        done_ms = (result.t_done - result.t_rx) * 1e3
        blocks = [(P.BLOCK_FRAME_INFO, P.pack_frame_info(
            result.frame_id, result.world_time_s, max(0.0, done_ms - result.infer_ms), done_ms,
            n_rx, n_done, n_skip, streamed=entry.stream, reused=reused))]
        if not self.info_sent:
            self.info_sent = True
            blocks.append((P.BLOCK_SERVER_INFO, self.ctx.server_info))
        return blocks

    def _pack(self, entry, result, reused=False):
        """-> ([header, *pieces], total_ms). `entry` = the request answered (version, flags,
        RESET_MAP generation, receive time); `result` = the frame the mask came from."""
        ids = result.ids
        h, w = ids.shape
        mask = self.ctx.colourize(ids)
        frac_b = np.asarray(result.frac, "<f4").tobytes()
        s3 = result.stats if result.stats is not None else (0.0, 0.0, 0.0)
        stats_b = np.array([s3[0], s3[1], s3[2], self.fps], "<f4").tobytes()
        if entry.version == 1:
            total_ms = (time.perf_counter() - entry.t_rx) * 1e3
            head = P.pack_response_header_v1(result.frame_id, w, h, result.infer_ms, total_ms)
            return [head, mask, frac_b + stats_b], total_ms
        blocks = [(P.BLOCK_MASK_RGBA, mask), (P.BLOCK_CLASS_FRAC, frac_b), (P.BLOCK_STATS, stats_b)]
        extra, map_ms = self._map_blocks(entry)
        blocks += extra
        if result.error is not None:
            blocks.append((P.BLOCK_ERROR, result.error))
        blocks += self._info_blocks(entry, result, reused)
        total_ms = (time.perf_counter() - entry.t_rx) * 1e3
        head, parts = P.build_response_v2(result.frame_id, w, h, result.infer_ms, map_ms,
                                          total_ms, blocks)
        return [head] + parts, total_ms

    def _run(self):
        c = self.ctx.counters
        while True:
            batch = self._take()
            if batch is None:
                return
            for entry, result, reused in batch:
                try:
                    parts, total_ms = self._pack(entry, result, reused)
                except Exception as exc:                    # never take the board down
                    self.ctx.log(f"[reply] packing frame {entry.frame_id} FAILED: "
                                 f"{type(exc).__name__}: {exc}")
                    self._abort("a response could not be packed")
                    return
                try:
                    for p in parts:                         # bounded by the socket timeout
                        self.conn.sendall(p)
                except socket.timeout:
                    c.closed_slow += 1
                    self._abort(f"a response send took longer than {self.ctx.send_timeout:g} s")
                    return
                except OSError as exc:
                    self._abort(f"send failed ({type(exc).__name__}: {exc})")
                    return
                c.sent += 1
                c.bytes_tx += sum(memoryview(p).nbytes for p in parts)
                if reused:
                    c.reused += 1
                if entry.stream:
                    c.streamed += 1
                if result.error is not None:
                    c.failed += 1
                inst = 1000.0 / max(total_ms, 1e-3)
                self.fps = inst if self.fps <= 0 else 0.9 * self.fps + 0.1 * inst
