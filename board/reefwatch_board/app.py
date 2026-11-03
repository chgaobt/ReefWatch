"""`python3 -m reefwatch_board`: the board program. See board/README.md.

    Unreal --CSG2, one-way, TCP 5556--> Receiver -> newest-frame slot -> Pipeline (backend)
                                                                     -> MapWorker (<= --map-hz)
    --reply: the same, plus a CSG2 response per request on the same socket (reply.py), so
    Unreal in its normal mode shows the board's masks and map; requests with STREAM_REPLIES
    (Unreal's real-time detection display) get replies only for the frames the board ran
    status line on stdout every second; optional status port 5557 for viewers on the PC;
    map artifacts + status.jsonl in --out (every --save-every s, on disconnect, on exit)
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import threading
import time

from . import BOARD_DIR, SIM_ROOT, __version__
from . import prepost as PP
from .backends import make_backend
from .framelog import FrameLog
from .pipeline import MapWorker, Pipeline, Saver
from .receiver import Receiver
from .reply import ReplyContext
from .slot import NewestSlot
from .status import DEFAULT_PORT as STATUS_PORT, StatusServer

DEFAULT_DEM = os.path.join(SIM_ROOT, "data", "healthmap", "dem.npz")
DEFAULT_GT = os.path.join(SIM_ROOT, "data", "healthmap", "gt_bleached.npz")


def parse_hw(spec):
    h, w = (int(v) for v in spec.lower().split("x"))
    return h, w


def make_parser():
    ap = argparse.ArgumentParser(
        prog="reefwatch_board",
        description="ReefWatch board receiver: one-way CSG2 in, DPU detection + health map on the board.")
    ap.add_argument("--host", default="0.0.0.0", help="listen address (default all interfaces)")
    ap.add_argument("--port", type=int, default=5556)
    ap.add_argument("--backend", choices=["none", "vart", "torch"], default="none",
                    help="none = pipe test (constant mask); vart = KV260 DPU; torch = PC A/B")
    ap.add_argument("--xmodel", default=None,
                    help="vart: compiled xmodel (default: board/models/coral_milestone_7c_512x768.xmodel)")
    ap.add_argument("--pynq-dpu-overlay", default=None, metavar="BITFILE",
                    help="vart: load this DPU-PYNQ overlay first (e.g. dpu.bit of DPU-PYNQ's Vitis AI "
                         "3.5 branch; needs root + the PYNQ venv). Off: the DPU is loaded with xmutil")
    ap.add_argument("--ckpt", default=None, help="torch: checkpoint (default: the milestone .pt)")
    ap.add_argument("--device", default=None, help="torch: cuda | cpu")
    ap.add_argument("--emulate-dpu", action="store_true",
                    help="torch: run the board's int8 pre/post-processing around the float model")
    ap.add_argument("--input-size", default="512x768", type=parse_hw,
                    help="torch --emulate-dpu: model input HxW (vart reads it from the xmodel)")
    ap.add_argument("--seven-to-health", choices=["health", "readme"], default="health")
    ap.add_argument("--warmup-res", default="640x360", help="WxH run once before READY ('' = none)")
    rg = ap.add_argument_group("reply mode (off by default: one-way, nothing is sent back)")
    rg.add_argument("--reply", action="store_true",
                    help="answer each request on the frame socket with a CSG2 (or CSEG v1) response: "
                         "mask, class fractions, stats, map metrics + minimap. For Unreal WITHOUT "
                         "-sendonly")
    rg.add_argument("--server-name", default=None,
                    help="name sent in SERVER_INFO (the Unreal HUD's mask-source label); default "
                         "'KV260 DPU' for --backend vart, else a name saying it is a rehearsal")
    rg.add_argument("--reply-timeout", type=float, default=2.0,
                    help="max seconds one response may take to send before the client is dropped "
                         "(2 = the Unreal client's own response deadline)")
    mg = ap.add_argument_group("health map")
    mg.add_argument("--dem", default=DEFAULT_DEM)
    mg.add_argument("--gt", default=DEFAULT_GT, help="bleached ground truth (optional; adds IoU)")
    mg.add_argument("--no-gt", action="store_true")
    mg.add_argument("--no-map", action="store_true")
    mg.add_argument("--map-hz", type=float, default=3.0, help="max map folds per second")
    mg.add_argument("--map-subsample", type=int, default=4)
    mg.add_argument("--map-range", type=float, default=None, help="max voting range, m (3.5)")
    mg.add_argument("--map-min-obs", type=int, default=None)
    mg.add_argument("--minimap-px", type=int, default=512)
    mg.add_argument("--minimap-hz", type=float, default=1.0,
                    help="minimap renders per second (for the status port; 0 = never)")
    og = ap.add_argument_group("output")
    og.add_argument("--out", default=None,
                    help="output dir (default board/out/board_<date>_<time>)")
    og.add_argument("--save-every", type=float, default=60.0)
    og.add_argument("--log-frames", action="store_true",
                    help="also save every mask and minimap with timestamps in <out>/frames/")
    og.add_argument("--status-port", type=int, default=STATUS_PORT)
    og.add_argument("--no-status", action="store_true", help="no status port")
    og.add_argument("--quiet", action="store_true", help="no per-second status line")
    og.add_argument("--exit-after", type=float, default=0.0,
                    help="stop after this many seconds (0 = run until Ctrl+C)")
    og.add_argument("--exit-on-disconnect", action="store_true",
                    help="stop when the client disconnects (scripted rehearsals)")
    return ap


def server_name(args):
    """What the Unreal HUD shows as the source of the masks (SERVER_INFO)."""
    if args.server_name:
        return args.server_name
    if args.backend == "vart":
        return "KV260 DPU"
    if args.backend == "torch":
        return "Board program (torch, emulated DPU)" if args.emulate_dpu else "Board program (torch)"
    return "Board program (no model)"


def build_runner(args, log):
    if args.no_map:
        log("[map] disabled (--no-map)")
        return None
    if not os.path.isfile(args.dem):
        log(f"[map] DISABLED - DEM not found: {args.dem} (pass --dem or --no-map)")
        return None
    from reefwatch_runtime.healthmap import MapRunner, engine as E
    gt = None if args.no_gt or not os.path.isfile(args.gt) else args.gt
    t0 = time.perf_counter()
    runner = MapRunner(
        dem=args.dem, gt=gt,
        max_range_m=args.map_range or E.DEFAULT_MAX_RANGE_M, subsample=args.map_subsample,
        min_obs=E.DEFAULT_MIN_OBS if args.map_min_obs is None else args.map_min_obs,
        minimap_px=args.minimap_px)
    g = runner.grid_info
    log(f"[map] DEM {args.dem}: {g['nx']}x{g['ny']} @ {g['cell_cm']:g} cm, {g['reef_cells']} reef "
        f"cells, GT {gt or 'none'} ({(time.perf_counter() - t0) * 1e3:.0f} ms); <= {args.map_hz:g} "
        f"folds/s, minimap <= {args.minimap_hz:g}/s")
    return runner


def make_server_info(args, backend):
    """SERVER_INFO block body for this board program."""
    from reefwatch_runtime import protocol as P
    d = backend.describe()
    extra = dict(program=f"reefwatch_board {__version__}", backend=d.get("backend"),
                 streaming=True, classes=int(backend.num_classes),
                 health_classes=P.class_table(PP.HEALTH_NAMES, PP.HEALTH_PALETTE_RGBA),
                 mask_default=4)
    nat = PP.native_classes(backend.num_classes)
    if nat is not None:                  # WANT_NATIVE_CLASSES is honoured
        extra["native_classes"] = P.class_table(*nat)
    for k in ("emulate_dpu", "device", "xmodel", "input_hw", "input"):
        if k in d:
            extra[k] = d[k] if k != "xmodel" else os.path.basename(str(d[k]))
    return P.pack_server_info(server_name(args), "board", **extra)


class App:
    def __init__(self, args, log=None):
        self.args = args
        self.log = log or (lambda m: print(m, flush=True))
        self.stop_event = threading.Event()
        self.t_start = time.time()
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.out_dir = args.out or os.path.join(BOARD_DIR, "out", f"board_{stamp}")
        self.backend = None
        self.receiver = self.pipeline = self.map_worker = self.saver = self.status = None
        self.frame_log = None
        self.reply_ctx = None
        self._status_fh = None
        self._prev = None

    # ------------------------------------------------------------------ setup
    def setup(self):
        a, log = self.args, self.log
        os.makedirs(self.out_dir, exist_ok=True)
        log(f"[board] reefwatch_board {__version__}, output -> {self.out_dir}")
        t0 = time.perf_counter()
        self.backend = make_backend(a.backend, xmodel=a.xmodel, ckpt=a.ckpt, device=a.device,
                                    emulate_dpu=a.emulate_dpu, input_hw=a.input_size, log=log,
                                    pynq_overlay=a.pynq_dpu_overlay)
        log(f"[board] backend {json.dumps(self.backend.describe())} "
            f"({(time.perf_counter() - t0) * 1e3:.0f} ms)")
        if a.warmup_res:
            ww, hh = (int(v) for v in a.warmup_res.lower().split("x"))
            t0 = time.perf_counter()
            self.backend.warmup(ww, hh)
            log(f"[board] warmed {ww}x{hh} in {(time.perf_counter() - t0) * 1e3:.0f} ms")
        to_health = PP.to_health_table(self.backend.num_classes, a.seven_to_health)

        if a.log_frames:
            self.frame_log = FrameLog(self.out_dir, log=log).start()
            log(f"[framelog] every mask and minimap -> {self.frame_log.dir}")
        runner = build_runner(a, log)
        if runner is not None:
            self.map_worker = MapWorker(runner, a.map_hz, a.minimap_hz, frame_log=self.frame_log,
                                        log=log).start()
            self.saver = Saver(runner, self.out_dir, a.save_every, extra=self._save_extra,
                               log=log).start()
        if a.reply:
            info = make_server_info(a, self.backend)
            self.reply_ctx = ReplyContext(to_health, self.map_worker, send_timeout=a.reply_timeout,
                                          log=log, server_info=info)
            log(f"[reply] ON: one CSG2/CSEG response per request on :{a.port} (mask in the 4 "
                f"health colours, or the 7 native ones for WANT_NATIVE_CLASSES "
                f"({'on' if self.reply_ctx.native_supported else 'not with this model'}), "
                f"MAP_METRICS, MINIMAP_PNG); STREAM_REPLIES requests: only frames "
                f"that ran are answered; send timeout {a.reply_timeout:g} s; name '{server_name(a)}'")
        frames = NewestSlot()
        self.pipeline = Pipeline(self.backend, frames, to_health, self.map_worker,
                                 frame_log=self.frame_log, log=log).start()
        self.receiver = Receiver(frames, a.host, a.port, log=log, on_disconnect=self._on_disconnect,
                                 reply_ctx=self.reply_ctx)
        self.receiver.bind()
        self.receiver.start()
        if not a.no_status:
            self.status = StatusServer(a.host, a.status_port, log=log)
            self.status.start()
        self._status_fh = open(os.path.join(self.out_dir, "status.jsonl"), "a", encoding="utf-8")
        log(f"READY on {a.host}:{self.receiver.port} "
            + ("(reply mode: a response per request on this socket)" if a.reply
               else "(one-way, nothing is sent back)")
            + ("" if self.status is None else f"; status port {self.status.port}"))

    def _save_extra(self):
        r, p, m = self.receiver, self.pipeline, self.map_worker
        board = dict(backend=self.backend.describe(),
                     frames_rx=r.frames_rx if r else 0,
                     frames_dropped=r.slot.dropped if r else 0,
                     frames_inferred=p.frames_done if p else 0,
                     map_folded=m.folded if m else 0,
                     map_superseded=m.dropped if m else 0,
                     uptime_s=round(time.time() - self.t_start, 1))
        if self.reply_ctx is not None:
            board["replies"] = dict(vars(self.reply_ctx.counters))
        return dict(board=board)

    def _on_disconnect(self, replaced):
        if self.saver is not None and not replaced:
            self.saver.request("client_disconnect")
        if self.args.exit_on_disconnect and not replaced:
            self.log("[board] client disconnected - exiting (--exit-on-disconnect)")
            self.stop_event.set()

    # ------------------------------------------------------------------ status
    def status_dict(self, dt):
        r, p, mw = self.receiver, self.pipeline, self.map_worker
        cur = dict(rx=r.frames_rx, bytes=r.bytes_rx, dropped=r.slot.dropped,
                   map_dropped=mw.dropped if mw else 0)
        rc = self.reply_ctx.counters if self.reply_ctx is not None else None
        if rc is not None:
            cur.update(tx=rc.sent, tx_bytes=rc.bytes_tx)
        prev = self._prev or cur
        self._prev = cur
        n, avg = p.window.take()
        s = dict(t=round(time.time(), 3), uptime_s=round(time.time() - self.t_start, 1),
                 backend=self.backend.name, client=r.client, frame_id=r.last_frame_id,
                 frames_rx=cur["rx"], rx_fps=round((cur["rx"] - prev["rx"]) / dt, 2),
                 rx_MBps=round((cur["bytes"] - prev["bytes"]) / dt / 1e6, 2),
                 dropped=cur["dropped"], dropped_s=cur["dropped"] - prev["dropped"],
                 frames_inferred=p.frames_done, pipe_fps=round(n / dt, 2),
                 infer_ms=round(avg["infer_ms"], 2), pre_ms=round(avg["pre_ms"], 2),
                 dpu_ms=round(avg["dpu_ms"], 2), post_ms=round(avg["post_ms"], 2),
                 pipe_ms=round(avg["pipe_ms"], 2), latency_ms=round(avg["latency_ms"], 2),
                 res=p.last["res"], class_frac=p.last["class_frac"], stats=p.last["stats"],
                 errors=p.errors)
        if rc is not None:
            s["reply"] = dict(sent=rc.sent, tx_fps=round((cur["tx"] - prev["tx"]) / dt, 2),
                              tx_MBps=round((cur["tx_bytes"] - prev["tx_bytes"]) / dt / 1e6, 2),
                              reused=rc.reused, failed=rc.failed, minimaps=rc.minimaps,
                              closed_slow=rc.closed_slow, streamed=rc.streamed,
                              skipped=rc.skipped)
        png, serial = None, 0
        if mw is not None:
            mn, mavg = mw.window.take()
            metrics, png, serial = mw.latest()
            s["map"] = dict(fps=round(mn / dt, 2), map_ms=round(mavg["map_ms"], 1),
                            folded=mw.folded, superseded=cur["map_dropped"], resets=mw.resets)
            if metrics is not None:
                s["map"].update(frames=metrics["frames"], coverage=metrics["coverage_fraction"],
                                surveyed_m2=metrics["surveyed_area_m2"],
                                bleached_m2=metrics["bleached_area_m2"],
                                class_area_m2=metrics["class_area_m2"])
                if "gt" in metrics:
                    s["map"]["gt_iou"] = metrics["gt"]["iou"]
                s["map_metrics"] = metrics
        return s, png, serial

    def status_line(self, s):
        m = s.get("map")
        mp = ""
        if m is not None:
            mp = f" | map {m['fps']:4.1f}/s {m['map_ms']:5.0f} ms"
            if "coverage" in m:
                mp += f" cov {m['coverage'] * 100:5.1f}% bleached {m['bleached_m2']:.2f} m2"
        dpu = (f" (pre {s['pre_ms']:.1f} dpu {s['dpu_ms']:.1f} post {s['post_ms']:.1f})"
               if s["backend"] != "none" else "")
        rp = s.get("reply")
        tx = "" if rp is None else (f" | tx {rp['tx_fps']:5.2f} fps {rp['tx_MBps']:4.1f} MB/s"
                                    + (" streamed" if rp["streamed"] else ""))
        return (f"[board] rx {s['rx_fps']:5.2f} fps {s['rx_MBps']:4.1f} MB/s drop {s['dropped_s']:2d} | "
                f"infer {s['infer_ms']:6.2f} ms{dpu} {s['pipe_fps']:5.2f} fps{mp}{tx} | "
                f"frames {s['frames_rx']} | {s['client']}")

    # ------------------------------------------------------------------ run
    def run(self):
        last = time.perf_counter()
        end = None if self.args.exit_after <= 0 else last + self.args.exit_after
        try:
            while not self.stop_event.wait(1.0):
                now = time.perf_counter()
                s, png, serial = self.status_dict(now - last)
                last = now
                if not self.args.quiet:
                    self.log(self.status_line(s))
                self._status_fh.write(json.dumps({k: v for k, v in s.items() if k != "map_metrics"}) + "\n")
                self._status_fh.flush()
                if self.status is not None:
                    self.status.publish(s, png, serial)
                if end is not None and now >= end:
                    self.log("[board] --exit-after reached")
                    break
        finally:
            self.shutdown()

    def shutdown(self):
        self.stop_event.set()
        for x in (self.receiver, self.status):
            if x is not None:
                x.stop()
        if self.pipeline is not None:
            self.pipeline.stop()
        if self.map_worker is not None:
            self.map_worker.stop()
        if self.saver is not None:
            self.saver.close()
        if self.frame_log is not None:
            self.frame_log.close()
        if self._status_fh is not None:
            self._status_fh.close()
            self._status_fh = None
        self.log(f"[board] stopped; outputs in {self.out_dir}")


def main(argv=None):
    args = make_parser().parse_args(argv)
    app = App(args)
    app.setup()

    def _signal(signum, _frame):
        app.log(f"\n[board] signal {signum} - shutting down")
        app.stop_event.set()

    for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        sig = getattr(signal, name, None)
        if sig is not None:
            try:
                signal.signal(sig, _signal)
            except (ValueError, OSError):
                pass
    app.run()
    return 0
