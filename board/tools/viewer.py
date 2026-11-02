#!/usr/bin/env python
"""PC viewer for the board's status port (TCP 5557): prints the status, saves the latest
minimap PNG, appends the status to a JSON-lines log, and shows a live window when matplotlib
is installed.

    python board/tools/viewer.py --host 192.168.1.2
    python board/tools/viewer.py --host 127.0.0.1 --no-plot --seconds 30
"""
from __future__ import annotations

import argparse
import io
import json
import os
import socket
import sys
import time

BOARD_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BOARD_DIR)
from reefwatch_board.status import DEFAULT_PORT, read_status  # noqa: E402


def fmt(s):
    m = s.get("map") or {}
    mp = ""
    if m:
        mp = f" | map {m.get('fps', 0):4.1f}/s {m.get('map_ms', 0):4.0f} ms"
        if "coverage" in m:
            mp += f" cov {m['coverage'] * 100:5.1f}% bleached {m['bleached_m2']:.2f} m2"
            if "gt_iou" in m:
                mp += f" IoU {m['gt_iou']:.3f}"
    return (f"rx {s['rx_fps']:5.2f} fps drop {s['dropped_s']:2d} | {s['backend']} infer "
            f"{s['infer_ms']:6.2f} ms {s['pipe_fps']:5.2f} fps{mp} | frames {s['frames_rx']} | "
            f"client {s['client']}")


class Plot:
    """Minimap + rate history in one matplotlib window (optional)."""

    def __init__(self):
        import matplotlib.pyplot as plt
        self.plt = plt
        plt.ion()
        self.fig, (self.ax_map, self.ax_ts) = plt.subplots(1, 2, figsize=(11, 5))
        self.fig.canvas.manager.set_window_title("ReefWatch board status")
        self.ax_map.set_title("health map (board)")
        self.ax_map.axis("off")
        self.im = None
        self.t, self.rx, self.pipe, self.mapf = [], [], [], []

    def update(self, s, png):
        self.t.append(s["uptime_s"])
        self.rx.append(s["rx_fps"])
        self.pipe.append(s["pipe_fps"])
        self.mapf.append((s.get("map") or {}).get("fps", 0.0))
        self.ax_ts.clear()
        for ys, lab in ((self.rx, "rx fps"), (self.pipe, "inference fps"), (self.mapf, "map folds/s")):
            self.ax_ts.plot(self.t[-300:], ys[-300:], label=lab)
        self.ax_ts.set_xlabel("board uptime, s")
        self.ax_ts.legend(loc="upper left")
        m = s.get("map") or {}
        self.ax_ts.set_title(f"{s['backend']}: infer {s['infer_ms']:.1f} ms, coverage "
                             f"{m.get('coverage', 0) * 100:.1f}%")
        if png:
            import matplotlib.image as mpimg
            img = mpimg.imread(io.BytesIO(png), format="png")
            if self.im is None:
                self.im = self.ax_map.imshow(img)
            else:
                self.im.set_data(img)
        self.plt.pause(0.01)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--save-minimap", default="board_minimap.png", help="latest minimap PNG path")
    ap.add_argument("--log", default=None, help="append every status message (JSON lines)")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--seconds", type=float, default=0.0, help="exit after S seconds (0 = never)")
    ap.add_argument("--retry", type=float, default=2.0, help="reconnect delay, s")
    a = ap.parse_args(argv)

    plot = None
    if not a.no_plot:
        try:
            plot = Plot()
        except Exception as exc:                 # no matplotlib, no display: text only
            print(f"[viewer] no live window ({type(exc).__name__}: {exc}); text only", flush=True)
    t0 = time.time()
    n_png = 0
    while not (a.seconds and time.time() - t0 >= a.seconds):
        try:
            sock = socket.create_connection((a.host, a.port), timeout=5.0)
        except OSError as exc:
            print(f"[viewer] {a.host}:{a.port} not reachable ({exc}); retrying", flush=True)
            time.sleep(a.retry)
            continue
        sock.settimeout(10.0)
        print(f"[viewer] connected to {a.host}:{a.port}", flush=True)
        try:
            while not (a.seconds and time.time() - t0 >= a.seconds):
                s, png = read_status(sock)
                print(f"[viewer] {fmt(s)}" + (f" | minimap {len(png) // 1024} kB" if png else ""),
                      flush=True)
                if png:
                    n_png += 1
                    with open(a.save_minimap, "wb") as fh:
                        fh.write(png)
                if a.log:
                    with open(a.log, "a", encoding="utf-8") as fh:
                        fh.write(json.dumps(s) + "\n")
                if plot is not None:
                    plot.update(s, png)
        except (OSError, ValueError, ConnectionError) as exc:
            print(f"[viewer] connection lost ({exc})", flush=True)
            time.sleep(a.retry)
        finally:
            sock.close()
    print(f"[viewer] done; {n_png} minimaps received, latest in {a.save_minimap}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
