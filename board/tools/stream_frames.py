#!/usr/bin/env python
"""Stream CSG2 frames one-way to the board receiver, like Unreal in send-only mode.

Sends the bundled 640x360 AUV frame with camera poses flown along the survey route
(data/survey/tabuhan_p1_lawnmower.json; capture camera 72 cm ahead of the pawn, pitch -60,
fov 78), so the board's health map fills. Never reads from the socket.

    python board/tools/stream_frames.py --host 192.168.1.2                # 10 Hz until Ctrl+C
    python board/tools/stream_frames.py --host 127.0.0.1 --hz 0 --seconds 20   # max-rate bench
    python board/tools/stream_frames.py --frames 300 --reset               # fresh map, 30 s
"""
from __future__ import annotations

import argparse
import json
import math
import os
import socket
import sys
import time

import numpy as np

BOARD_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIM_ROOT = os.path.dirname(BOARD_DIR)
sys.path.insert(0, os.path.join(SIM_ROOT, "runtime"))
from reefwatch_runtime import protocol as P  # noqa: E402

DEFAULT_FRAME = os.path.join(SIM_ROOT, "runtime", "tests", "data", "auv_frame_640x360.png")
DEFAULT_ROUTE = os.path.join(SIM_ROOT, "data", "survey", "tabuhan_p1_lawnmower.json")
CAMERA_FORWARD_CM, CAMERA_UP_CM = 72.0, 8.0          # SegCapture offset on the AUV hull


def load_frame(path):
    try:
        from PIL import Image
        return np.ascontiguousarray(np.asarray(Image.open(path).convert("RGB")))
    except ImportError:
        print("[stream] Pillow missing: sending a synthetic gradient frame", flush=True)
        y, x = np.mgrid[0:360, 0:640]
        return np.dstack([x * 255 // 639, y * 255 // 359, np.full_like(x, 128)]).astype(np.uint8)


class RoutePoses:
    """Camera poses at constant speed along the waypoint polyline (loops at the end)."""

    def __init__(self, path, speed_cm_s=50.0, pitch=-60.0):
        with open(path, encoding="utf-8") as fh:
            self.wp = np.asarray(json.load(fh)["waypoints"], float)
        seg = np.diff(self.wp, axis=0)
        self.seg_len = np.linalg.norm(seg[:, :2], axis=1)
        self.cum = np.concatenate([[0.0], np.cumsum(self.seg_len)])
        self.speed, self.pitch = float(speed_cm_s), float(pitch)

    @property
    def length_cm(self):
        return float(self.cum[-1])

    def at(self, t_s):
        d = (self.speed * t_s) % self.length_cm
        i = int(np.clip(np.searchsorted(self.cum, d, side="right") - 1, 0, len(self.seg_len) - 1))
        f = (d - self.cum[i]) / max(self.seg_len[i], 1e-6)
        p = self.wp[i] + f * (self.wp[i + 1] - self.wp[i])
        dx, dy = self.wp[i + 1, :2] - self.wp[i, :2]
        yaw = math.degrees(math.atan2(dy, dx))
        cam = (p[0] + CAMERA_FORWARD_CM * math.cos(math.radians(yaw)),
               p[1] + CAMERA_FORWARD_CM * math.sin(math.radians(yaw)), p[2] + CAMERA_UP_CM)
        return cam, (self.pitch, yaw, 0.0)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5556)
    ap.add_argument("--hz", type=float, default=10.0, help="frames per second (0 = as fast as possible)")
    ap.add_argument("--frames", type=int, default=0, help="stop after N frames (0 = no limit)")
    ap.add_argument("--seconds", type=float, default=0.0, help="stop after S seconds (0 = no limit)")
    ap.add_argument("--frame", default=DEFAULT_FRAME, help="RGB image to send (640x360 PNG)")
    ap.add_argument("--route", default=DEFAULT_ROUTE)
    ap.add_argument("--speed", type=float, default=50.0, help="cm/s along the route (survey speed)")
    ap.add_argument("--pitch", type=float, default=-60.0)
    ap.add_argument("--fov", type=float, default=78.0)
    ap.add_argument("--reset", action="store_true", help="RESET_MAP on the first frame")
    ap.add_argument("--no-pose", action="store_true", help="clear POSE_VALID (no map)")
    a = ap.parse_args(argv)

    rgb = load_frame(a.frame)
    h, w = rgb.shape[:2]
    payload = rgb.tobytes()
    route = RoutePoses(a.route, a.speed, a.pitch)
    sock = socket.create_connection((a.host, a.port), timeout=10.0)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 << 20)
    print(f"[stream] -> {a.host}:{a.port} {w}x{h} RGB8 at {a.hz or 'max'} Hz, route "
          f"{route.length_cm / 100:.0f} m at {a.speed:g} cm/s", flush=True)

    period = 1.0 / a.hz if a.hz > 0 else 0.0
    nominal = period or 0.1             # route time per frame at max rate: as if 10 Hz
    t0 = time.perf_counter()
    last_print, sent_at_print = t0, 0
    n = 0
    try:
        while True:
            if a.frames and n >= a.frames:
                break
            now = time.perf_counter()
            if a.seconds and now - t0 >= a.seconds:
                break
            cam, rot = route.at(n * nominal)
            flags = 0 if a.no_pose else P.FLAG_POSE_VALID
            if a.reset and n == 0:
                flags |= P.FLAG_RESET_MAP
            head = P.pack_request_v2(n + 1, w, h, len(payload), flags=flags, cam=cam, rot=rot,
                                     fov_h_deg=a.fov, world_time_s=n * nominal)
            sock.sendall(head)
            sock.sendall(payload)
            n += 1
            now = time.perf_counter()
            if now - last_print >= 1.0:
                print(f"[stream] sent {n} frames, {(n - sent_at_print) / (now - last_print):5.2f} fps",
                      flush=True)
                last_print, sent_at_print = now, n
            if period:
                sleep = t0 + n * period - time.perf_counter()
                if sleep > 0:
                    time.sleep(sleep)
    except (KeyboardInterrupt, OSError) as exc:
        if not isinstance(exc, KeyboardInterrupt):
            print(f"[stream] connection lost: {exc}", flush=True)
    finally:
        sock.close()
    dt = time.perf_counter() - t0
    print(f"[stream] done: {n} frames in {dt:.1f} s = {n / max(dt, 1e-9):.2f} fps "
          f"({n * (len(payload) + P.REQ_V2_SIZE) / max(dt, 1e-9) / 1e6:.1f} MB/s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
