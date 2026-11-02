#!/usr/bin/env python
"""Play a .csgrec recording (record_stream.py) to the board, like a ROS bag.

Sends every recorded CSG2 message byte for byte, one way, with the recorded timing. Needs only
the Python standard library, so any machine wired to the board can play it.

    python replay_stream.py survey.csgrec --host 192.168.137.40          # recorded timing
    python replay_stream.py survey.csgrec --host 192.168.137.40 --hz 30  # fixed 30 frames/s
    python replay_stream.py survey.csgrec --host 192.168.137.40 --loop

The first frames of an -autosurvey recording carry RESET_MAP, so every replay starts a fresh map.
"""
from __future__ import annotations

import argparse
import socket
import struct
import sys
import time
import zlib

FILE_MAGIC = b"CSGREC01"
RECORD = struct.Struct("<dI")


def records(path):
    """-> (t_s, message bytes) for each recorded message."""
    with open(path, "rb") as f:
        if f.read(len(FILE_MAGIC)) != FILE_MAGIC:
            raise SystemExit(f"{path}: not a .csgrec file")
        while True:
            head = f.read(RECORD.size)
            if len(head) < RECORD.size:
                return
            t, n = RECORD.unpack(head)
            blob = f.read(n)
            if len(blob) < n:
                return                                          # truncated tail: stop cleanly
            yield t, zlib.decompress(blob)


def connect(host, port, wait_s):
    end = time.monotonic() + wait_s
    while True:
        try:
            s = socket.create_connection((host, port), timeout=5)
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            s.settimeout(None)
            return s
        except OSError as exc:
            if time.monotonic() > end:
                raise SystemExit(f"[replay] cannot connect to {host}:{port}: {exc}")
            time.sleep(1.0)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", help=".csgrec recording")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5556)
    ap.add_argument("--speed", type=float, default=1.0, help="time scale for recorded timing (2 = twice as fast)")
    ap.add_argument("--hz", type=float, default=0.0, help="fixed send rate instead of recorded timing")
    ap.add_argument("--loop", action="store_true", help="start again at the end")
    ap.add_argument("--wait", type=float, default=30.0, help="seconds to keep retrying the connection")
    args = ap.parse_args(argv)

    s = connect(args.host, args.port, args.wait)
    print(f"[replay] {args.file} -> {args.host}:{args.port}"
          f" ({'%g Hz fixed' % args.hz if args.hz > 0 else 'recorded timing x%g' % args.speed})", flush=True)
    sent = nbytes = 0
    start = time.monotonic()
    last_print = start
    try:
        while True:
            t_base = time.monotonic()
            for k, (t, msg) in enumerate(records(args.file)):
                due = t_base + (k / args.hz if args.hz > 0 else t / args.speed)
                delay = due - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                s.sendall(msg)
                sent += 1
                nbytes += len(msg)
                now = time.monotonic()
                if now - last_print >= 2.0:
                    last_print = now
                    el = now - start
                    print(f"[replay] {sent} frames  {el:6.1f} s  {sent / el:5.2f} fps"
                          f"  {nbytes / el / 1e6:5.1f} MB/s", flush=True)
            if not args.loop:
                break
    except (KeyboardInterrupt, BrokenPipeError, ConnectionResetError) as exc:
        if not isinstance(exc, KeyboardInterrupt):
            print(f"[replay] board closed the connection: {exc}", flush=True)
    finally:
        s.close()
    el = max(time.monotonic() - start, 1e-6)
    print(f"[replay] done: {sent} frames in {el:.1f} s = {sent / el:.2f} fps", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
