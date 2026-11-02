#!/usr/bin/env python
"""Record the CSG2 stream Unreal sends, to a file that replay_stream.py plays back (a "bag").

Listens like the board does and writes every message byte for byte with its arrival time.
Unreal reconnects whenever a send stalls, so the recorder keeps accepting clients and appends
to the same file; it stops after `--idle` seconds with no client, or on Ctrl+C. Standard library
only.

    python board/tools/record_stream.py out/rec/survey.csgrec        # listen on 0.0.0.0:5556
    tools\\run_sim.bat -sendonly -segserver=127.0.0.1 -autosurvey -autosurveyquit

File format (.csgrec): b"CSGREC01", then one record per message:
    <d t_s> <I blob_len> <blob>     t_s = seconds since the first message
                                    blob = zlib(64-byte CSG2 header + payload)
The default zlib level 0 only stores: the socket must be drained at Unreal's rate (6.9 MB/s at
10 Hz) even while Unreal loads the reef on every core, and real compression cannot keep up then.
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
REQ_V2 = struct.Struct("<4s7I8f")          # = reefwatch_runtime.protocol.REQ_V2 (64 bytes)
MAGIC_V2 = b"CSG2"


def recv_exact(conn, n):
    buf = bytearray(n)
    mv, got = memoryview(buf), 0
    while got < n:
        k = conn.recv_into(mv[got:], n - got)
        if k == 0:
            return None
        got += k
    return bytes(buf)


class Recorder:
    def __init__(self, f, level):
        self.f, self.level = f, level
        self.n = self.raw = self.packed = 0
        self.t0 = self.last_print = None

    def write(self, msg):
        now = time.monotonic()
        if self.t0 is None:
            self.t0 = self.last_print = now
        blob = zlib.compress(msg, self.level)
        self.f.write(RECORD.pack(now - self.t0, len(blob)))
        self.f.write(blob)
        self.n += 1
        self.raw += len(msg)
        self.packed += len(blob)
        if now - self.last_print >= 2.0:
            self.last_print = now
            el = now - self.t0
            print(f"[record] {self.n} frames  {el:6.1f} s  {self.n / max(el, 1e-6):5.2f} fps"
                  f"  raw {self.raw / 1e6:7.1f} MB  file {self.packed / 1e6:7.1f} MB", flush=True)

    def client(self, conn):
        """Record one connection until it closes; -> number of messages."""
        n0 = self.n
        while True:
            hdr = recv_exact(conn, REQ_V2.size)
            if hdr is None:
                break
            fields = REQ_V2.unpack(hdr)
            if fields[0] != MAGIC_V2:
                print(f"[record] not a CSG2 header ({fields[0]!r}); dropping this client", flush=True)
                break
            payload = recv_exact(conn, fields[6])              # payload_len
            if payload is None:
                break
            self.write(hdr + payload)
        return self.n - n0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", help="output .csgrec file")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5556)
    ap.add_argument("--level", type=int, default=0, help="zlib level (0 = store, fastest)")
    ap.add_argument("--idle", type=float, default=20.0,
                    help="stop after this many seconds with no client, once something was recorded")
    args = ap.parse_args(argv)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.host, args.port))
    srv.listen(1)
    print(f"[record] listening on {args.host}:{args.port} -> {args.out}", flush=True)

    with open(args.out, "wb") as f:
        f.write(FILE_MAGIC)
        rec = Recorder(f, args.level)
        try:
            while True:
                srv.settimeout(args.idle if rec.n else None)
                try:
                    conn, addr = srv.accept()
                except socket.timeout:
                    print(f"[record] no client for {args.idle:g} s; stopping", flush=True)
                    break
                conn.settimeout(None)
                print(f"[record] client {addr[0]}:{addr[1]}", flush=True)
                got = rec.client(conn)
                conn.close()
                print(f"[record] client closed after {got} frames (total {rec.n})", flush=True)
        except KeyboardInterrupt:
            pass
    srv.close()
    dur = 0.0 if rec.t0 is None else time.monotonic() - rec.t0
    print(f"[record] done: {rec.n} frames over {dur:.1f} s, raw {rec.raw / 1e6:.1f} MB,"
          f" file {rec.packed / 1e6:.1f} MB -> {args.out}", flush=True)
    return 0 if rec.n else 1


if __name__ == "__main__":
    sys.exit(main())
