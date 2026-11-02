"""Optional status port (default TCP 5557) for plotting on the PC. The board only writes.

Each message, little-endian:

    0 magic "RWST" | 4 json_len u32 | 8 png_len u32 | 12 json (UTF-8) | 12+json_len png

`png` is the latest minimap (RGBA, north-up, alpha 0 on unsurveyed cells, as MINIMAP_PNG in
docs/PROTOCOL.md), sent to a viewer only when it changed since that viewer's last message;
otherwise png_len is 0. Messages go out at most once a second. Any number of viewers; each has
its own sender thread and a newest-message slot, so a slow or stuck viewer only loses messages
(and is dropped after `send_timeout` s) - the pipeline never waits on it.
"""
from __future__ import annotations

import json
import socket
import struct
import threading

from .slot import NewestSlot

MAGIC = b"RWST"
HDR = struct.Struct("<4sII")
DEFAULT_PORT = 5557


def pack_status(status: dict, png: bytes = b"") -> bytes:
    js = json.dumps(status, separators=(",", ":")).encode("utf-8")
    return HDR.pack(MAGIC, len(js), len(png or b"")) + js + (png or b"")


def _recv_exact(sock, n):
    buf = bytearray(n)
    mv = memoryview(buf)
    got = 0
    while got < n:
        k = sock.recv_into(mv[got:], n - got)
        if k == 0:
            raise ConnectionError("status port closed")
        got += k
    return bytes(buf)


def read_status(sock):
    """Viewer side: block for one message -> (status dict, png bytes or b'')."""
    magic, jl, pl = HDR.unpack(_recv_exact(sock, HDR.size))
    if magic != MAGIC:
        raise ValueError(f"bad status magic {magic!r}")
    js = json.loads(_recv_exact(sock, jl).decode("utf-8"))
    png = _recv_exact(sock, pl) if pl else b""
    return js, png


class _Viewer:
    def __init__(self, server, conn, addr, send_timeout):
        self.server, self.conn, self.addr = server, conn, addr
        self.slot = NewestSlot()
        self.png_serial = -1
        self.alive = True
        conn.settimeout(send_timeout)
        self.thread = threading.Thread(target=self._run, daemon=True, name=f"status-{addr[1]}")
        self.thread.start()

    def _run(self):
        try:
            while self.alive:
                item = self.slot.get(timeout=1.0)
                if item is None:
                    continue
                status, png, serial = item
                if png and serial != self.png_serial:     # only a minimap this viewer lacks
                    self.png_serial = serial
                else:
                    png = b""
                self.conn.sendall(pack_status(status, png))
        except OSError:
            pass
        finally:
            self.close()

    def close(self):
        if self.alive:
            self.alive = False
            self.slot.close()
            try:
                self.conn.close()
            except OSError:
                pass
            self.server._remove(self)


class StatusServer:
    def __init__(self, host="0.0.0.0", port=DEFAULT_PORT, log=print, send_timeout=5.0):
        self.host, self.port = host, int(port)
        self.log = log
        self.send_timeout = float(send_timeout)
        self.lock = threading.Lock()
        self.viewers = []
        self.listener = None
        self.stopping = False

    def start(self):
        ls = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        ls.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        ls.bind((self.host, self.port))
        ls.listen(8)
        self.listener = ls
        self.port = ls.getsockname()[1]
        threading.Thread(target=self._accept, daemon=True, name="status-accept").start()
        return self.port

    def _accept(self):
        while not self.stopping:
            try:
                conn, addr = self.listener.accept()
            except OSError:
                return
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            with self.lock:
                self.viewers.append(_Viewer(self, conn, addr, self.send_timeout))
            self.log(f"[status] viewer connected {addr[0]}:{addr[1]}")

    def _remove(self, viewer):
        with self.lock:
            if viewer in self.viewers:
                self.viewers.remove(viewer)
                if not self.stopping:
                    self.log(f"[status] viewer {viewer.addr[0]}:{viewer.addr[1]} left")

    @property
    def n_viewers(self):
        return len(self.viewers)

    def publish(self, status: dict, png=None, png_serial=0):
        """Queue one message per viewer; never blocks on a socket."""
        with self.lock:
            viewers = list(self.viewers)
        for v in viewers:
            v.slot.put((status, png, png_serial))

    def stop(self):
        self.stopping = True
        if self.listener is not None:
            try:
                self.listener.close()
            except OSError:
                pass
        with self.lock:
            viewers = list(self.viewers)
        for v in viewers:
            v.close()
