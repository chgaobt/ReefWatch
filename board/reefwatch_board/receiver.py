"""One-way CSG2 receiver: Unreal streams frames + pose like a camera and a navigation unit would;
this side never writes a byte back on the socket.

* listens on host:port (default 0.0.0.0:5556); one client at a time, a new connection closes
  the current one;
* a reader thread per connection parses requests exactly as docs/PROTOCOL.md defines them
  (CSG2, and CSEG v1 without a pose) and drains the socket continuously into a single
  newest-frame slot, so a slow pipeline drops frames (counted) instead of backing up TCP and
  stalling the sender;
* RESET_MAP is turned into a generation counter on its rising edge, so a reset can never be lost
  with a dropped frame: every frame carries the generation it arrived under and the map applies
  the reset before the first frame of the new generation.

A malformed header closes the connection, as on the PC server (the stream cannot be resynced).

With `reply_ctx` (`--reply`) each connection also gets a `reply.Replier`: the reader registers
every request with it before the slot hand-off, and the Replier's own thread writes the responses
(see reply.py). Without it (the default) nothing here changes and nothing is ever written.
"""
from __future__ import annotations

import socket
import threading
import time

import numpy as np

from reefwatch_runtime import protocol as P


class Frame:
    __slots__ = ("frame_id", "width", "height", "rgb", "pose", "fov_h_deg", "world_time_s",
                 "flags", "reset_gen", "t_rx", "version", "reply")

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))


def recv_exact(conn, mv, n, stop=None):
    """Fill mv[:n]; False on EOF. With `stop` (an Event; reply mode, where the socket has a send
    timeout that also applies to reads) a read timeout is retried until `stop` is set."""
    got = 0
    while got < n:
        if stop is None:
            k = conn.recv_into(mv[got:n], n - got)
        else:
            try:
                k = conn.recv_into(mv[got:n], n - got)
            except socket.timeout:
                if stop.is_set():
                    return False
                continue
        if k == 0:
            return False
        got += k
    return True


def decode_jpeg(buf):
    """JPEG -> RGB with cv2 or Pillow if the board has one; None otherwise (Unreal sends RGB8)."""
    try:
        import cv2
        a = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
        return None if a is None else cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
    except ImportError:
        pass
    try:
        import io
        from PIL import Image
        return np.asarray(Image.open(io.BytesIO(buf)).convert("RGB"))
    except ImportError:
        return None


class Receiver:
    def __init__(self, slot, host="0.0.0.0", port=5556, log=print, on_disconnect=None,
                 rcvbuf=4 << 20, reply_ctx=None):
        self.slot = slot
        self.reply_ctx = reply_ctx
        self.host, self.port = host, int(port)
        self.log = log
        self.on_disconnect = on_disconnect
        self.rcvbuf = int(rcvbuf)
        self.lock = threading.Lock()
        self.listener = None
        self.conn = None
        self.conn_id = 0
        self.client = "waiting"
        self.stop_event = threading.Event()
        self.reset_gen = 0
        # counters (read by the status thread without the lock; ints are atomic enough)
        self.frames_rx = 0
        self.bytes_rx = 0
        self.frames_bad = 0
        self.connections = 0
        self.last_frame_id = 0
        self._warned = set()
        self._threads = []

    # ------------------------------------------------------------------ lifecycle
    def bind(self):
        ls = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        ls.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        ls.bind((self.host, self.port))
        ls.listen(4)
        self.listener = ls
        self.port = ls.getsockname()[1]
        return self.port

    def start(self):
        if self.listener is None:
            self.bind()
        t = threading.Thread(target=self._acceptor, daemon=True, name="rx-accept")
        t.start()
        self._threads.append(t)

    def stop(self):
        self.stop_event.set()
        for s in (self.listener, self.conn):
            if s is not None:
                try:
                    s.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                try:
                    s.close()
                except OSError:
                    pass

    # ------------------------------------------------------------------ connections
    def _acceptor(self):
        while not self.stop_event.is_set():
            try:
                conn, addr = self.listener.accept()
            except OSError:
                return
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            try:
                conn.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self.rcvbuf)
            except OSError:
                pass
            replier = None
            if self.reply_ctx is not None:
                from .reply import Replier
                try:
                    conn.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, self.rcvbuf)
                except OSError:
                    pass
                conn.settimeout(self.reply_ctx.send_timeout)   # bounds every response send
                replier = Replier(self.reply_ctx, conn, addr)
            with self.lock:
                old = self.conn
                self.conn = conn
                self.conn_id += 1
                cid = self.conn_id
                self.connections += 1
                self.client = f"{addr[0]}:{addr[1]}"
            if old is not None:
                self.log(f"[rx] new connection from {addr[0]}:{addr[1]} replaces the current client")
                try:
                    old.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
            else:
                self.log(f"[rx] client connected {addr[0]}:{addr[1]}")
            t = threading.Thread(target=self._reader, args=(conn, addr, cid, replier), daemon=True,
                                 name=f"rx-reader-{cid}")
            t.start()

    def _warn_once(self, key, msg):
        if key not in self._warned:
            self._warned.add(key)
            self.log(msg)

    def _reader(self, conn, addr, cid, replier=None):
        hdr = bytearray(max(P.REQ_V1_SIZE, P.REQ_V2_SIZE))
        mv = memoryview(hdr)
        prev_reset = False
        reason = "EOF"
        stop = None if replier is None else self.stop_event     # reply mode: socket has a timeout
        try:
            while not self.stop_event.is_set():
                if not recv_exact(conn, mv, P.MAGIC_SIZE, stop):
                    return
                magic = bytes(hdr[:P.MAGIC_SIZE])
                if magic == P.MAGIC_V2:
                    if not recv_exact(conn, mv[P.MAGIC_SIZE:], P.REQ_V2_SIZE - P.MAGIC_SIZE, stop):
                        return
                    r = P.unpack_request_v2(hdr)
                    if r["version"] != P.VERSION_V2:
                        reason = f"MALFORMED CSG2 version {r['version']}"
                        return
                    hsize = P.REQ_V2_SIZE
                elif magic == P.MAGIC_V1:
                    if not recv_exact(conn, mv[P.MAGIC_SIZE:P.REQ_V1_SIZE],
                                      P.REQ_V1_SIZE - P.MAGIC_SIZE, stop):
                        return
                    r = P.unpack_request_v1(hdr)
                    r.update(flags=0, pose_valid=False, reset_map=False, cam=None, rot=None,
                             fov_h_deg=0.0, world_time_s=0.0, version=1)
                    hsize = P.REQ_V1_SIZE
                else:
                    reason = f"MALFORMED magic {magic!r}"
                    return
                w, h, fmt, plen = r["width"], r["height"], r["fmt"], r["payload_len"]
                if (not (0 < w <= P.MAX_DIM) or not (0 < h <= P.MAX_DIM)
                        or plen > P.MAX_PAYLOAD or fmt not in (P.FMT_RGB8, P.FMT_JPEG)
                        or (fmt == P.FMT_RGB8 and plen != w * h * 3)):
                    reason = f"MALFORMED header {w}x{h} fmt={fmt} len={plen}"
                    return
                buf = bytearray(plen)
                if plen and not recv_exact(conn, memoryview(buf), plen, stop):
                    return
                t_rx = time.perf_counter()
                self.frames_rx += 1
                self.bytes_rx += hsize + plen
                self.last_frame_id = r["frame_id"]

                reset = bool(r["reset_map"])
                if reset and not prev_reset:
                    self.reset_gen += 1
                    self.log(f"[rx] RESET_MAP at frame {r['frame_id']} (generation {self.reset_gen})")
                prev_reset = reset

                reply = None
                if replier is not None:              # reply mode: register before the hand-off
                    entry = replier.expect(r["frame_id"], w, h, r["version"], r["flags"],
                                           self.reset_gen, t_rx, r["world_time_s"])
                    if entry is None:                # connection is being closed by the replier
                        reason = "closed by the reply writer"
                        return
                    reply = (replier, entry)

                if fmt == P.FMT_RGB8:
                    rgb = np.frombuffer(buf, np.uint8).reshape(h, w, 3)
                else:
                    rgb = decode_jpeg(bytes(buf))
                    if rgb is None or rgb.shape[:2] != (h, w):
                        self.frames_bad += 1
                        self._warn_once("jpeg", "[rx] JPEG frame skipped: no decoder on this host "
                                        "or wrong size (send RGB8, fmt 0)")
                        if reply is not None:
                            replier.failed(entry, "JPEG frame not decodable on the board "
                                                  "(no cv2/Pillow, or wrong size): send RGB8")
                        continue
                pose = (dict(location_cm=r["cam"], rotation_deg=r["rot"])
                        if r["pose_valid"] else None)
                old = self.slot.put(Frame(frame_id=r["frame_id"], width=w, height=h, rgb=rgb,
                                          pose=pose, fov_h_deg=float(r["fov_h_deg"]),
                                          world_time_s=float(r["world_time_s"]), flags=r["flags"],
                                          reset_gen=self.reset_gen, t_rx=t_rx,
                                          version=r["version"], reply=reply))
                if old is not None and old.reply is not None:
                    old.reply[0].superseded(old.reply[1])
        except OSError as exc:
            reason = f"{type(exc).__name__}: {exc}"
        finally:
            if replier is not None:
                replier.close()
            try:
                conn.close()
            except OSError:
                pass
            with self.lock:
                current = cid == self.conn_id
                if current:
                    self.conn = None
                    self.client = "waiting"
            if not self.stop_event.is_set():
                self.log(f"[rx] client {addr[0]}:{addr[1]} {'gone' if current else 'replaced'} ({reason})")
                if self.on_disconnect is not None:
                    self.on_disconnect(replaced=not current)
