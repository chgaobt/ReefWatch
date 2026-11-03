"""CSEG (v1) / CSG2 (v2) wire format: the single Python definition (no torch, no numpy).

The Unreal client (`SegNetClient.cpp`) mirrors these layouts byte for byte; `docs/PROTOCOL.md`
is the human-readable contract. Everything is little-endian.

    REQUEST  "CSEG"  24-byte header, then payload
      0 magic char[4] | 4 frame_id u32 | 8 width u32 | 12 height u32 | 16 fmt u32 | 20 payload_len u32

    RESPONSE "CSEG"  28-byte header, then w*h*4 mask + 4 float32 class_frac + 4 float32 stats
      0 magic | 4 frame_id | 8 width | 12 height | 16 infer_ms f32 | 20 total_ms f32 | 24 payload_len

    REQUEST  "CSG2"  64-byte header, then payload
      0 magic | 4 version u32 (=2) | 8 frame_id | 12 width | 16 height | 20 fmt | 24 payload_len
      28 flags u32 | 32 cam_x | 36 cam_y | 40 cam_z (f32, world cm) | 44 pitch | 48 yaw | 52 roll
      (f32, FRotator degrees) | 56 fov_h_deg f32 | 60 world_time_s f32

    RESPONSE "CSG2"  32-byte header + TLV payload
      0 magic | 4 frame_id | 8 width | 12 height | 16 infer_ms | 20 map_ms | 24 total_ms
      28 payload_len | 32 n x (block_id u32, block_len u32, bytes[block_len])

The version is chosen by the request MAGIC; v1's frame_id sits where a version field would go.
An unknown TLV block id is skipped using its length.

Streaming (2026-11): request flag STREAM_REPLIES (bit 3) asks the server to process only the
newest frame, drop what it cannot keep up with, and answer the frames it processed asynchronously;
WANT_INFO (bit 4) asks for FRAME_INFO (block 7, every response) and SERVER_INFO (block 8, once per
connection). Without either bit the bytes on the wire are exactly the pre-streaming ones.

Native classes (2026-11): request flag WANT_NATIVE_CLASSES (bit 5) asks a server that knows the
model's own classes (SERVER_INFO "native_classes") to colour MASK_RGBA in the native palette and
to add CLASS_FRAC_NATIVE (block 9: count u32 + count float32). CLASS_FRAC stays the 4 health
fractions and the map stays health. A server that ignores the bit sends no block 9; without the
bit the bytes are unchanged.
"""
from __future__ import annotations

import json
import struct

# --------------------------------------------------------------------------- v1 ("CSEG")
MAGIC_V1 = b"CSEG"
REQ_V1 = struct.Struct("<4sIIIII")          # magic, frame_id, w, h, fmt, payload_len
RESP_V1 = struct.Struct("<4sIIIffI")        # magic, frame_id, w, h, infer_ms, total_ms, plen
REQ_V1_SIZE = REQ_V1.size                   # 24
RESP_V1_SIZE = RESP_V1.size                 # 28

# --------------------------------------------------------------------------- v2 ("CSG2")
MAGIC_V2 = b"CSG2"
VERSION_V2 = 2
REQ_V2 = struct.Struct("<4s7I8f")           # 64 bytes
RESP_V2 = struct.Struct("<4sIIIfffI")       # 32 bytes
REQ_V2_SIZE = REQ_V2.size                   # 64
RESP_V2_SIZE = RESP_V2.size                 # 32
TLV = struct.Struct("<II")                  # block_id, block_len

assert (REQ_V1_SIZE, RESP_V1_SIZE, REQ_V2_SIZE, RESP_V2_SIZE) == (24, 28, 64, 32)

MAGIC_SIZE = 4                              # both versions are dispatched on the magic alone
MAX_DIM = 8192                              # same bound the C++ client applies
MAX_PAYLOAD = 256 << 20                     # guard against a desynchronised stream

FMT_RGB8 = 0
FMT_JPEG = 1

# request flags (v2)
FLAG_POSE_VALID = 1 << 0
FLAG_WANT_MINIMAP = 1 << 1
FLAG_RESET_MAP = 1 << 2
#: streaming: the client sends at camera rate without waiting; the server answers only the
#: frames it processed (newest wins), asynchronously, each with FRAME_INFO
FLAG_STREAM_REPLIES = 1 << 3
#: add FRAME_INFO to every response and SERVER_INFO to the first response of the connection
#: (implied by FLAG_STREAM_REPLIES); old servers ignore both bits
FLAG_WANT_INFO = 1 << 4
#: colour MASK_RGBA in the model's native classes and add CLASS_FRAC_NATIVE (block 9), if the
#: server supports it (SERVER_INFO "native_classes"); old servers ignore the bit
FLAG_WANT_NATIVE_CLASSES = 1 << 5
# bits 8..11: water-look preset id (informational, ignored by the server; docs/WATER_LOOK.md)
WATER_LOOK_SHIFT = 8
WATER_LOOK_NAMES = {0: "unreported", 1: "training", 2: "tabuhan", 3: "clear", 4: "off", 15: "other"}


def water_look(flags):
    """Name of the water look a CSG2 request was captured with (flags bits 8..11)."""
    return WATER_LOOK_NAMES.get((int(flags) >> WATER_LOOK_SHIFT) & 0xF, "unknown")

# response block ids (v2)
BLOCK_MASK_RGBA = 1
BLOCK_CLASS_FRAC = 2
BLOCK_STATS = 3
BLOCK_MAP_METRICS = 4
BLOCK_MINIMAP_PNG = 5
BLOCK_ERROR = 6
BLOCK_FRAME_INFO = 7
BLOCK_SERVER_INFO = 8
#: count u32 + float32[count]: pixel fraction of each native class, in SERVER_INFO
#: "native_classes" order; only on a response whose mask is in native colours (WANT_NATIVE_CLASSES)
BLOCK_CLASS_FRAC_NATIVE = 9

BLOCK_NAMES = {BLOCK_MASK_RGBA: "MASK_RGBA", BLOCK_CLASS_FRAC: "CLASS_FRAC", BLOCK_STATS: "STATS",
               BLOCK_MAP_METRICS: "MAP_METRICS", BLOCK_MINIMAP_PNG: "MINIMAP_PNG",
               BLOCK_ERROR: "ERROR", BLOCK_FRAME_INFO: "FRAME_INFO",
               BLOCK_SERVER_INFO: "SERVER_INFO", BLOCK_CLASS_FRAC_NATIVE: "CLASS_FRAC_NATIVE"}
#: most entries a CLASS_FRAC_NATIVE block may carry (the C++ client's limit)
MAX_NATIVE_CLASSES = 32

#: FRAME_INFO (block 7), 32 bytes. A reader must accept a longer block and ignore the tail.
#:   0 frame_id u32      the frame the mask came from (= the header's frame_id)
#:   4 info_flags u32    bit 0 STREAMED (answered under STREAM_REPLIES), bit 1 REUSED (lockstep:
#:                       the mask is a later frame's, the request's own frame was replaced)
#:   8 world_time_s f32  that frame's capture world time, echoed from its request
#:  12 queue_ms f32      request fully received -> inference started (server clock)
#:  16 done_ms f32       request fully received -> class ids ready
#:  20 frames_rx u32     requests received on this connection so far
#:  24 frames_done u32   frames inferred on this connection so far
#:  28 frames_skipped u32  requests never inferred (replaced by a newer frame) on this connection
FRAME_INFO = struct.Struct("<IIfffIII")
FRAME_INFO_SIZE = FRAME_INFO.size           # 32
INFO_STREAMED = 1 << 0
INFO_REUSED = 1 << 1

#: CLASS_FRAC order on the wire (always the 4 health classes, whatever the model predicts)
CLASS_NAMES = ("LIVE", "BLEACHED", "DEGRADED", "OTHER")
STAT_NAMES = ("mean_max_prob", "mean_entropy", "mean_p_bleached", "seg_fps_server")


# --------------------------------------------------------------------------- v1
def pack_request_v1(frame_id, width, height, payload_len, fmt=FMT_RGB8) -> bytes:
    return REQ_V1.pack(MAGIC_V1, int(frame_id), int(width), int(height), int(fmt),
                       int(payload_len))


def unpack_request_v1(buf) -> dict:
    magic, frame_id, width, height, fmt, payload_len = REQ_V1.unpack_from(buf, 0)
    return dict(magic=magic, frame_id=frame_id, width=width, height=height, fmt=fmt,
                payload_len=payload_len)


def v1_payload_len(width, height) -> int:
    """The only payload length a v1 response may carry (the C++ client checks it exactly)."""
    return int(width) * int(height) * 4 + 4 * 4 + 4 * 4


def pack_response_header_v1(frame_id, width, height, infer_ms, total_ms) -> bytes:
    return RESP_V1.pack(MAGIC_V1, int(frame_id), int(width), int(height), float(infer_ms),
                        float(total_ms), v1_payload_len(width, height))


def parse_response_v1(header_bytes, payload) -> dict:
    magic, frame_id, width, height, infer_ms, total_ms, plen = RESP_V1.unpack_from(header_bytes, 0)
    n = width * height * 4
    if plen != v1_payload_len(width, height) or len(payload) != plen:
        raise ValueError(f"v1 payload_len {plen} / {len(payload)} B for {width}x{height}")
    return dict(magic=magic, frame_id=frame_id, width=width, height=height, infer_ms=infer_ms,
                total_ms=total_ms, payload_len=plen, mask_rgba=bytes(payload[:n]),
                class_frac=list(struct.unpack_from("<4f", payload, n)),
                stats=list(struct.unpack_from("<4f", payload, n + 16)))


# --------------------------------------------------------------------------- v2 request
def pack_request_v2(frame_id, width, height, payload_len, *, fmt=FMT_RGB8, flags=0,
                    cam=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0), fov_h_deg=78.0,
                    world_time_s=0.0, version=VERSION_V2) -> bytes:
    """The 64-byte v2 request header. `rot` is (pitch, yaw, roll) in degrees."""
    return REQ_V2.pack(MAGIC_V2, int(version), int(frame_id), int(width), int(height),
                       int(fmt), int(payload_len), int(flags),
                       float(cam[0]), float(cam[1]), float(cam[2]),
                       float(rot[0]), float(rot[1]), float(rot[2]),
                       float(fov_h_deg), float(world_time_s))


def unpack_request_v2(buf) -> dict:
    (magic, version, frame_id, width, height, fmt, payload_len, flags,
     cx, cy, cz, pitch, yaw, roll, fov, t) = REQ_V2.unpack_from(buf, 0)
    return dict(magic=magic, version=version, frame_id=frame_id, width=width, height=height,
                fmt=fmt, payload_len=payload_len, flags=flags,
                cam=(cx, cy, cz), rot=(pitch, yaw, roll), fov_h_deg=fov, world_time_s=t,
                pose_valid=bool(flags & FLAG_POSE_VALID),
                want_minimap=bool(flags & FLAG_WANT_MINIMAP),
                reset_map=bool(flags & FLAG_RESET_MAP))


def wants_info(flags) -> bool:
    """FRAME_INFO / SERVER_INFO requested (WANT_INFO, or implied by STREAM_REPLIES)."""
    return bool(int(flags) & (FLAG_WANT_INFO | FLAG_STREAM_REPLIES))


def wants_native(flags) -> bool:
    """Native-class mask + CLASS_FRAC_NATIVE requested (WANT_NATIVE_CLASSES)."""
    return bool(int(flags) & FLAG_WANT_NATIVE_CLASSES)


# --------------------------------------------------------------------------- v2 response
def pack_frame_info(frame_id, world_time_s=0.0, queue_ms=0.0, done_ms=0.0, frames_rx=0,
                    frames_done=0, frames_skipped=0, streamed=False, reused=False) -> bytes:
    """The 32-byte FRAME_INFO block body."""
    flags = (INFO_STREAMED if streamed else 0) | (INFO_REUSED if reused else 0)
    return FRAME_INFO.pack(int(frame_id), flags, float(world_time_s), float(queue_ms),
                           float(done_ms), int(frames_rx), int(frames_done), int(frames_skipped))


def unpack_frame_info(buf) -> dict:
    if len(buf) < FRAME_INFO_SIZE:
        raise ValueError(f"FRAME_INFO is {len(buf)} B, needs >= {FRAME_INFO_SIZE}")
    (fid, flags, wt, q, d, rx, done, skipped) = FRAME_INFO.unpack_from(buf, 0)
    return dict(frame_id=fid, info_flags=flags, streamed=bool(flags & INFO_STREAMED),
                reused=bool(flags & INFO_REUSED), world_time_s=wt, queue_ms=q, done_ms=d,
                frames_rx=rx, frames_done=done, frames_skipped=skipped)


def pack_class_frac_native(frac) -> bytes:
    """CLASS_FRAC_NATIVE (block 9) body: count u32, then count little-endian float32."""
    vals = [float(v) for v in frac]
    if not 0 < len(vals) <= MAX_NATIVE_CLASSES:
        raise ValueError(f"CLASS_FRAC_NATIVE needs 1..{MAX_NATIVE_CLASSES} entries, got {len(vals)}")
    return struct.pack(f"<I{len(vals)}f", len(vals), *vals)


def unpack_class_frac_native(buf) -> list:
    if len(buf) < 4:
        raise ValueError(f"CLASS_FRAC_NATIVE is {len(buf)} B")
    (n,) = struct.unpack_from("<I", buf, 0)
    if not 0 < n <= MAX_NATIVE_CLASSES or len(buf) != 4 + 4 * n:
        raise ValueError(f"CLASS_FRAC_NATIVE count {n} does not fit {len(buf)} B")
    return list(struct.unpack_from(f"<{n}f", buf, 4))


def rgb_hex(rgb) -> str:
    """(r, g, b[, a]) -> "#rrggbb"."""
    return "#%02x%02x%02x" % tuple(int(c) for c in tuple(rgb)[:3])


def class_table(names, colours) -> dict:
    """A SERVER_INFO class table: {"names": [...], "colors": ["#rrggbb", ...]}, in the order of
    the class ids / fractions it describes. `colours` rows are (r, g, b[, a])."""
    names = [str(n) for n in names]
    cols = [rgb_hex(c) for c in colours]
    if len(names) != len(cols):
        raise ValueError(f"{len(names)} names but {len(cols)} colours")
    return {"names": names, "colors": cols}


def pack_server_info(name, kind, **extra) -> bytes:
    """SERVER_INFO (block 8): one line of UTF-8 JSON. `name` is what the HUD shows ("PC GPU",
    "KV260 DPU", ...); `kind` is "pc" or "board". The HUD also reads "health_classes" and
    "native_classes" (`class_table`; the latter only from a server that honours
    WANT_NATIVE_CLASSES); anything else is informational."""
    d = dict(name=str(name), kind=str(kind))
    d.update(extra)
    return json.dumps(d, separators=(",", ":"), default=str).encode("utf-8")


def pack_tlv(block_id, data) -> bytes:
    b = bytes(data)
    return TLV.pack(int(block_id), len(b)) + b


def build_response_v2(frame_id, width, height, infer_ms, map_ms, total_ms, blocks):
    """-> (header_bytes, [pieces...]) ready for a scatter send.

    `blocks` is an ordered iterable of `(block_id, data)`; `data` may be any buffer and is
    not copied when it is already a memoryview.
    """
    parts = []
    payload_len = 0
    for block_id, data in blocks:
        view = data if isinstance(data, memoryview) else memoryview(bytes(data))
        n = view.nbytes
        parts.append(TLV.pack(int(block_id), n))
        parts.append(view)
        payload_len += TLV.size + n
    head = RESP_V2.pack(MAGIC_V2, int(frame_id), int(width), int(height),
                        float(infer_ms), float(map_ms), float(total_ms), payload_len)
    return head, parts


def unpack_response_header_v2(buf) -> dict:
    magic, frame_id, width, height, infer_ms, map_ms, total_ms, plen = RESP_V2.unpack_from(buf, 0)
    return dict(magic=magic, frame_id=frame_id, width=width, height=height,
                infer_ms=infer_ms, map_ms=map_ms, total_ms=total_ms, payload_len=plen)


def iter_tlv(payload):
    """Yield `(block_id, memoryview)` per TLV block; ValueError on a truncated block."""
    mv = memoryview(payload)
    off, n = 0, mv.nbytes
    while off < n:
        if off + TLV.size > n:
            raise ValueError(f"truncated TLV header at offset {off} of {n}")
        block_id, block_len = TLV.unpack_from(mv, off)
        off += TLV.size
        if block_len > n - off:
            raise ValueError(f"TLV block {block_id} claims {block_len} B, only {n - off} left")
        yield block_id, mv[off:off + block_len]
        off += block_len


def parse_response_v2(header_bytes, payload) -> dict:
    """Client-side decode: header fields + every known block (unknown ids are kept raw)."""
    out = unpack_response_header_v2(header_bytes)
    out["blocks"] = {bid: bytes(view) for bid, view in iter_tlv(payload)}
    b = out["blocks"]
    if BLOCK_MASK_RGBA in b:
        out["mask_rgba"] = b[BLOCK_MASK_RGBA]
    if BLOCK_CLASS_FRAC in b:
        out["class_frac"] = list(struct.unpack("<4f", b[BLOCK_CLASS_FRAC]))
    if BLOCK_STATS in b:
        out["stats"] = list(struct.unpack("<4f", b[BLOCK_STATS]))
    if BLOCK_MAP_METRICS in b:
        out["map_metrics"] = json.loads(b[BLOCK_MAP_METRICS].decode("utf-8"))
    if BLOCK_MINIMAP_PNG in b:
        out["minimap_png"] = b[BLOCK_MINIMAP_PNG]
    if BLOCK_ERROR in b:
        out["error"] = b[BLOCK_ERROR].decode("utf-8", "replace")
    if BLOCK_FRAME_INFO in b:
        out["frame_info"] = unpack_frame_info(b[BLOCK_FRAME_INFO])
    if BLOCK_SERVER_INFO in b:
        out["server_info"] = json.loads(b[BLOCK_SERVER_INFO].decode("utf-8"))
    if BLOCK_CLASS_FRAC_NATIVE in b:
        out["class_frac_native"] = unpack_class_frac_native(b[BLOCK_CLASS_FRAC_NATIVE])
    return out
