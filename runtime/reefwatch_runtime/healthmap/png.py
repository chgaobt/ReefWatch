"""Minimal PNG encoder on zlib + numpy, for hosts without Pillow (the Kria board).

8-bit RGB or RGBA, one IDAT chunk, filter type 0 (none) on every row. Valid PNG that any
decoder (Pillow, Unreal's IImageWrapper, browsers) reads; slightly larger than Pillow's
output because no row filter is tried.
"""
from __future__ import annotations

import struct
import zlib

import numpy as np

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _chunk(tag: bytes, data: bytes) -> bytes:
    return (struct.pack(">I", len(data)) + tag + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))


def encode_png_zlib(img: np.ndarray, level: int = 1) -> bytes:
    """(H, W, 3|4) uint8 -> PNG bytes."""
    a = np.ascontiguousarray(img, np.uint8)
    if a.ndim != 3 or a.shape[2] not in (3, 4):
        raise ValueError(f"expected (H, W, 3|4) uint8, got {a.shape}")
    h, w, c = a.shape
    raw = np.zeros((h, 1 + w * c), np.uint8)       # leading 0 = filter "none" per row
    raw[:, 1:] = a.reshape(h, w * c)
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 6 if c == 4 else 2, 0, 0, 0)
    return (PNG_SIGNATURE + _chunk(b"IHDR", ihdr)
            + _chunk(b"IDAT", zlib.compress(raw.tobytes(), level)) + _chunk(b"IEND", b""))


def decode_png_zlib(buf: bytes) -> np.ndarray:
    """Inverse of `encode_png_zlib` (filter 0 only); for tests on hosts without Pillow."""
    if buf[:8] != PNG_SIGNATURE:
        raise ValueError("not a PNG")
    off, idat = 8, b""
    w = h = c = None
    while off < len(buf):
        n, = struct.unpack_from(">I", buf, off)
        tag = buf[off + 4:off + 8]
        data = buf[off + 8:off + 8 + n]
        off += 12 + n
        if tag == b"IHDR":
            w, h, depth, ctype = struct.unpack_from(">IIBB", data)
            if depth != 8 or ctype not in (2, 6):
                raise ValueError("only 8-bit RGB/RGBA")
            c = 4 if ctype == 6 else 3
        elif tag == b"IDAT":
            idat += data
    raw = np.frombuffer(zlib.decompress(idat), np.uint8).reshape(h, 1 + w * c)
    if raw[:, 0].any():
        raise ValueError("row filters other than 0 are not supported")
    return raw[:, 1:].reshape(h, w, c).copy()
