"""Pre- and post-processing around the DPU, in numpy only. Shared by every backend that feeds
the DPU-shaped input (`vart`, and `torch --emulate-dpu` on the PC).

Contract of the compiled xmodel (board/models/TENSORS_*.md, from an xir dump; the int8 layout
and fix points were also checked against the quantizer's deploy-check dump):

  input   1 x H x W x 3 int8, NHWC, RGB, value = round((rgb/255 - mean) / std * 2**fix_point)
          with ImageNet mean/std; fix_point is the input tensor's attribute (5 for the 7-class
          R34-DAPPM builds). H x W comes from the xmodel (512x768 or 512x1024; never hard-coded).
  output  1 x H x W x C int8, NHWC, value * 2**-fix_point = float logit (fix_point 3). Full
          resolution: vai_q turned the network's final 4x bilinear resize into a transposed
          depthwise conv that runs on the DPU. The only CPU op left in the compiled graph is a
          fix2float, which is not needed: argmax runs on the int8 values directly.
          `logits_to_ids` also accepts a strided output (stride = H / output H) and then
          upsamples the argmax by nearest neighbour, for a build that ends at the head conv.

The frame (640x360 from Unreal) is reflect-padded bottom/right to H x W exactly as the PC
server pads to its multiple (`TorchBackend._logits`: reflect, or replicate if a pad would
reach the image size). The normalisation is a per-channel uint8 -> int8 lookup table, and the
padding is done after it in int8 (an elementwise map commutes with padding), so the ARM does
three table lookups and two slice copies per frame.
"""
from __future__ import annotations

import numpy as np

IMAGENET_MEAN = (0.485, 0.456, 0.406)          # = reefwatch_runtime.model.IMAGENET_MEAN
IMAGENET_STD = (0.229, 0.224, 0.225)

#: 7-class milestone head, standalone_train.SEVEN_NAMES order (= reefwatch_runtime.model)
SEVEN_NAMES = ("BLEACHED", "DEAD", "ALIVE", "ROCK", "OCEAN", "SAND", "OTHER")
HEALTH_NAMES = ("LIVE", "BLEACHED", "DEGRADED", "OTHER")
#: 7-class id -> health id; copied from reefwatch_runtime.model.SEVEN_TO_HEALTH (which imports
#: torch); tests/test_board.py pins the two together on the PC
SEVEN_TO_HEALTH = {"health": (1, 2, 0, 2, 3, 3, 3),    # DEAD + ROCK -> DEGRADED (default)
                   "readme": (1, 2, 0, 3, 3, 3, 3)}    # ROCK -> OTHER
#: mask colours (R, G, B, A) of the 4 health classes, index = health id: the HUD legend of
#: docs/PROTOCOL.md "Classes and palette"; copied from reefwatch_runtime.model.HEALTH_PALETTE_RGBA
#: (pinned by a test). `--reply` colours the mask with it, as the PC server does by default.
HEALTH_PALETTE_RGBA = ((0, 200, 80, 255), (255, 255, 255, 255), (200, 60, 40, 255),
                       (60, 60, 200, 255))
#: label-editor colours of the 7 native classes, SEVEN_NAMES order; copied from
#: reefwatch_runtime.model.SEVEN_HEX (pinned by a test). `--reply` colours the mask with them
#: for requests with WANT_NATIVE_CLASSES (docs/PROTOCOL.md "Classes and palette").
SEVEN_HEX = ("#f4ec75", "#ea7979", "#67d88d", "#6b4f3a", "#68b9ef", "#f38400", "#e591d6")
SEVEN_PALETTE_RGBA = tuple(tuple(int(h[i:i + 2], 16) for i in (1, 3, 5)) + (255,) for h in SEVEN_HEX)


def native_classes(num_classes):
    """(names, RGBA rows) of the model's own classes when they differ from health (the 7-class
    head), else None: WANT_NATIVE_CLASSES is then ignored and the mask stays health."""
    return (SEVEN_NAMES, SEVEN_PALETTE_RGBA) if num_classes == 7 else None


def to_health_table(num_classes, scheme="health"):
    """native id -> health id (uint8 LUT)."""
    if num_classes == 7:
        return np.asarray(SEVEN_TO_HEALTH[scheme], np.uint8)
    if num_classes == 4:
        return np.arange(4, dtype=np.uint8)
    raise ValueError(f"unsupported class count {num_classes} (4 or 7)")


def bleached_index(num_classes):
    return SEVEN_NAMES.index("BLEACHED") if num_classes == 7 else HEALTH_NAMES.index("BLEACHED")


# --------------------------------------------------------------------------- input
def quant_lut(fix_point):
    """(3, 256) int8: per channel, uint8 value -> quantized normalised input."""
    v = np.arange(256, dtype=np.float64) / 255.0
    scale = 2.0 ** int(fix_point)
    lut = np.empty((3, 256), np.int8)
    for c in range(3):
        q = np.round((v - IMAGENET_MEAN[c]) / IMAGENET_STD[c] * scale)
        lut[c] = np.clip(q, -128, 127).astype(np.int8)
    return lut


def pad_bottom_right(dst, h, w):
    """In place: `dst[:h, :w]` holds the image; fill the rest of `dst` (H, W, C) like
    `torch.nn.functional.pad(..., mode)` with mode reflect when both pads are smaller than the
    image (the PC server's rule), replicate otherwise."""
    H, W = dst.shape[:2]
    ph, pw = H - h, W - w
    reflect = ph < h and pw < w
    if pw:
        j = np.arange(w, W)
        cols = 2 * (w - 1) - j if reflect else np.full(pw, w - 1)
        dst[:h, w:] = dst[:h, cols]
    if ph:
        i = np.arange(h, H)
        rows = 2 * (h - 1) - i if reflect else np.full(ph, h - 1)
        dst[h:, :] = dst[rows, :]
    return dst


def nearest_resize(a, out_h, out_w):
    h, w = a.shape[:2]
    rows = np.minimum((np.arange(out_h) * h) // out_h, h - 1)
    cols = np.minimum((np.arange(out_w) * w) // out_w, w - 1)
    return a[rows[:, None], cols[None, :]]


class Int8Input:
    """Owns the 1 x H x W x 3 int8 input buffer of the DPU and fills it from an RGB frame.

    A frame larger than H x W (never sent by Unreal) is first shrunk by nearest-neighbour to
    fit, keeping the aspect ratio; `logits_to_ids` scales the ids back up to the frame size.
    """

    def __init__(self, in_h, in_w, fix_point, buffer=None):
        self.in_h, self.in_w = int(in_h), int(in_w)
        self.fix_point = int(fix_point)
        self.lut = quant_lut(self.fix_point)
        self.buf = buffer if buffer is not None else np.empty((1, self.in_h, self.in_w, 3), np.int8)
        assert self.buf.shape == (1, self.in_h, self.in_w, 3) and self.buf.dtype == np.int8, \
            (self.buf.shape, self.buf.dtype)

    def fill(self, rgb):
        """-> (h, w) of the image region inside the buffer (== frame size unless shrunk)."""
        rgb = np.asarray(rgb, np.uint8)
        h, w = rgb.shape[:2]
        if h > self.in_h or w > self.in_w:
            s = min(self.in_h / h, self.in_w / w)
            h, w = max(1, int(h * s)), max(1, int(w * s))
            rgb = nearest_resize(rgb, h, w)
        dst = self.buf[0]
        for c in range(3):
            dst[:h, :w, c] = self.lut[c][rgb[:, :, c]]
        pad_bottom_right(dst, h, w)
        return h, w

    def as_float_nchw(self):
        """The buffer as the float tensor the quantized graph saw (for the torch emulation)."""
        return self.buf.astype(np.float32).transpose(0, 3, 1, 2) * np.float32(2.0 ** -self.fix_point)


# --------------------------------------------------------------------------- output
def logits_to_ids(logits_hwc, h, w, stride=1, out_h=None, out_w=None):
    """Logits (Ho, Wo, C), int8 or float, NHWC -> (out_h, out_w) uint8 class ids.

    Crop to the image region first (the padding is bottom/right, so the crop is exact), then
    argmax over C. Argmax is invariant to the positive dequant scale, so int8 logits are used
    as they come. With stride 1 (the compiled R34-DAPPM builds) that is all; with stride s > 1
    the argmax grid is upsampled by nearest neighbour. `out_h, out_w` != (h, w) only when the
    frame was shrunk into the input buffer.
    """
    s = int(stride)
    hc, wc = -(-h // s), -(-w // s)
    a = np.asarray(logits_hwc)[:hc, :wc].argmax(-1).astype(np.uint8)
    out_h = h if out_h is None else out_h
    out_w = w if out_w is None else out_w
    if s == 1 and (out_h, out_w) == (h, w):
        return a
    rows = np.minimum((np.arange(out_h) * h // out_h) // s, hc - 1)
    cols = np.minimum((np.arange(out_w) * w // out_w) // s, wc - 1)
    return a[rows[:, None], cols[None, :]]


def logit_stats(logits_hwc, h, w, stride, scale, bleached_idx, sample_px=4):
    """[mean max softmax, mean entropy (nats), mean softmax of BLEACHED] over the image region.

    The PC server averages the same quantities over every pixel; here they are averaged over a
    grid of one sample per `sample_px` pixels each way (a softmax over 360x640x7 values is
    ~1.6 M exps, too much for the A53 per frame; 1/16 of them is plenty for a mean)."""
    s = int(stride)
    step = max(1, int(sample_px) // s)
    off = step // 2
    x = np.asarray(logits_hwc)[off:-(-h // s):step, off:-(-w // s):step].astype(np.float32)
    x *= np.float32(scale)
    x -= x.max(-1, keepdims=True)
    e = np.exp(x)
    p = e / e.sum(-1, keepdims=True)
    ent = -(p * np.log(np.maximum(p, 1e-8))).sum(-1)
    return np.array([p.max(-1).mean(), ent.mean(), p[..., bleached_idx].mean()], np.float32)
