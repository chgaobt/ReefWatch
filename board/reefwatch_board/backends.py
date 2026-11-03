"""Inference backends behind one interface (mirrors runtime/reefwatch_runtime/backend.py).

    backend.predict_with_stats(rgb_uint8 (H, W, 3)) -> ((H, W) uint8 native class ids, stats3 | None)
    backend.timings  -> dict of the last call's stage times in ms (pre / dpu / post)

  none   no inference, a constant mask: proves the Ethernet pipe and measures receive fps
  vart   Vitis AI 3.5 VART on the KV260 DPU (int8 in, int8 full-resolution logits out)
  torch  PC only: the runtime's TorchBackend (the PC server's exact output), or with
         `emulate_dpu` the board's own int8 input + padding + argmax around a float forward

vart / xir / torch are imported lazily, so the package runs where they are absent.
"""
from __future__ import annotations

import glob
import os
import time

import numpy as np

from . import BOARD_DIR
from . import prepost as PP


class Backend:
    name = "base"
    num_classes = 7
    bleached_idx = 0

    def __init__(self):
        self.timings = {}

    def predict_with_stats(self, rgb):                     # pragma: no cover - interface
        raise NotImplementedError

    def warmup(self, width, height):
        self.predict_with_stats(np.zeros((height, width, 3), np.uint8))

    def describe(self) -> dict:
        return dict(backend=self.name, num_classes=self.num_classes)


class NoneBackend(Backend):
    """Every pixel = `const_id` (default the 7-class OTHER, health OTHER). No model, no stats."""

    name = "none"

    def __init__(self, num_classes=7, const_id=None):
        super().__init__()
        self.num_classes = int(num_classes)
        self.bleached_idx = PP.bleached_index(self.num_classes)
        self.const_id = self.num_classes - 1 if const_id is None else int(const_id)

    def predict_with_stats(self, rgb):
        t0 = time.perf_counter()
        h, w = rgb.shape[:2]
        ids = np.full((h, w), self.const_id, np.uint8)
        self.timings = dict(pre=0.0, dpu=0.0, post=(time.perf_counter() - t0) * 1000.0)
        return ids, None


# --------------------------------------------------------------------------- vart
VART_MISSING = """\
The Vitis AI Python bindings (`vart`, `xir`) are not importable here: {exc}

The `vart` backend runs on the KV260 only, with the Vitis AI 3.5 runtime installed for the
Python you are using. Check on the board:
  python3 --version                       (VART 3.5's bindings target the image's system Python, 3.10 on Kria Ubuntu 22.04)
  python3 -c "import vart, xir"           (must print nothing)
  ls /usr/lib/python3*/site-packages /usr/local/lib/python3*/dist-packages | grep -i -E "vart|xir"
  xdputil query                           (DPU visible? fingerprint matches DPUCZDX8G_ISA1_B4096?)
  sudo xmutil listapps                    (is a DPU app/overlay loaded?)
If the bindings live in another interpreter or a conda/venv, run this program with that one.
`--backend none` needs none of this and tests the Ethernet pipe and the health map meanwhile."""


#: the default model: the milestone checkpoint compiled at 512x768, which the 640x360 Unreal
#: frame pads to exactly as the PC server pads it (512x1024 pads 1.8x the area and agrees less with the PC server)
DEFAULT_XMODEL = "coral_milestone_7c_512x768.xmodel"


def find_xmodel(path=None):
    """--xmodel, else board/models/DEFAULT_XMODEL, else the first board/models/*.xmodel."""
    if path:
        return path
    models = os.path.join(BOARD_DIR, "models")
    if os.path.isfile(os.path.join(models, DEFAULT_XMODEL)):
        return os.path.join(models, DEFAULT_XMODEL)
    found = sorted(glob.glob(os.path.join(models, "*.xmodel")))
    if not found:
        raise FileNotFoundError(f"no --xmodel given and no *.xmodel in {models}")
    return found[0]


def load_pynq_dpu_overlay(bitfile, log=print):
    """`--pynq-dpu-overlay`: load a DPU-PYNQ overlay (e.g. `dpu.bit` of DPU-PYNQ's Vitis AI 3.5
    branch, a B4096 DPU on the KV260) in this process before VART opens the DPU.

    `pynq_dpu.DpuOverlay(bitfile)` programs the PL and copies the overlay's xclbin to
    /usr/lib/dpu.xclbin; its `check_vart_config()` makes /etc/vart.conf point VART there. The
    overlay object is kept for the life of the backend. Needs root and the PYNQ venv (see
    SETUP_UBUNTU.md); off by default, for boards whose DPU is loaded with `xmutil` instead."""
    try:
        from pynq_dpu import DpuOverlay
    except ImportError as exc:
        raise RuntimeError(
            f"--pynq-dpu-overlay needs the pynq_dpu package (DPU-PYNQ): {exc}. Run inside the PYNQ "
            f"venv as root (source /etc/profile.d/pynq_venv.sh), or load the DPU with xmutil and "
            f"drop the option. See SETUP_UBUNTU.md.") from None
    t0 = time.perf_counter()
    ov = DpuOverlay(bitfile)
    # check_vart_config is a module-level function in pynq_dpu.dpu (DPU-PYNQ 3.5), not a method:
    # it points /etc/vart.conf at the xclbin DpuOverlay just copied to /usr/lib/dpu.xclbin.
    # Without it VART opens whatever the conf names (on Kria Ubuntu: /run/media/mmcblk0p1/...).
    try:
        from pynq_dpu.dpu import check_vart_config
    except ImportError:
        check_vart_config = getattr(ov, "check_vart_config", None)
    if check_vart_config is not None:
        check_vart_config()
    log(f"[vart] DPU-PYNQ overlay {bitfile} loaded ({(time.perf_counter() - t0) * 1e3:.0f} ms)")
    return ov


def dpu_subgraphs(graph):
    """The DPU child subgraphs of an xir graph, in topological order (the standard VART recipe)."""
    root = graph.get_root_subgraph()
    return [s for s in root.toposort_child_subgraph()
            if s.has_attr("device") and s.get_attr("device").upper() == "DPU"]


class VartBackend(Backend):
    """One DPU runner on the xmodel's single DPU subgraph.

    Input/output shapes, layouts and fix points are read from the runner's tensors; nothing
    about the model size is hard-coded. Expected (and checked): one NHWC int8 input with 3
    channels, one NHWC int8 output whose spatial size divides the input's. The compiled
    R34-DAPPM builds output full-resolution logits (stride 1: argmax + crop, nothing else);
    a strided output (a build ending at the head conv) is upsampled by nearest neighbour.
    The fix2float CPU op after the DPU subgraph is not run: argmax works on the int8 values.
    """

    name = "vart"

    def __init__(self, xmodel=None, log=print, pynq_overlay=None):
        super().__init__()
        self.overlay = None
        if pynq_overlay:
            self.overlay = load_pynq_dpu_overlay(pynq_overlay, log)
        try:
            import vart
            import xir
        except ImportError as exc:
            raise RuntimeError(VART_MISSING.format(exc=exc)) from None
        self.xmodel = find_xmodel(xmodel)
        graph = xir.Graph.deserialize(self.xmodel)
        sgs = dpu_subgraphs(graph)
        if not sgs:
            raise RuntimeError(f"{self.xmodel}: no DPU subgraph (was it compiled for the DPU?)")
        if len(sgs) > 1:
            log(f"[vart] WARNING: {len(sgs)} DPU subgraphs; running only the first "
                f"({sgs[0].get_name()}). This program expects the single-subgraph build.")
        self.subgraph = sgs[0]
        self.runner = vart.Runner.create_runner(self.subgraph, "run")
        it = self.runner.get_input_tensors()
        ot = self.runner.get_output_tensors()
        if len(it) != 1 or len(ot) != 1:
            raise RuntimeError(f"expected 1 input and 1 output tensor, got {len(it)} / {len(ot)}")
        self.in_t, self.out_t = it[0], ot[0]
        self.in_dims = tuple(int(d) for d in self.in_t.dims)
        self.out_dims = tuple(int(d) for d in self.out_t.dims)
        self.in_fix = int(self.in_t.get_attr("fix_point"))
        self.out_fix = int(self.out_t.get_attr("fix_point"))
        n, in_h, in_w, in_c = self.in_dims
        _, out_h, out_w, out_c = self.out_dims
        if n != 1 or in_c != 3:
            raise RuntimeError(f"input tensor {self.in_dims}: expected 1 x H x W x 3 (NHWC)")
        if in_h % out_h or in_w % out_w or in_h // out_h != in_w // out_w:
            raise RuntimeError(f"output {self.out_dims} is not an integer stride of input {self.in_dims}")
        self.stride = in_h // out_h
        self.num_classes = out_c
        self.bleached_idx = PP.bleached_index(out_c)
        self.out_scale = 2.0 ** -self.out_fix
        self.inp = PP.Int8Input(in_h, in_w, self.in_fix)
        self.out = np.empty(self.out_dims, np.int8, order="C")
        self.with_stats = True

    def predict_with_stats(self, rgb):
        t0 = time.perf_counter()
        h0, w0 = rgb.shape[:2]
        h, w = self.inp.fill(rgb)
        t1 = time.perf_counter()
        job = self.runner.execute_async([self.inp.buf], [self.out])
        self.runner.wait(job)
        t2 = time.perf_counter()
        logits = self.out[0]
        ids = PP.logits_to_ids(logits, h, w, self.stride, h0, w0)
        stats = (PP.logit_stats(logits, h, w, self.stride, self.out_scale, self.bleached_idx)
                 if self.with_stats else None)
        t3 = time.perf_counter()
        self.timings = dict(pre=(t1 - t0) * 1e3, dpu=(t2 - t1) * 1e3, post=(t3 - t2) * 1e3)
        return ids, stats

    def describe(self) -> dict:
        return dict(backend=self.name, xmodel=self.xmodel, subgraph=self.subgraph.get_name(),
                    input=list(self.in_dims), input_fix_point=self.in_fix,
                    output=list(self.out_dims), output_fix_point=self.out_fix,
                    stride=self.stride, num_classes=self.num_classes)


# --------------------------------------------------------------------------- torch (PC)
class TorchBackend(Backend):
    """PC A/B backend.

    * default: `reefwatch_runtime.backend.TorchBackend` unchanged, i.e. exactly what the PC
      server computes (float, pad to the model multiple, bilinear output, argmax);
    * `emulate_dpu=True`: the board's own path with the DPU replaced by a float forward:
      the int8 LUT input of `Int8Input` (pad to `input_hw`, default 512x768 like the default
      xmodel) dequantised, the full network (its output is full resolution, like the DPU's),
      then `logits_to_ids` (crop + argmax). Only the INT8 arithmetic inside the net is missing.
    """

    name = "torch"

    def __init__(self, ckpt, device=None, emulate_dpu=False, input_hw=(512, 768), fix_point=5):
        super().__init__()
        from reefwatch_runtime.backend import TorchBackend as RT
        self.rt = RT(ckpt, device=device)
        self.torch = self.rt.torch
        self.num_classes = self.rt.num_classes
        self.bleached_idx = self.rt.bleached_idx
        self.emulate_dpu = bool(emulate_dpu)
        self.input_hw = tuple(int(v) for v in input_hw)
        if self.emulate_dpu:
            self.inp = PP.Int8Input(self.input_hw[0], self.input_hw[1], fix_point)

    def predict_with_stats(self, rgb):
        t0 = time.perf_counter()
        if not self.emulate_dpu:
            ids, stats = self.rt.predict_with_stats(rgb)
            self.timings = dict(pre=0.0, dpu=(time.perf_counter() - t0) * 1e3, post=0.0)
            return ids, stats
        torch = self.torch
        h0, w0 = rgb.shape[:2]
        h, w = self.inp.fill(rgb)
        x = torch.from_numpy(self.inp.as_float_nchw()).to(self.rt.device)
        t1 = time.perf_counter()
        with torch.inference_mode():
            logits = self.rt.model(x)[0].permute(1, 2, 0).float().cpu().numpy()   # (H, W, C)
        t2 = time.perf_counter()
        ids = PP.logits_to_ids(logits, h, w, 1, h0, w0)
        stats = PP.logit_stats(logits, h, w, 1, 1.0, self.bleached_idx)
        t3 = time.perf_counter()
        self.timings = dict(pre=(t1 - t0) * 1e3, dpu=(t2 - t1) * 1e3, post=(t3 - t2) * 1e3)
        return ids, stats

    def describe(self) -> dict:
        d = dict(backend=self.name, emulate_dpu=self.emulate_dpu, device=str(self.rt.device),
                 ckpt=self.rt.ckpt_path, num_classes=self.num_classes)
        if self.emulate_dpu:
            d["input_hw"] = list(self.input_hw)
        return d


def make_backend(name, xmodel=None, ckpt=None, device=None, emulate_dpu=False,
                 input_hw=(512, 768), log=print, pynq_overlay=None):
    if name == "none":
        return NoneBackend()
    if name == "vart":
        return VartBackend(xmodel, log=log, pynq_overlay=pynq_overlay)
    if name == "torch":
        if not ckpt:
            from reefwatch_runtime import config
            ckpt = str(config.DEFAULT_CKPT)
        return TorchBackend(ckpt, device=device, emulate_dpu=emulate_dpu, input_hw=input_hw)
    raise ValueError(f"unknown backend {name!r}")
