"""FAKE `vart` for PC tests of the board's vart backend. NOT the Vitis AI runtime; never ship it.

Mimics the VART 3.5 Python surface reefwatch_board.backends uses:
    Runner.create_runner(subgraph, mode) -> runner
    runner.get_input_tensors() / get_output_tensors() -> [xir.Tensor] (.name .dims .dtype get_attr)
    runner.execute_async([in_int8], [out_int8]) -> (job_id, 0);  runner.wait(job) -> 0
Measured on the real binding (Vitis AI 3.5 docker, CPU runner): execute_async returns a tuple and
wait returns 0; a uint8/float32 buffer ABORTS the process (glog CHECK); a wrong-shape or
non-contiguous int8 buffer is silently accepted. This fake is deliberately stricter: anything but a
C-contiguous int8 array of exactly the tensor's shape (and a writeable output) raises.

What computes the output (FAKE_VART_COMPUTE, default "auto"):
  replay   int8 logits recorded from the bit-exact DPU reference (VART CPU runner on the compiled
           subgraph, == vai_q), looked up by sha1 of the input buffer; files in FAKE_VART_REPLAY
           (os.pathsep-separated .npz with input_sha1[N], logits[N,H,W,C] int8)
  torch    the float milestone model on the dequantised input, logits re-quantised to the output
           fix_point. An approximation: on 20 sim frames its argmax agrees with the INT8 reference
           on ~92 % of pixels (board/tools/vitis_cpu_ref; INT8 changes ~8 % of pixels)
  pattern  a deterministic numpy function of the input (plumbing tests only)
  auto     replay, else torch, else error
FAKE_VART_DPU_MS=<ms> makes execute_async+wait take at least that long (rehearse board latency).
"""
import hashlib
import os
import sys
import time

import numpy as np

import xir

FAKE = True
CREATED = []                 # (subgraph name, mode) of every create_runner call, for tests
sys.stderr.write("[fake_vitis] FAKE vart loaded (test shim, no DPU); compute=%s\n"
                 % os.environ.get("FAKE_VART_COMPUTE", "auto"))


class _Replay:
    def __init__(self, paths):
        self.table = {}
        for p in [p for p in paths.split(os.pathsep) if p]:
            z = np.load(p)
            for h, lg in zip(z["input_sha1"], z["logits"]):
                self.table[str(h)] = lg

    def get(self, x):
        return self.table.get(hashlib.sha1(np.ascontiguousarray(x).tobytes()).hexdigest())


class _Torch:
    def __init__(self):
        from reefwatch_runtime import config
        from reefwatch_runtime.backend import TorchBackend
        self.rt = TorchBackend(os.environ.get("FAKE_VART_CKPT", str(config.DEFAULT_CKPT)),
                               device=os.environ.get("FAKE_VART_DEVICE", "cpu"))

    def __call__(self, x, in_fix, out_fix):
        torch = self.rt.torch
        t = torch.from_numpy(x.astype(np.float32).transpose(0, 3, 1, 2) * np.float32(2.0 ** -in_fix))
        with torch.inference_mode():
            y = self.rt.model(t.to(self.rt.device)).permute(0, 2, 3, 1).float().cpu().numpy()
        return np.clip(np.floor(y * 2.0 ** out_fix + 0.5), -128, 127).astype(np.int8)


def _pattern(x, out_shape):
    y = np.zeros(out_shape, np.int8)
    xs = x.astype(np.int16)
    for c in range(out_shape[-1]):
        y[..., c] = np.clip((xs[..., c % 3] >> 2) + 3 * c - 9, -128, 127)
    return y


def _check(bufs, tensors, what, writeable=False):
    if not isinstance(bufs, (list, tuple)) or len(bufs) != len(tensors):
        raise RuntimeError("fake vart: %s must be a list of %d arrays" % (what, len(tensors)))
    for b, t in zip(bufs, tensors):
        exp = tuple(t.dims)
        if not isinstance(b, np.ndarray):
            raise RuntimeError("fake vart: %s for %s is %s, not a numpy array" % (what, t.name, type(b)))
        if b.dtype != np.int8:
            raise RuntimeError("fake vart: %s for %s is %s; tensor is %s (the real binding aborts "
                               "the process on this)" % (what, t.name, b.dtype, t.dtype))
        if b.shape != exp:
            raise RuntimeError("fake vart: %s for %s has shape %s, tensor dims %s (the real binding "
                               "silently accepts this)" % (what, t.name, b.shape, list(exp)))
        if not b.flags.c_contiguous:
            raise RuntimeError("fake vart: %s for %s is not C-contiguous (the real binding silently "
                               "reads/writes the wrong bytes)" % (what, t.name))
        if writeable and not b.flags.writeable:
            raise RuntimeError("fake vart: %s for %s is read-only" % (what, t.name))


class Runner:
    def __init__(self, subgraph, mode):
        self.subgraph = subgraph
        self.mode = mode
        self._in = list(sorted(subgraph.get_input_tensors(), key=lambda t: t.name))
        self._out = list(sorted(subgraph.get_output_tensors(), key=lambda t: t.name))
        self._jobs = {}
        self._next = 0
        self.calls = 0
        self.compute_used = {}             # compute kind -> calls
        self._replay = _Replay(os.environ.get("FAKE_VART_REPLAY", ""))
        self._torch = None

    @staticmethod
    def create_runner(subgraph, mode=""):
        if not isinstance(subgraph, xir.Subgraph):
            raise TypeError("fake vart: create_runner needs a (fake) xir.Subgraph")
        dev = subgraph.get_attr("device") if subgraph.has_attr("device") else "?"
        if dev.upper() != "DPU":
            raise RuntimeError("fake vart: subgraph %s is on %s, not DPU" % (subgraph.get_name(), dev))
        runners = subgraph.get_attr("runner") if subgraph.has_attr("runner") else {}
        if mode not in runners:
            raise RuntimeError("fake vart: mode %r not in the subgraph's runner map %s" % (mode, runners))
        if mode != "run":
            sys.stderr.write("[fake_vitis] note: mode %r (the board uses 'run')\n" % mode)
        CREATED.append((subgraph.get_name(), mode))
        return Runner(subgraph, mode)

    def get_input_tensors(self):
        return list(self._in)

    def get_output_tensors(self):
        return list(self._out)

    def _compute(self, x):
        want = os.environ.get("FAKE_VART_COMPUTE", "auto")
        in_fix = int(self._in[0].get_attr("fix_point"))
        out_fix = int(self._out[0].get_attr("fix_point"))
        if want in ("auto", "replay"):
            y = self._replay.get(x)
            if y is not None:
                return "replay", y
            if want == "replay":
                raise RuntimeError("fake vart: input not in the replay table (FAKE_VART_REPLAY=%r)"
                                   % os.environ.get("FAKE_VART_REPLAY", ""))
        if want in ("auto", "torch"):
            try:
                if self._torch is None:
                    self._torch = _Torch()
                return "torch", self._torch(x, in_fix, out_fix)
            except ImportError as exc:
                if want == "torch":
                    raise
                raise RuntimeError("fake vart: no replay hit and torch unavailable (%s); set "
                                   "FAKE_VART_COMPUTE=pattern for plumbing tests" % exc)
        if want == "pattern":
            return "pattern", _pattern(x, tuple(self._out[0].dims))
        raise ValueError("fake vart: FAKE_VART_COMPUTE=%r" % want)

    def execute_async(self, inputs, outputs):
        t0 = time.perf_counter()
        _check(inputs, self._in, "input")
        _check(outputs, self._out, "output", writeable=True)
        how, y = self._compute(inputs[0])
        outputs[0][...] = y.reshape(outputs[0].shape)
        if how not in self.compute_used:
            sys.stderr.write("[fake_vitis] call %d: first output computed by %r\n" % (self.calls + 1, how))
        self.compute_used[how] = self.compute_used.get(how, 0) + 1
        self.calls += 1
        jid = self._next
        self._next += 1
        self._jobs[jid] = t0
        return (jid, 0)

    def wait(self, job, timeout=-1):
        jid = job[0] if isinstance(job, tuple) else int(job)
        t0 = self._jobs.pop(jid)
        ms = float(os.environ.get("FAKE_VART_DPU_MS", "0") or 0)
        left = ms / 1e3 - (time.perf_counter() - t0)
        if left > 0:
            time.sleep(left)
        return 0


class RunnerExt(Runner):
    pass


class TensorBuffer:                          # present in the real module; unused by the board
    pass
