"""ReefWatch board side: one-way CSG2 receiver, DPU inference and the health map on the KV260.

Needs only numpy. `reefwatch_runtime` (protocol + healthmap, both torch-free) must be importable:
it is found on sys.path, or next to this package in the repo layout (`board/` beside
`runtime/`), which is also the layout of the bundle `tools/make_bundle.py` builds.
"""
import os
import sys

__version__ = "1.0.0"

BOARD_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIM_ROOT = os.path.dirname(BOARD_DIR)


def _ensure_runtime_on_path():
    try:
        import reefwatch_runtime  # noqa: F401
    except ImportError:
        cand = os.path.join(SIM_ROOT, "runtime")
        if os.path.isdir(os.path.join(cand, "reefwatch_runtime")):
            sys.path.insert(0, cand)


_ensure_runtime_on_path()
