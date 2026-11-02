"""Default paths and settings. Every path can be overridden by an environment variable or a CLI flag.

Layout assumed (relative, so the tree can move as a whole):

    reefWatchMilestone2/
      ReefWatch/segmentation/weights/MILESTONE_R34_DAPPM_7c.pt   <- model, referenced (90 MB, not copied)
      ReefWatchSim/
        data/healthmap/dem.npz, gt_bleached.npz                  <- health-map inputs, copied (2 MB)
        runtime/reefwatch_runtime/                               <- this package
"""
from __future__ import annotations

import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
RUNTIME_DIR = PACKAGE_DIR.parent
SIM_ROOT = RUNTIME_DIR.parent

DEFAULT_CKPT = Path(os.environ.get(
    "REEFWATCH_CKPT",
    SIM_ROOT.parent / "ReefWatch" / "segmentation" / "weights" / "MILESTONE_R34_DAPPM_7c.pt"))
DEFAULT_DEM = Path(os.environ.get("REEFWATCH_DEM", SIM_ROOT / "data" / "healthmap" / "dem.npz"))
DEFAULT_GT = Path(os.environ.get("REEFWATCH_GT", SIM_ROOT / "data" / "healthmap" / "gt_bleached.npz"))

#: a real 640x360 frame from an Unreal AUV flight; used by the fp16 gate and by the tests
TEST_FRAME = RUNTIME_DIR / "tests" / "data" / "auv_frame_640x360.png"

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 5556
#: ULiveSegComponent captures 640x360 by default. Every shape the client may send must be
#: pre-warmed: the first forward at an unseen shape can take seconds, past the client's 2 s
#: OFFLINE timeout.
DEFAULT_WARMUP = "640x360"
