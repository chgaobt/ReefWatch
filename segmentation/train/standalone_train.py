#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
 standalone_train.py -- train (and score) THE model, in one self-contained file
===============================================================================

ReefWatch note -- the milestone 1 model
---------------------------------------
The milestone 1 model (weights/MILESTONE_R34_DAPPM_7c.pt) is this same network with
a 7-class head (--classes 7), trained on CoralScapes (4-class labels used as partial
7-class labels) plus our 7-class labels of CoralNet survey photos, with rare-class
crop sampling.  The exact command is in train/run_train.sh and the README; the
original per-seed commands and logs are in results/logs/.  The text below is the
original documentation of the file, written for the 4-class CoralScapes model.

This single file trains the best coral-health segmentation model this project has
measured:

    ResNet-34 encoder  +  DAPPM context head on C5  +  FPN-lite decoder (64 ch)
    trained on 512 x 512 crops of Coralscapes V2, "v1" recipe, 20 epochs.

Internally the repository calls that model `fpn_resnet34_dappm`.  Published seed-0
result, whole-frame on the 473-image test split: **mIoU 0.6873, BLEACHED IoU 0.6570**.

-------------------------------------------------------------------------------
 Why this file exists
-------------------------------------------------------------------------------
There is a second script next door, `train_r34_dappm.py`, which *imports* the
repository's trainer (`scripts/v2/train_v2.py`) and hands it a fixed argument list.
That is a thin wrapper: to read what actually happens you have to open six more
files.  THIS file imports nothing from the repository at all.  Everything -- the
network, the data loader, the augmentation, the loss, the schedule, the metric --
is written out below, in order, with comments.

The two are checked against each other, not merely asserted to agree:

  * `check_standalone.py` (next to this file) loads the published checkpoint into
    BOTH definitions of the network and compares logits on a real image.
  * `--eval` here and `train_r34_dappm.py --eval` must print the same numbers.

-------------------------------------------------------------------------------
 Words you need, once
-------------------------------------------------------------------------------
 semantic segmentation : label every *pixel* of an image with a class.
 mIoU                  : mean Intersection-over-Union.  For one class,
                         IoU = correct pixels / (correct + missed + falsely added).
                         mIoU averages that over the classes.  1.0 is perfect.
 BLEACHED IoU          : the same score for the one rare class we care about
                         (~1.7 % of test pixels).  It moves much more than mIoU.
 crop                  : training does not use whole frames; it cuts a random
                         512x512 square out of each 1024x512 frame.
 encoder / backbone    : the ImageNet-pretrained convolutional trunk (ResNet-34)
                         that turns an image into feature maps at 1/4, 1/8, 1/16
                         and 1/32 of the input size.  Those four are called C2,
                         C3, C4, C5 below.
 FPN-lite decoder      : Feature Pyramid Network.  Project each of the four maps
                         to 64 channels, add them together coarse-to-fine, one
                         3x3 convolution, one 1x1 classifier, resize to input size.
 DAPPM                 : Deep Aggregation Pyramid Pooling Module (from the DDRNet
                         paper).  Five parallel views of C5 at different amounts of
                         average-pooling, merged in a cascade.  It is the ONLY
                         difference from the plain `fpn_resnet34` baseline, and it
                         is worth +0.023 mIoU / +0.061 BLEACHED IoU over 5 seeds.
 DPU                   : the fixed-function accelerator on the Kria KV260 FPGA
                         board this model is meant to run on.  Several odd-looking
                         choices below (2x upsample cascades, Conv->BN->ReLU order)
                         exist only because that chip is picky; they are explained
                         where they appear.
 epoch                 : one pass over the training set.
 AMP / fp16 autocast   : mixed precision -- most of the arithmetic runs in 16-bit
                         on the GPU for speed, with a "gradient scaler" to stop
                         small gradients underflowing to zero.

-------------------------------------------------------------------------------
 How to run it
-------------------------------------------------------------------------------
    # 0. wiring test, CPU only, ~2 min
    python standalone_train.py --smoke

    # 1. score the published checkpoint (GPU, ~1 min).  MUST print 0.6873 / 0.6570.
    python standalone_train.py --eval <published 4-class checkpoint>

    # 2. throughput gate: 2 epochs, prints seconds/epoch and GPU utilisation, exits
    python standalone_train.py --profile 2

    # 3. the real run (~33 min on an RTX 5050 laptop GPU when it is cool)
    python standalone_train.py --seed 0

Step 2 is not optional politeness: The project rules forbid starting training until the
GPU, not the data loader, is the bottleneck.  Step 3 refuses to start until a
passing gate for this exact configuration exists (pass `--no-gate` to override).

-------------------------------------------------------------------------------
 What it needs on disk
-------------------------------------------------------------------------------
    <segmentation>/data/coralscapes_v2/processed/{train,validation,test}_512/
           images/NNNN.png     1024 x 512 RGB
           health/NNNN.png     1024 x 512 uint8, values 0..3 and 255
           masks/NNNN.png      1024 x 512 uint8, the 95 fine class ids (optional)
    <segmentation>/data/coralscapes_v2/metadata/coralscapes_v2_index.csv
    <segmentation>/cache_v2/512/{train,validation,test}/{images.u8,health.u8,index.json}
           -- optional but strongly recommended; see the CACHE section.

Override the dataset root with $CORAL_DATA_V2 and the cache with $CORAL_V2_CACHE.

Allowed imports here: the Python standard library, torch, torchvision, timm,
numpy, PIL, yaml, tqdm.  NOTHING from this repository.  The assertion right after
the imports enforces that.
"""

# =============================================================================
# SECTION 0 -- imports, and the guard that keeps this file standalone
# =============================================================================
from __future__ import annotations

import sys

# ---- the standalone guard, part 1: what was already loaded ------------------
# If any of these modules is pulled in by THIS file, the file is not self-contained
# any more and the whole point of it is gone.  The snapshot is taken here, on the
# first executable line, BEFORE this file imports anything else, so the check below
# can tell "standalone_train.py imported repository code" (a bug) apart from "the
# process that imported standalone_train.py had already imported repository code"
# (which happens in a pytest session that also exercises the wrapper, and says
# nothing about this file).
_FORBIDDEN = ("coral", "v2_core", "v2_common", "common", "ctx_models", "vitis_defs",
              "dappm", "train_v2", "zoo_fpn", "gpu_profile")


def _repo_modules():
    return {m for m in sys.modules
            if m in _FORBIDDEN or any(m.startswith(f + ".") for f in _FORBIDDEN)}


#: repository modules the host process already had when this file started loading
_PRELOADED_REPO_MODULES = _repo_modules()

import argparse                                      # noqa: E402
import csv                                           # noqa: E402
import json                                          # noqa: E402
import math                                          # noqa: E402
import os                                            # noqa: E402
import random                                        # noqa: E402
import subprocess                                    # noqa: E402
import threading                                     # noqa: E402
import time                                          # noqa: E402
from pathlib import Path                             # noqa: E402

import numpy as np                                   # noqa: E402
import torch                                         # noqa: E402
import torch.nn as nn                                # noqa: E402
import torch.nn.functional as F                      # noqa: E402
from PIL import Image                                # noqa: E402
from torch.utils.data import DataLoader, Dataset     # noqa: E402


# ---- the standalone guard, part 2: the assertion ----------------------------
def _assert_standalone():
    """Raise if THIS file caused any repository module to be imported.

    Run as a script (the normal case) nothing is preloaded, so this is the literal
    "no repository module is in sys.modules" check.
    """
    introduced = sorted(_repo_modules() - _PRELOADED_REPO_MODULES)
    if introduced:
        raise ImportError(
            "standalone_train.py imported repository code: %r. This file must depend on "
            "nothing but the standard library, torch, torchvision, timm, numpy, PIL, yaml "
            "and tqdm." % (introduced,))
    return sorted(_PRELOADED_REPO_MODULES)


_assert_standalone()


# =============================================================================
# SECTION 1 -- where things live, and the label ontology
# =============================================================================
HERE = Path(__file__).resolve().parent          # .../segmentation/train
REPO = HERE.parent                              # .../segmentation

#: dataset root.  $CORAL_DATA_V2 wins, then the in-repo copy.  (The repository's
#: own loader resolves it the same way; there is no Hugging Face *cache* path at
#: this stage -- `scripts/v2/download_v2.py` pulls `josauder/coralscapesV2` from
#: the Hub and `decode_v2.py` writes the PNG tree above, which is what everything
#: downstream, including this file, reads.)
DATA_ROOT = Path(os.environ.get("CORAL_DATA_V2") or (REPO / "data" / "coralscapes_v2"))
INDEX_CSV = DATA_ROOT / "metadata" / "coralscapes_v2_index.csv"

#: decoded cache directory (see SECTION 3).  $CORAL_V2_CACHE wins, then <segmentation>/cache_v2.
DEFAULT_CACHE = Path(os.environ.get("CORAL_V2_CACHE") or (REPO / "cache_v2"))

#: where checkpoints and run logs go by default
DEFAULT_OUT = REPO / "runs"

IGNORE_ID = 255                                  # pixels excluded from loss AND metric
NUM_CLASSES = 4
HEALTH_NAMES = ["LIVE", "BLEACHED", "DEGRADED", "OTHER"]

# -----------------------------------------------------------------------------
# The optional 7-class head (--classes 7).  Default runs never touch any of this.
# -----------------------------------------------------------------------------
# Only the 1x1 classifier's output count changes (64 -> 7 instead of 64 -> 4), so the
# network stays exactly as DPU-legal as the 4-class one.  Order = the labelling kit's
# hotkey order (labeling_kit/core.py FINE_ORDER), so 7-class id = export fine id - 1.
SEVEN_NAMES = ["BLEACHED", "DEAD", "ALIVE", "ROCK", "OCEAN", "SAND", "OTHER"]
#: export `masks/` fine id -> 7-class id.  Exports (labeling_kit) write 0 = UNLABELED,
#: 1..7 = SEVEN_NAMES in order, 255 = IGNORE; 0, 255 and anything else -> IGNORE.
FINE7_LUT = np.full(256, IGNORE_ID, dtype=np.uint8)
FINE7_LUT[1:8] = np.arange(7, dtype=np.uint8)
#: In a 7-class run, CoralScapes pixels (4 health classes) carry a SET of 7-class labels
#: and are trained with the partial-label loss (`partial_label_loss`).  Their targets are
#: coded PARTIAL_BASE + health id, so they never collide with exact ids 0..6 or 255.
#:   LIVE     -> {ALIVE}
#:   BLEACHED -> {BLEACHED}
#:   DEGRADED -> {DEAD, ROCK}  CoralScapes' DEGRADED = '* dead' corals, 'algae covered *'
#:               corals, 'algae covered substrate' and 'rubble' (configs/health_label_
#:               mapping_v2.yaml, v1 "Option A").  Dead skeleton is DEAD; rubble and
#:               algae-covered hard substrate with no colony shape are ROCK by the kit's
#:               rules, so both are allowed and the loss does not pick between them.
#:   OTHER    -> {ROCK, SAND, OCEAN, OTHER}  ('hard substrate', 'sand', 'water column',
#:               'water surface', fish, sponge, seagrass, transect gear, ...)
PARTIAL_BASE = 16
PARTIAL_SETS = {"LIVE": ("ALIVE",), "BLEACHED": ("BLEACHED",),
                "DEGRADED": ("DEAD", "ROCK"), "OTHER": ("ROCK", "SAND", "OCEAN", "OTHER")}
HEALTH_TO_PARTIAL_LUT = np.full(256, IGNORE_ID, dtype=np.uint8)
for _h in range(len(HEALTH_NAMES)):
    HEALTH_TO_PARTIAL_LUT[_h] = PARTIAL_BASE + _h
SEVEN_MAPPING_BLOCK = {
    "version": 1,
    "classes": list(SEVEN_NAMES),
    "export_fine_ids": {i + 1: n for i, n in enumerate(SEVEN_NAMES)},
    "coralscapes_partial_sets": {k: list(v) for k, v in PARTIAL_SETS.items()},
    "_source": "standalone_train.py --classes 7 (labeling_kit/core.py FINE_NAMES order)",
}

# ImageNet normalisation -- the encoder was pretrained with these, so the input has
# to be presented the same way.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
#: ImageNet mean expressed as uint8 -- the colour used to pad a crop that falls off
#: the edge of a frame.  (The v1 recipe never pads, but the constant is kept so the
#: arithmetic below is the same one the repository uses.)
MEAN_PIX = np.array([124, 116, 104], dtype=np.uint8)

# -----------------------------------------------------------------------------
# The 95 -> 4 label mapping, embedded.
# -----------------------------------------------------------------------------
# Source: configs/health_label_mapping_v2.yaml, `version: 1` ("V2 ontology, first
# release"), generated by scripts/v2/make_mapping_v2.py from the id2label.json that
# ships with the `josauder/coralscapesV2` dataset.  It is copied in here as a dict
# literal on purpose: this file must run with no dependency on anything in the
# repository, including its configs.  If that YAML is ever regenerated, this table
# has to be updated by hand -- `check_standalone.py` compares the two and fails if
# they have drifted.
#
# Note on when it is actually used: `decode_v2.py` already wrote the mapped 4-class
# label PNGs into `health/`, so the normal training path reads those directly and
# never applies this table at run time.  It is applied only when the `health/` maps
# are missing and the fine `masks/` have to be collapsed on the fly, and it is
# embedded in every checkpoint so a checkpoint always carries its own ontology.
SUPERCLASS_IDS = {"LIVE": 0, "BLEACHED": 1, "DEGRADED": 2, "OTHER": 3, "IGNORE": 255}
FINE_TO_SUPER = {
    0: "IGNORE", 1: "OTHER", 2: "LIVE", 3: "BLEACHED", 4: "DEGRADED",
    5: "DEGRADED", 6: "DEGRADED", 7: "DEGRADED", 8: "DEGRADED", 9: "DEGRADED",
    10: "DEGRADED", 11: "DEGRADED", 12: "DEGRADED", 13: "OTHER", 14: "IGNORE",
    15: "LIVE", 16: "BLEACHED", 17: "DEGRADED", 18: "LIVE", 19: "BLEACHED",
    20: "DEGRADED", 21: "OTHER", 22: "IGNORE", 23: "OTHER", 24: "OTHER",
    25: "OTHER", 26: "LIVE", 27: "BLEACHED", 28: "DEGRADED", 29: "LIVE",
    30: "BLEACHED", 31: "DEGRADED", 32: "OTHER", 33: "LIVE", 34: "DEGRADED",
    35: "LIVE", 36: "BLEACHED", 37: "DEGRADED", 38: "OTHER", 39: "OTHER",
    40: "OTHER", 41: "LIVE", 42: "LIVE", 43: "BLEACHED", 44: "DEGRADED",
    45: "LIVE", 46: "BLEACHED", 47: "DEGRADED", 48: "LIVE", 49: "BLEACHED",
    50: "DEGRADED", 51: "LIVE", 52: "BLEACHED", 53: "DEGRADED", 54: "LIVE",
    55: "OTHER", 56: "LIVE", 57: "BLEACHED", 58: "DEGRADED", 59: "LIVE",
    60: "BLEACHED", 61: "DEGRADED", 62: "LIVE", 63: "BLEACHED", 64: "DEGRADED",
    65: "LIVE", 66: "BLEACHED", 67: "DEGRADED", 68: "DEGRADED", 69: "OTHER",
    70: "OTHER", 71: "OTHER", 72: "OTHER", 73: "LIVE", 74: "BLEACHED",
    75: "DEGRADED", 76: "LIVE", 77: "BLEACHED", 78: "OTHER", 79: "LIVE",
    80: "BLEACHED", 81: "DEGRADED", 82: "LIVE", 83: "BLEACHED", 84: "DEGRADED",
    85: "LIVE", 86: "BLEACHED", 87: "DEGRADED", 88: "OTHER", 89: "OTHER",
    90: "OTHER", 91: "LIVE", 92: "DEGRADED", 93: "OTHER", 94: "OTHER", 95: "LIVE",
}
#: the same table as the 256-entry lookup array the loader would use: fine id -> 0..3,
#: everything unlisted (and everything mapped to IGNORE) -> 255.
FINE_TO_HEALTH_LUT = np.full(256, IGNORE_ID, dtype=np.uint8)
for _fid, _sup in FINE_TO_SUPER.items():
    FINE_TO_HEALTH_LUT[_fid] = SUPERCLASS_IDS[_sup]

#: what gets stored in a checkpoint's `mapping` field, so the checkpoint is
#: self-describing exactly as the repository's checkpoints are.
MAPPING_BLOCK = {
    "version": 1,
    "dataset": "josauder/coralscapesV2",
    "num_fine_classes": 95,
    "fine_void_id": 0,
    "superclasses": {
        "LIVE": {"id": 0, "color": [40, 200, 80]},
        "BLEACHED": {"id": 1, "color": [255, 255, 255]},
        "DEGRADED": {"id": 2, "color": [110, 55, 30]},
        "OTHER": {"id": 3, "color": [70, 110, 190]},
        "IGNORE": {"id": 255, "color": [0, 0, 0]},
    },
    "mapping": dict(FINE_TO_SUPER),
    "_source": "configs/health_label_mapping_v2.yaml v1 (embedded copy; the prose "
               "`notes`, `ambiguous_classes` and `group_sizes` blocks are not carried)",
}


# =============================================================================
# SECTION 2 -- the published recipe, as constants
# =============================================================================
# These are the exact values the published run used; they were read back out of
# artifacts_v2/models/swp_r34_dappm_c512_s0.pt["args"].  They are constants rather
# than flags because changing one means you are no longer reproducing this model.
RECIPE = {
    "model": "fpn_resnet34_dappm",   # ResNet-34 + DAPPM on C5 + FPN-lite decoder
    "target": "health",              # 4 superclasses, not the 95 fine ids
    "res": "512",                    # the 1024x512 copies of the frames
    "eval_res": "512",               # validation and test are scored on the same grid
    "crop": 512,                     # 512x512 random crops for training
    "bs": 16,                        # batch size
    "epochs": 20,
    "min_epochs": 15,                # early stopping cannot fire before this...
    "patience": 20,                  # ...and needs this many bad epochs (so: never)
    "lr": 1e-3,
    "wd": 1e-4,                      # AdamW weight decay
    "crops_per_image": 4,            # each image is visited 4x per epoch
    "workers": 6,                    # DataLoader worker processes
    "aug": "v1",                     # scale 0.75-1.5, random crop, hflip, brightness
    "sched": "onecycle",             # OneCycleLR, pct_start 0.1
    "loss": "ce",                    # plain cross-entropy...
    "dice_w": 1.0,                   # ...so the Dice weight is carried but UNUSED
    "class_weights": "none",         # unweighted CE (see the note in main())
    "pretrained": "imagenet",
    "freeze_bn": False,
    "branch_ch": 96,                 # channels per DAPPM branch
    "dch": 64,                       # FPN decoder channels
}

#: the profile gate's bar, from the project training rules (rule 1): do not start a run until the GPU
#: is the bottleneck.
GATE_UTIL = 80.0


# =============================================================================
# SECTION 3 -- data: split lists, the decoded cache, and PNG fallback
# =============================================================================
# THE DECODED CACHE, and why it exists
# ------------------------------------
# Training is data-loader bound, not GPU bound, if every crop costs a PNG decode:
# decoding a 1024x512 image *and* its label map to keep a 512x512 square is ~54 ms
# of CPU per item, against ~2 ms of GPU work.  On 2026-09-14 that made an A100 run
# at exactly a laptop's speed.  The fix is to decode every image ONCE into flat
# uint8 files and then memory-map them, so a crop is a numpy slice.
#
# Cache layout (built by scripts/v2/build_cache.py; re-implemented read-only here):
#
#     <cache>/<res>/<split>/images.u8    (N, H, W, 3) uint8, C-contiguous, no header
#     <cache>/<res>/<split>/health.u8    (N, H, W)    uint8, 0..3 and 255
#     <cache>/<res>/<split>/masks.u8     (N, H, W)    uint8, the 95 fine ids (optional)
#     <cache>/<res>/<split>/index.json   {"ids": [...], "height": H, "width": W,
#                                         "arrays": {"images": {"file","dtype","shape"},
#                                                    "health": {...}},
#                                         "complete": true, ...}
#
# `ids[r]` is the dataset id stored at row r, so id -> row is a dict lookup.  The
# files are raw dumps: np.memmap(file, dtype, mode="r", shape=tuple(shape)) is the
# whole reader.  "complete": false means the build was interrupted -- ignore it.
# The pixels are byte-identical to the PNGs; nothing is mapped or rescaled here.

def read_split_ids(split):
    """Dataset ids belonging to `split`, in the order the index CSV lists them.

    The repository reads the same CSV with pandas; the standard-library `csv`
    module gives the same rows in the same order, which is all that matters.
    Falls back to the cache index, then to globbing the PNG directory, so the file
    still works on a machine that has the data but not the metadata CSV.
    """
    if INDEX_CSV.is_file():
        ids = []
        with INDEX_CSV.open("r", newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if row["split"] == split:
                    ids.append(int(row["idx"]))
        if ids:
            return ids
    cache_index = DEFAULT_CACHE / RECIPE["res"] / split / "index.json"
    if cache_index.is_file():
        meta = json.loads(cache_index.read_text(encoding="utf-8"))
        return [int(i) for i in meta["ids"]]
    d = png_dir(split, RECIPE["res"], "images")
    ids = sorted(int(p.stem) for p in d.glob("*.png"))
    if not ids:
        raise SystemExit(
            "cannot find the %s split: no %s, no cache index, no PNGs in %s"
            % (split, INDEX_CSV, d))
    return ids


def png_dir(split, res, kind):
    """`decode_v2.py`'s on-disk layout: <root>/processed/<split>[_<res>]/<kind>/."""
    sub = split if str(res) == "full" else f"{split}_{res}"
    return DATA_ROOT / "processed" / sub / kind


class SplitCache:
    """Read-only view of one split's decoded uint8 arrays.

    Memory maps are opened lazily and per process: a DataLoader worker is a
    separate process, and a memmap object cannot be pickled into one.  So the
    dataset stores the cache *directory* and each worker builds its own
    `SplitCache` on first use.
    """

    def __init__(self, root, split, res="512"):
        self.dir = Path(root) / str(res) / split
        self.meta = json.loads((self.dir / "index.json").read_text(encoding="utf-8"))
        self.height, self.width = int(self.meta["height"]), int(self.meta["width"])
        self.ids = [int(i) for i in self.meta["ids"]]
        self.row = {i: r for r, i in enumerate(self.ids)}
        self._mm = {}

    def has(self, kind):
        return kind in self.meta["arrays"]

    def get(self, kind, idx):
        """One whole image or label map, as a read-only memmap view (no copy)."""
        mm = self._mm.get(kind)
        if mm is None:
            a = self.meta["arrays"][kind]
            mm = np.memmap(self.dir / a["file"], dtype=a["dtype"], mode="r",
                           shape=tuple(a["shape"]))
            self._mm[kind] = mm
        return mm[self.row[idx]]


def cache_ok(root, split, res="512", kinds=("images", "health")):
    """True only if the cache really holds complete, correctly sized arrays."""
    if not root:
        return False
    d = Path(root) / str(res) / split
    p = d / "index.json"
    if not p.is_file():
        return False
    try:
        meta = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not meta.get("complete"):
        return False
    for k in kinds:
        a = meta["arrays"].get(k)
        if a is None or not (d / a["file"]).is_file():
            return False
        want = int(np.prod(a["shape"])) * np.dtype(a["dtype"]).itemsize
        if (d / a["file"]).stat().st_size != want:
            return False
    return True


_OPEN_CACHES: dict = {}


def open_cache(root, split, res="512", kinds=("images", "health")):
    """Process-local, memoised `SplitCache`, or None when the cache is unusable."""
    if not root:
        return None
    key = (str(root), split, str(res))
    if key not in _OPEN_CACHES:
        _OPEN_CACHES[key] = (SplitCache(root, split, res)
                             if cache_ok(root, split, res, kinds) else None)
    return _OPEN_CACHES[key]


def read_array(split, idx, res, kind, cache):
    """One image (`kind="images"`) or label map (`"health"` / `"masks"`).

    Straight out of the cache when there is one, otherwise a PIL decode of the PNG.
    Identical pixels either way -- the cache was built by decoding these same files.
    """
    c = open_cache(cache, split, res, kinds=(kind,))
    if c is not None and c.has(kind):
        return c.get(kind, idx)
    p = png_dir(split, res, kind) / f"{idx:04d}.png"
    if kind == "health" and not p.is_file():
        # No pre-mapped 4-class map on disk: collapse the 95 fine ids ourselves with
        # the embedded table.  This is the one place FINE_TO_HEALTH_LUT is applied.
        fine = read_array(split, idx, res, "masks", cache)
        return FINE_TO_HEALTH_LUT[np.asarray(fine)]
    with Image.open(p) as im:
        return np.array(im.convert("RGB") if kind == "images" else im)


# =============================================================================
# SECTION 4 -- augmentation: the "v1" recipe
# =============================================================================
# The v1 recipe, in full, and in this order:
#
#   1. random scale   s ~ Uniform(0.75, 1.5); the frame is resized to
#                     (max(crop, int(H*s)), max(crop, int(W*s)))
#   2. random crop    a `crop` x `crop` square at a uniformly random position in
#                     the scaled frame
#   3. horizontal flip with probability 0.5
#   4. brightness     with probability 0.5: gain a ~ U(0.8, 1.2), bias b ~ U(-20, 20),
#                     pixel -> clip(pixel * a + b, 0, 255)
#
# That is all.  No rotation, no colour cast, no noise -- those belong to the
# "improved" recipe, which measured WORSE on this dataset, which is why the plain
# v1 recipe is the one kept (the plain v1 recipe beat every tuned variant).
#
# TWO IMPLEMENTATIONS, same random draws
# --------------------------------------
# `aug_v1_slow` is the literal reading: resize the whole 1024x512 frame, then cut a
# 512x512 square out of it.  Most of that resize is thrown away.
#
# `aug_v1_fast` (the default, and what the published run used) resamples ONLY the
# source rectangle the crop comes from, via PIL's `box=` argument.  It draws s, y, x
# in the same order from the same generator, so it sees exactly the same scale and
# the same crop position; the brightness step becomes a 256-entry lookup table,
# which is bit-exact.  The one approximation is that PIL accumulates its filter taps
# from a different origin, so the two differ by at most ~2 grey levels
# (measured: mean 0.0034/255, max 3; 0.011 % of label pixels differ).  In exchange
# the loader costs 4 ms/item instead of 19, which is the difference between a GPU at
# 25 % and a GPU at 97 %.
def v1_crop_box(rng, H, W, crop):
    """The scale and crop-position draws, factored out so both implementations
    consume the identical random stream: `s`, then `y`, then `x`.

    Returns (box, nh, nw, y, x):
      box       (left, upper, right, lower) in SOURCE-image coordinates -- the
                rectangle the output crop is resampled from (the fast path).
      nh, nw    the size the whole frame would be resized to (the slow path).
      y, x      the crop origin inside that resized frame (the slow path).
    """
    s = rng.uniform(0.75, 1.5)
    nh, nw = max(crop, int(H * s)), max(crop, int(W * s))
    y, x = int(rng.integers(0, nh - crop + 1)), int(rng.integers(0, nw - crop + 1))
    sy, sx = H / nh, W / nw
    box = (x * sx, y * sy, min((x + crop) * sx, W), min((y + crop) * sy, H))
    return box, nh, nw, y, x


def _brightness_lut(img, a, b):
    """Exactly clip(img.astype(float32) * a + b, 0, 255).astype(uint8), as one
    256-entry table lookup instead of three full-image float passes."""
    v = np.arange(256, dtype=np.float32)
    return np.clip(v * a + b, 0, 255).astype(np.uint8)[img]


def aug_v1_fast(img, msk, rng, crop):
    """The v1 recipe, resampling only the window the crop comes from."""
    H, W = msk.shape
    box, _, _, _, _ = v1_crop_box(rng, H, W, crop)
    # BILINEAR for the image (smooth interpolation), NEAREST for the label map
    # (a label must never be averaged with its neighbour into a third class).
    img = np.array(Image.fromarray(np.ascontiguousarray(img)).resize(
        (crop, crop), Image.BILINEAR, box=box))
    msk = np.array(Image.fromarray(np.ascontiguousarray(msk)).resize(
        (crop, crop), Image.NEAREST, box=box))
    if rng.random() < 0.5:
        img, msk = img[:, ::-1], msk[:, ::-1]
    if rng.random() < 0.5:
        a, b = rng.uniform(0.8, 1.2), rng.uniform(-20, 20)
        img = _brightness_lut(img, a, b)
    return img, msk


def aug_v1_slow(img, msk, rng, crop):
    """The v1 recipe, literally: resize the whole frame, then crop out of it."""
    H, W = msk.shape
    _, nh, nw, y, x = v1_crop_box(rng, H, W, crop)
    img = np.array(Image.fromarray(np.ascontiguousarray(img)).resize((nw, nh), Image.BILINEAR))
    msk = np.array(Image.fromarray(np.ascontiguousarray(msk)).resize((nw, nh), Image.NEAREST))
    img, msk = img[y:y + crop, x:x + crop], msk[y:y + crop, x:x + crop]
    if rng.random() < 0.5:
        img, msk = img[:, ::-1], msk[:, ::-1]
    if rng.random() < 0.5:
        a, b = rng.uniform(0.8, 1.2), rng.uniform(-20, 20)
        img = np.clip(img.astype(np.float32) * a + b, 0, 255).astype(np.uint8)
    return img, msk


def aug_v1_fast_at(img, msk, rng, crop, cy, cx):
    """--rare-crop-frac only: the v1 fast recipe with the crop window forced to contain
    source pixel (cy, cx). Same draws in the same order as aug_v1_fast (scale, y, x, flip,
    brightness); only the y / x ranges shrink to the positions whose window holds the pixel."""
    H, W = msk.shape
    s = rng.uniform(0.75, 1.5)
    nh, nw = max(crop, int(H * s)), max(crop, int(W * s))
    py, px = min(int(cy * nh / H), nh - 1), min(int(cx * nw / W), nw - 1)
    y = int(rng.integers(max(0, py - crop + 1), min(nh - crop, py) + 1))
    x = int(rng.integers(max(0, px - crop + 1), min(nw - crop, px) + 1))
    sy, sx = H / nh, W / nw
    box = (x * sx, y * sy, min((x + crop) * sx, W), min((y + crop) * sy, H))
    img = np.array(Image.fromarray(np.ascontiguousarray(img)).resize(
        (crop, crop), Image.BILINEAR, box=box))
    msk = np.array(Image.fromarray(np.ascontiguousarray(msk)).resize(
        (crop, crop), Image.NEAREST, box=box))
    if rng.random() < 0.5:
        img, msk = img[:, ::-1], msk[:, ::-1]
    if rng.random() < 0.5:
        a, b = rng.uniform(0.8, 1.2), rng.uniform(-20, 20)
        img = _brightness_lut(img, a, b)
    return img, msk


#: --rare-crop-frac default classes: the 7-class ids that are rare in the Sonnet exports
RARE_DEFAULT = "DEAD,SAND,OCEAN,ALIVE"


def normalize(img_uint8):
    """uint8 HxWx3 -> float32 3xHxW tensor, ImageNet-normalised."""
    x = img_uint8.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x.transpose(2, 0, 1).copy())


class CoralscapesV2Crops(Dataset):
    """Training set: random 512x512 crops of the 1024x512 frames.

    `crops_per_image` is implemented the way the repository does it -- by repeating
    the id list, so one "epoch" is 4 passes over 1790 images = 7160 items, each of
    which draws its own random crop.

    NOTE ON RANDOMNESS.  Each item creates `np.random.default_rng()` with no seed,
    i.e. seeded from operating-system entropy.  This is deliberate in the original
    and is copied here: with `persistent_workers=True` a per-epoch seed update on
    the main process would never reach the workers.  The practical consequence is
    that `--seed` does NOT make a run bit-reproducible: it fixes the weight
    initialisation, the shuffle order and the CUDA kernels' seeds, but not the
    augmentation draws.  Two runs at the same seed have landed 0.014 mIoU apart.
    """

    def __init__(self, split, ids, crop, res="512", cache=None, fast_aug=True,
                 crops_per_image=1, label_lut=None):
        self.split, self.res, self.crop = split, res, crop
        self.ids = list(ids) * int(crops_per_image)
        self.cache = str(cache) if cache else None
        self.aug = aug_v1_fast if fast_aug else aug_v1_slow
        # 7-class runs only: health id -> partial-label code (HEALTH_TO_PARTIAL_LUT),
        # applied to the crop after augmentation (nearest resampling commutes with it).
        self.label_lut = label_lut

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        idx = self.ids[i]
        rng = np.random.default_rng()
        img = read_array(self.split, idx, self.res, "images", self.cache)
        msk = read_array(self.split, idx, self.res, "health", self.cache)
        img, msk = self.aug(img, msk, rng, self.crop)
        if self.label_lut is not None:
            msk = self.label_lut[msk]
        return (normalize(np.ascontiguousarray(img)),
                torch.from_numpy(np.ascontiguousarray(msk).astype(np.int64)),
                idx)


def _load_export_split(root, split, kind="health", label_lut=None):
    """(images, label maps, names) of one split of an export written by
    data/EXPORT_FORMAT.md: <root>/processed/<split>_512/{images,health}/*.png,
    long side 1024, padded with IGNORE to multiples of 256 (sizes vary per photo).
    7-class runs read kind="masks" (the fine ids) through label_lut=FINE7_LUT."""
    d = Path(root) / "processed" / f"{split}_512"
    names = sorted(q.name for q in (d / "images").glob("*.png"))
    imgs = [np.asarray(Image.open(d / "images" / n).convert("RGB")) for n in names]
    msks = [np.asarray(Image.open(d / kind / n)) for n in names]
    if label_lut is not None:
        msks = [label_lut[m] for m in msks]
    return imgs, msks, names


class ExtraCrops(Dataset):
    """Extra training photos (an export directory), decoded ONCE into RAM so the
    loader stays as fast as the memmap cache; same v1 augmentation and the same
    crops_per_image as CoralScapes. Only built when --extra-train is given."""

    def __init__(self, root, crop, fast_aug=True, crops_per_image=1, split="train",
                 kind="health", label_lut=None, rare_frac=0.0, rare_ids=()):
        self.imgs, self.msks, self.names = _load_export_split(root, split, kind, label_lut)
        self.crop = crop
        self.aug = aug_v1_fast if fast_aug else aug_v1_slow
        self.ids = list(range(len(self.imgs))) * int(crops_per_image)
        # --rare-crop-frac (default off): with probability rare_frac an item's crop is centred
        # on a pixel of a rare class present in that photo. The class is drawn among those
        # present with weight 1 / (its pixel share in this export), the pixel uniformly from a
        # fixed subsample of its pixels (<= 4096 per class per photo). Off = no extra draws,
        # so the default path consumes exactly the same random stream as before.
        self.rare_frac = float(rare_frac)
        self.rare_pix, self.rare_w, self.rare_share = [], [], {}
        if self.rare_frac > 0:
            sub = np.random.default_rng(0)
            tot = np.zeros(256, np.int64)
            for m in self.msks:
                tot += np.bincount(m.ravel(), minlength=256)[:256]
            lab = tot[:IGNORE_ID].sum()
            self.rare_share = {int(c): float(tot[c]) / max(lab, 1) for c in rare_ids}
            for m in self.msks:
                d, w = {}, []
                flat = m.ravel()
                for c in rare_ids:
                    if self.rare_share[int(c)] <= 0:
                        continue
                    ix = np.flatnonzero(flat == c)
                    if len(ix):
                        if len(ix) > 4096:
                            ix = np.sort(sub.choice(ix, 4096, replace=False))
                        d[int(c)] = ix.astype(np.int32)
                        w.append(1.0 / self.rare_share[int(c)])
                self.rare_pix.append(d)
                self.rare_w.append(np.array(w) / sum(w) if w else None)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        j = self.ids[i]
        rng = np.random.default_rng()
        if self.rare_frac > 0 and self.rare_pix[j] and rng.random() < self.rare_frac:
            cls = list(self.rare_pix[j])
            c = cls[int(rng.choice(len(cls), p=self.rare_w[j]))]
            pix = self.rare_pix[j][c]
            cy, cx = divmod(int(pix[int(rng.integers(len(pix)))]), self.msks[j].shape[1])
            img, msk = aug_v1_fast_at(self.imgs[j], self.msks[j], rng, self.crop, cy, cx)
        else:
            img, msk = self.aug(self.imgs[j], self.msks[j], rng, self.crop)
        return (normalize(np.ascontiguousarray(img)),
                torch.from_numpy(np.ascontiguousarray(msk).astype(np.int64)),
                1_000_000 + j)


# =============================================================================
# SECTION 5 -- the model
# =============================================================================
# Three pieces, bottom to top:
#
#   ConvBNAct     Conv -> BatchNorm -> ReLU, with no bias on the convolution
#                 (the BatchNorm carries the shift, so a bias would be redundant).
#   DAPPM         the context head that goes on C5.
#   DAPPMFPNSeg   encoder + DAPPM + FPN-lite decoder = the whole network.
#
# EVERY module and attribute name below matches the repository's definition
# (scripts/quant/vitis_defs/dappm.py + scripts/quant/ctx_models.py +
# CoralCapstone/src/coral/models.py).  That is not cosmetic: PyTorch derives
# checkpoint key names from attribute names, so `self.scale0` has to be called
# `scale0` for `swp_r34_dappm_c512_s0.pt` to load into this file with strict=True.

class ConvBNAct(nn.Sequential):
    """Conv(k x k, no bias) -> BatchNorm -> ReLU (or nothing, with act=False).

    The repository has two spellings of this block -- `coral.models.ConvBNAct`
    (default kernel 3, with a `groups` argument) and the copy inside
    `vitis_defs/dappm.py` (default kernel 1, no `groups`).  They build the SAME
    three submodules whenever groups=1, which is everywhere in this model, so one
    class covers both; every call site below passes its kernel size explicitly
    except the decoder's `smooth`, which wants the 3x3 default.
    """

    def __init__(self, cin, cout, k=3, s=1, groups=1, act=True):
        super().__init__(
            nn.Conv2d(cin, cout, k, s, k // 2, groups=groups, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True) if act else nn.Identity())


#: (kernel, stride) of the three pooled DAPPM branches; the stride is also the
#: factor the branch has to be upsampled back by.
POOLS = ((5, 2), (9, 4), (17, 8))


def up2_cascade(factor):
    """`factor`x bilinear upsampling written as a stack of 2x steps. No parameters.

    Why not one `nn.Upsample(scale_factor=8)`?  Because of a measured property of
    the Vitis-AI toolchain (project training rules): a 2x bilinear upsample is
    rewritten into an op the DPU runs, while a single 4x or 8x bilinear resize is
    left as `upsample-fix` and assigned to the CPU, which splits the compiled graph
    in two and collapses throughput.  Three 2x steps compile; one 8x step does not.
    """
    n = int(factor).bit_length() - 1
    assert 2 ** n == int(factor), "upsample factor must be a power of two, got %r" % (factor,)
    return nn.Sequential(*[nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False)
                           for _ in range(n)])


class DAPPM(nn.Module):
    """Deep Aggregation Pyramid Pooling Module (DDRNet, arXiv 2101.06085, section 3.3).

    Five views of the same input map, at five amounts of blur/context:

        branch 0   1x1 convolution on the input map                (finest)
        branch 1   AvgPool(k=5,  stride=2, pad=2)  + 1x1
        branch 2   AvgPool(k=9,  stride=4, pad=4)  + 1x1
        branch 3   AvgPool(k=17, stride=8, pad=8)  + 1x1
        branch 4   global average pool (whole map -> 1x1) + 1x1    (coarsest)

    then a cascade, coarse folded into fine one step at a time:

        y0 = branch0
        y_i = process_i( upsample(branch_i) + y_{i-1} )      3x3 conv each

    and finally `compression(concat(y0..y4)) + shortcut(input)`, ReLU.

    Two deliberate deviations from the paper, both for the DPU:

      1. Every block is Conv -> BN -> ReLU.  The paper uses pre-activation
         (BN -> ReLU -> Conv), which leaves a BatchNorm whose input is an add or a
         concatenation rather than a convolution.  The quantizer folds a BN into the
         convolution IN FRONT of it; a BN with no convolution in front is lowered as
         a depthwise convolution of its own and lands on the CPU, splitting the
         graph.  Reordering removed 11 such rewrites from DDRNet's own head.
      2. A ReLU after the residual add, so the block's output has the same sign
         structure as the post-ReLU C5 map it replaces.  The paper leaves the sum
         unactivated.

    Why the pooling is legal at all: the DPU's average-pool kernel limit is 256 per
    side, and at the deployment size (512x1024 input, stride-32 C5) this runs on a
    16x32 map, so k=17 is a 17x17 window on a 16x32 map.  Far inside the limit.
    EfficientNet's squeeze-and-excite failed because it pooled a stride-2 map.

    `pooled=False` is the capacity-matched control (`fpn_resnet34_dappm_nopool`):
    identical parameter count, pooling and upsampling removed.  It is included
    because the 3-seed ablation that used it is what proves the gain is context and
    not capacity -- it scores the same as the headless baseline.  This file always
    trains `pooled=True`.
    """

    def __init__(self, cin, branch_ch=96, cout=None, pooled=True):
        super().__init__()
        cout = cin if cout is None else cout
        self.cin, self.branch_ch, self.cout, self.pooled = cin, branch_ch, cout, pooled
        self.scale0 = ConvBNAct(cin, branch_ch, 1)
        if pooled:
            self.scale = nn.ModuleList(
                [nn.Sequential(nn.AvgPool2d(k, s, k // 2), ConvBNAct(cin, branch_ch, 1))
                 for k, s in POOLS])
            self.scale_global = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                              ConvBNAct(cin, branch_ch, 1))
            self.up = nn.ModuleList([up2_cascade(s) for _, s in POOLS])
        else:                       # capacity-matched: same convs, no pooling, no resize
            self.scale = nn.ModuleList([nn.Sequential(ConvBNAct(cin, branch_ch, 1))
                                        for _ in POOLS])
            self.scale_global = nn.Sequential(ConvBNAct(cin, branch_ch, 1))
            self.up = nn.ModuleList([nn.Identity() for _ in POOLS])
        self.process = nn.ModuleList([ConvBNAct(branch_ch, branch_ch, 3) for _ in range(4)])
        self.compression = ConvBNAct(branch_ch * 5, cout, 1, act=False)
        self.shortcut = ConvBNAct(cin, cout, 1, act=False)
        self.act = nn.ReLU(inplace=False)

    def forward(self, x):
        h, w = x.shape[-2:]
        if self.pooled:
            assert h % 8 == 0 and w % 8 == 0, (
                "DAPPM: the map it pools must have both sides divisible by 8 (got %dx%d); "
                "the pooled branches come back up by a fixed 2x/4x/8x. At stride 32 that "
                "means an input whose sides are multiples of 256." % (h, w))
        ys = [self.scale0(x)]
        for i in range(3):
            ys.append(self.process[i](self.up[i](self.scale[i](x)) + ys[-1]))
        g = self.scale_global(x)
        if self.pooled:
            # Upsampling a 1x1 map is a broadcast: every interpolation mode returns the
            # same constant field, so NEAREST here is not an approximation -- and it is
            # the mode the DPU implements at an arbitrary factor.  A fixed scale factor
            # cannot be used: 1x1 -> 16x32 is 16x in H and 32x in W.
            g = F.interpolate(g, size=(h, w), mode="nearest")
        ys.append(self.process[3](g + ys[-1]))
        return self.act(self.compression(torch.cat(ys, 1)) + self.shortcut(x))


class DAPPMFPNSeg(nn.Module):
    """ResNet-34 encoder + DAPPM on C5 + FPN-lite decoder.  The whole network.

    The decoder is the project's standard "FPN-lite":
      * one 1x1 convolution per encoder stage, projecting it to `dch` (=64) channels
        -- these are the "lateral" convolutions, hence `self.lat`;
      * starting from the coarsest, upsample by bilinear interpolation to the next
        level's size and ADD (not concatenate -- that is the "lite" part);
      * one 3x3 Conv-BN-ReLU (`smooth`) to clean up the interpolation seams;
      * one 1x1 convolution to `num_classes` logits (`head`);
      * one final bilinear resize back to the input resolution, so the output is
        (B, 4, H, W) -- one score per class per input pixel.

    The ONLY difference from the `fpn_resnet34` baseline is `self.dappm`, and
    because the block's output channel count equals C5's, even the lateral
    convolutions have identical shapes.  That is what makes the comparison clean.
    """

    def __init__(self, num_classes=4, backbone="resnet34", dch=64, branch_ch=96,
                 pretrained=True, pooled=True):
        super().__init__()
        import timm
        # features_only + out_indices=(1,2,3,4) gives the four maps at strides
        # 4, 8, 16, 32 -- for ResNet-34, 64 / 128 / 256 / 512 channels.
        self.enc = timm.create_model(backbone, pretrained=pretrained, features_only=True,
                                     out_indices=(1, 2, 3, 4))
        chs = self.enc.feature_info.channels()
        self.dappm = DAPPM(chs[-1], branch_ch, chs[-1], pooled=pooled)
        self.decoder_type = "fpn"
        self.lat = nn.ModuleList([nn.Conv2d(c, dch, 1) for c in chs])
        self.smooth = ConvBNAct(dch, dch)          # 3x3, the coral.models default
        self.head = nn.Conv2d(dch, num_classes, 1)
        self.num_classes = num_classes

    def forward(self, x):
        H, W = x.shape[-2:]
        feats = list(self.enc(x))
        feats[-1] = self.dappm(feats[-1])          # context head on C5 only
        f = [l(t) for l, t in zip(self.lat, feats)]
        y = f[-1]
        for t in reversed(f[:-1]):                 # coarse -> fine, upsample and add
            y = F.interpolate(y, size=t.shape[-2:], mode="bilinear", align_corners=False) + t
        y = self.head(self.smooth(y))
        return F.interpolate(y, size=(H, W), mode="bilinear", align_corners=False)


def build_model(num_classes=NUM_CLASSES, pretrained=True, pooled=True):
    """The one factory.  `pretrained=False` when loading a checkpoint (the weights
    are about to be overwritten anyway, and it avoids a pointless download)."""
    return DAPPMFPNSeg(num_classes=num_classes, backbone="resnet34",
                       dch=RECIPE["dch"], branch_ch=RECIPE["branch_ch"],
                       pretrained=pretrained, pooled=pooled)


# =============================================================================
# SECTION 6 -- the metric
# =============================================================================
class ConfusionMatrix:
    """One NxN integer matrix accumulated over a whole split.

    `mat[t, p]` counts pixels whose ground truth is class t and whose prediction is
    class p.  Everything else is read off it:

        TP_c = mat[c, c]                        correctly labelled class c
        FP_c = column c minus TP                labelled c but is not
        FN_c = row c minus TP                   is c but labelled something else
        IoU_c = TP / (TP + FP + FN)
        mIoU  = mean of IoU over the classes PRESENT in the ground truth
        pixel accuracy = trace / total

    Pixels whose ground truth is `ignore_index` (255) are dropped before counting,
    so they affect neither the loss nor the score.  This is the official Coralscapes
    protocol and the same definition the repository uses.
    """

    def __init__(self, num_classes, ignore_index=IGNORE_ID):
        self.n = num_classes
        self.ignore = ignore_index
        self.mat = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    def update(self, pred, target):
        pred, target = pred.reshape(-1), target.reshape(-1)
        keep = target != self.ignore
        pred, target = pred[keep], target[keep]
        idx = target.to(torch.int64) * self.n + pred.to(torch.int64)
        self.mat += torch.bincount(idx, minlength=self.n * self.n).reshape(self.n, self.n).cpu()

    def results(self, class_names):
        m = self.mat.double().numpy()
        tp = np.diag(m)
        fp = m.sum(0) - tp
        fn = m.sum(1) - tp
        union = tp + fp + fn
        iou = np.where(union > 0, tp / np.maximum(union, 1), np.nan)
        present = m.sum(1) > 0                     # classes that occur in the ground truth
        recall = np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1), np.nan)
        precision = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), np.nan)

        def per_class(vals):
            return {class_names[i]: (None if np.isnan(vals[i]) else float(vals[i]))
                    for i in range(self.n)}

        return {
            "pixel_acc": float(tp.sum() / max(m.sum(), 1)),
            "miou": float(np.nanmean(iou[present])) if present.any() else float("nan"),
            "per_class_iou": per_class(iou),
            "per_class_recall": per_class(recall),
            "per_class_precision": per_class(precision),
            "confusion_matrix": m.astype(np.int64).tolist(),
        }


def class_stats(r, names=HEALTH_NAMES):
    """Round the confusion-matrix results and add the pixel-share numbers.

    `gt_pixel_share` is a class's row sum over the total; `pred_pixel_share` is its
    column sum over the same total, so the two are directly comparable and
    "the model paints 12x too much BLEACHED" is readable at a glance.  IoU alone
    hides the difference between missing bleaching and over-firing on sand.
    """
    out = {"miou": round(float(r["miou"]), 4), "pixel_acc": round(float(r["pixel_acc"]), 4)}
    m = np.asarray(r["confusion_matrix"], dtype=np.float64)
    tot = max(m.sum(), 1.0)
    rnd = lambda d: {k: (None if v is None else round(float(v), 4)) for k, v in d.items()}
    out.update({
        "per_class_iou": rnd(r["per_class_iou"]),
        "per_class_precision": rnd(r["per_class_precision"]),
        "per_class_recall": rnd(r["per_class_recall"]),
        "gt_pixel_share": {n: round(float(m[i].sum() / tot), 6) for i, n in enumerate(names)},
        "pred_pixel_share": {n: round(float(m[:, i].sum() / tot), 6) for i, n in enumerate(names)},
        "confusion_matrix": m.astype(np.int64).tolist(),
    })
    b = names.index("BLEACHED")
    gt, pr = float(m[b].sum() / tot), float(m[:, b].sum() / tot)
    out["bleached"] = {"iou": out["per_class_iou"]["BLEACHED"],
                       "precision": out["per_class_precision"]["BLEACHED"],
                       "recall": out["per_class_recall"]["BLEACHED"],
                       "gt_pixel_share": round(gt, 6), "pred_pixel_share": round(pr, 6),
                       "pred_over_gt_area": round(pr / gt, 3) if gt > 0 else None}
    return out


# =============================================================================
# SECTION 6b -- 7-class runs: the partial-label loss and the lenient 4-class score
# =============================================================================
def allowed_table(names=SEVEN_NAMES):
    """(256, C) bool: row t = the classes a pixel with target code t may be.

    Rows 0..C-1 are one-hot (an exact label), rows PARTIAL_BASE + h are the CoralScapes
    health sets of PARTIAL_SETS, and every other row (255 included) is empty = ignored."""
    A = torch.zeros(256, len(names), dtype=torch.bool)
    for i in range(len(names)):
        A[i, i] = True
    for h, hn in enumerate(HEALTH_NAMES):
        for c in PARTIAL_SETS[hn]:
            A[PARTIAL_BASE + h, names.index(c)] = True
    return A


def partial_label_loss(logits, target, allowed, bleach_weight=1.0):
    """Set-valued cross-entropy: mean over labelled pixels of

        -log( sum over c in S(pixel) of softmax(logits)_c )

    where S(pixel) = allowed[target].  For a one-hot S this IS cross-entropy, so exact
    labels and set labels share one loss; pixels whose set is empty (IGNORE) are left
    out of the mean exactly as `ignore_index` does.  Computed in fp32 as
    logsumexp(all logits) - logsumexp(logits in S)."""
    logits = logits.float()
    A = allowed.to(logits.device)[target]                # (N, H, W, C)
    valid = A.any(-1)
    # An all-False row would make the second logsumexp -inf and its backward NaN, even
    # though the pixel is then dropped; give ignored pixels the full set instead.
    A = (A | ~valid.unsqueeze(-1)).permute(0, 3, 1, 2)
    lse_all = torch.logsumexp(logits, 1)
    lse_set = torch.logsumexp(logits.masked_fill(~A, float("-inf")), 1)
    if bleach_weight != 1.0:
        # --bleach-weight (default off): pixels whose label set is exactly {BLEACHED} count
        # `bleach_weight` times in a weighted mean; every other labelled pixel counts once.
        b = SEVEN_NAMES.index("BLEACHED")
        only_b = A[:, b] & (A.sum(1) == 1) & valid
        w = 1.0 + (float(bleach_weight) - 1.0) * only_b.float()
        nll_all, w = (lse_all - lse_set)[valid], w[valid]
        return (nll_all * w).sum() / w.sum() if nll_all.numel() else logits.sum() * 0.0
    nll = (lse_all - lse_set)[valid]
    return nll.mean() if nll.numel() else logits.sum() * 0.0


def lenient_health_from_seven(pred7, gt4):
    """7-class prediction -> 4 health ids for scoring against CoralScapes' 4 classes.

    ALIVE -> LIVE, BLEACHED -> BLEACHED, DEAD -> DEGRADED, SAND/OCEAN/OTHER -> OTHER, and
    ROCK -> DEGRADED where the ground truth is DEGRADED, else OTHER (ROCK is in both
    partial sets, so it is never counted wrong against either)."""
    lut = torch.tensor([1, 2, 0, 3, 3, 3, 3], dtype=torch.int64, device=pred7.device)
    p = lut[pred7.to(torch.int64)]
    rock = pred7 == SEVEN_NAMES.index("ROCK")
    return torch.where(rock & (gt4.to(p.device) == 2), torch.full_like(p, 2), p)


# =============================================================================
# SECTION 7 -- evaluation (whole frame, 1024 x 512)
# =============================================================================
# Training sees 512x512 crops, but the score is taken on the WHOLE 1024x512 frame in
# one forward pass.  That is legitimate because the network is fully convolutional:
# it accepts any input whose sides are multiples of 32 (and, because of the DAPPM
# assertion, multiples of 256).  It is also the protocol the Kria build uses, so the
# number here is the number that matters.
def make_predictor(model, device, amp=True, num_classes=NUM_CLASSES):
    """Returns `predict(img_uint8) -> (H, W) int64 tensor of class ids`."""
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    model = model.to(device).eval()

    @torch.no_grad()
    def predict(img_np):
        a = np.ascontiguousarray(img_np)
        if not a.flags.writeable:          # a read-only view straight out of the memmap
            a = a.copy()
        x = torch.from_numpy(a).to(device).permute(2, 0, 1)[None].float() / 255.0
        x = (x - mean) / std
        # fp16 autocast is used for evaluation too -- the published numbers were
        # produced this way, and turning it off shifts the 4th decimal.
        with torch.autocast(device.type, enabled=(amp and device.type == "cuda")):
            lg = model(x)
        lg = lg.float()
        if lg.shape[-2:] != x.shape[-2:]:
            lg = F.interpolate(lg, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return lg.argmax(1)[0]

    return predict


def evaluate(predict_fn, split, ids, res="512", cache=None, limit=None, desc="eval",
             progress=False):
    """Accumulate one confusion matrix over a whole split and return its results."""
    cm = ConfusionMatrix(NUM_CLASSES, ignore_index=IGNORE_ID)
    ids = list(ids)[:limit] if limit else list(ids)
    it = ids
    if progress:
        try:
            from tqdm import tqdm
            it = tqdm(ids, desc=desc, leave=False)
        except ImportError:
            pass
    for i in it:
        img = read_array(split, i, res, "images", cache)
        gt = torch.from_numpy(np.asarray(read_array(split, i, res, "health", cache)).astype(np.int64))
        cm.update(predict_fn(img).cpu(), gt)
    out = cm.results(HEALTH_NAMES)
    out["n_images"] = len(ids)
    return out


def evaluate_lenient(predict_fn, split, ids, res="512", cache=None, limit=None):
    """`evaluate` for a 7-class model on CoralScapes' 4 health classes: predictions are
    collapsed with lenient_health_from_seven (ROCK is right for DEGRADED and OTHER)."""
    cm = ConfusionMatrix(NUM_CLASSES, ignore_index=IGNORE_ID)
    ids = list(ids)[:limit] if limit else list(ids)
    for i in ids:
        img = read_array(split, i, res, "images", cache)
        gt = torch.from_numpy(np.asarray(read_array(split, i, res, "health", cache)).astype(np.int64))
        cm.update(lenient_health_from_seven(predict_fn(img).cpu(), gt), gt)
    out = cm.results(HEALTH_NAMES)
    out["n_images"] = len(ids)
    return out


# =============================================================================
# SECTION 8 -- checkpoints
# =============================================================================
# The dict below has exactly the keys `scripts/v2/v2_common.py::save_ckpt` writes,
# so the repository's own loaders (`v2_common.load_ckpt`, `eval_v2.py`,
# `export_v2_for_vitis.py`, `train_r34_dappm.py --eval`) read a checkpoint written
# here without knowing it came from this file.  `load_ckpt` needs `model`,
# `num_classes` and `state_dict`; the rest is provenance, and the evaluator reads
# `args.eval_res`.
def save_ckpt(path, model, args, *, epoch, val_miou, history=None):
    seven = int(args.get("classes", NUM_CLASSES)) == 7   # absent in 4-class runs
    torch.save({
        "state_dict": model.state_dict(),
        "model": RECIPE["model"],
        "num_classes": 7 if seven else NUM_CLASSES,
        "crop": args["crop"],
        "res": args["res"],
        "target": "seven" if seven else RECIPE["target"],
        "args": args,
        "epoch": epoch,
        "val_miou": val_miou,
        "dataset": "josauder/coralscapesV2",
        "class_names": SEVEN_NAMES if seven else HEALTH_NAMES,
        "mapping": SEVEN_MAPPING_BLOCK if seven else MAPPING_BLOCK,
        "mapping_file": "configs/health_label_mapping_v2.yaml (embedded in standalone_train.py)",
        "class_weights": None,                 # the v1 recipe uses unweighted CE
        "history": history or [],
    }, path)
    return path


def load_ckpt(path, device="cpu", pooled=True):
    """Rebuild the network and load a checkpoint into it with strict=True.

    strict=True is the point: it fails if a single key is missing, extra or the
    wrong shape, which is the cheapest possible proof that the definition in this
    file is the definition the checkpoint was trained with.
    """
    ck = torch.load(path, map_location="cpu", weights_only=False)
    name = ck.get("model", RECIPE["model"])
    if name not in ("fpn_resnet34_dappm", "fpn_resnet34_dappm_nopool"):
        raise SystemExit(
            f"{path} holds a '{name}', not a ResNet-34+DAPPM model. This file only "
            f"builds fpn_resnet34_dappm.")
    m = build_model(num_classes=int(ck.get("num_classes", NUM_CLASSES)), pretrained=False,
                    pooled=pooled and not name.endswith("_nopool"))
    keys = m.load_state_dict(ck["state_dict"], strict=True)
    assert not keys.missing_keys and not keys.unexpected_keys, keys
    return m.eval().to(device), ck


#: 7-class head row <- 4-class health row, for --init from a 4-class checkpoint.
SEVEN_FROM_HEALTH = {"BLEACHED": "BLEACHED", "DEAD": "DEGRADED", "ALIVE": "LIVE",
                     "ROCK": "OTHER", "OCEAN": "OTHER", "SAND": "OTHER", "OTHER": "OTHER"}


def init_from_checkpoint(model, path, names):
    """--init: start from a trained checkpoint instead of ImageNet (fine-tuning).

    Every tensor is copied (strict=True).  If the head's class count differs (a 4-class
    CoralScapes checkpoint into a 7-class run), each 7-class head row is copied from its
    health row in SEVEN_FROM_HEALTH and the bias is lowered by log(k) for the k rows
    sharing a source, so the initial softmax mass per health class is unchanged."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = dict(ck["state_dict"])
    tgt = model.state_dict()
    if sd["head.weight"].shape != tgt["head.weight"].shape:
        src_names = list(ck.get("class_names") or HEALTH_NAMES)
        if list(names) != SEVEN_NAMES or src_names != HEALTH_NAMES:
            raise SystemExit(f"--init: cannot map a {src_names} head onto {names}")
        rows = [src_names.index(SEVEN_FROM_HEALTH[n]) for n in names]
        k = {r: rows.count(r) for r in rows}
        sd["head.weight"] = sd["head.weight"][rows].clone()
        sd["head.bias"] = torch.stack([sd["head.bias"][r] - math.log(k[r]) for r in rows])
        print(f"init: head 4->7 rows {dict(zip(names, rows))}", flush=True)
    model.load_state_dict(sd, strict=True)
    print(f"init: weights from {path} (epoch {ck.get('epoch')}, val {ck.get('val_miou')})",
          flush=True)


# =============================================================================
# SECTION 9 -- the GPU profile gate
# =============================================================================
# Project rule 1: before launching a queue of runs, profile one epoch and report
# seconds/epoch, nvidia-smi utilisation and loader ms/item; do not start until the
# GPU is the bottleneck (utilisation > 80 %).  The history behind the rule: an
# unprofiled loader once made a 31-run sweep CPU-bound, and an A100 ran at exactly
# the laptop's speed.
#
# `--profile 2` runs two epochs and gates on the SECOND.  Epoch 1 pays for worker
# process spawn, cuDNN autotuning and a cold page cache, so a 1-epoch gate is a gate
# that can only fail.
class GpuSampler(threading.Thread):
    """Poll `nvidia-smi` once a second while an epoch runs."""

    def __init__(self, period=1.0):
        super().__init__(daemon=True)
        self.period, self.samples, self._ev = period, [], threading.Event()
        self.others = 0
        self.apps_listed = False

    def _query(self, q):
        try:
            out = subprocess.run(["nvidia-smi", f"--query-{q}",
                                  "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=5)
            return [l.strip() for l in out.stdout.splitlines() if l.strip()]
        except Exception:
            return []

    def run(self):
        while not self._ev.wait(self.period):
            for l in self._query("gpu=utilization.gpu"):
                try:
                    self.samples.append(float(l.split(",")[0]))
                except ValueError:
                    pass
            apps = self._query("compute-apps=pid")
            if apps:
                self.apps_listed = True
                self.others = max(self.others, max(0, len(apps) - 1))

    def stop(self):
        self._ev.set()
        if self.is_alive():
            self.join(timeout=8)
        s = sorted(self.samples)
        return {"gpu_util_mean": round(float(np.mean(s)), 1) if s else None,
                "gpu_util_p50": round(float(s[len(s) // 2]), 1) if s else None,
                "gpu_util_samples": len(s),
                "compute_apps_listed": bool(self.apps_listed),
                # null, not 0: on this laptop's consumer WDDM driver nvidia-smi
                # enumerates no compute apps at all, not even this process, so
                # "0 others" would be a number that is always wrong.
                "other_gpu_processes": self.others if self.apps_listed else None}


def gate_file(out_dir):
    return Path(out_dir) / "standalone_profile_gate.json"


def gate_key(args):
    """What has to match for a stored gate result to still apply."""
    key = {"model": RECIPE["model"], "crop": RECIPE["crop"], "bs": RECIPE["bs"],
           "workers": args.workers, "fast_aug": bool(args.fast_aug),
           "cache": str(args.cache or "")}
    if getattr(args, "classes", NUM_CLASSES) != NUM_CLASSES:
        key["classes"] = args.classes
    return key


def check_gate(args):
    """Refuse to train unless a passing profile for this configuration exists."""
    if args.no_gate:
        print("gate: SKIPPED (--no-gate)", flush=True)
        return
    p = gate_file(args.out)
    if not p.is_file():
        raise SystemExit(
            "PROFILE GATE: no %s. The project rules forbid starting a run before measuring that\n"
            "the GPU, not the data loader, is the bottleneck. Run:\n"
            "    python %s --profile 2\n"
            "and then start training again (or pass --no-gate if you have already\n"
            "measured this exact configuration elsewhere)." % (p, Path(__file__).name))
    rec = json.loads(p.read_text(encoding="utf-8"))
    if rec.get("key") != gate_key(args):
        raise SystemExit(
            "PROFILE GATE: %s was measured for a different configuration\n  stored: %s\n"
            "  now:    %s\nRe-run `--profile 2`." % (p, rec.get("key"), gate_key(args)))
    util = rec.get("gpu_util_mean") or 0
    if util < GATE_UTIL:
        raise SystemExit(
            "PROFILE GATE FAIL: mean GPU utilisation %.1f %% < %.0f %% in the stored profile.\n"
            "The loader, not the GPU, is the bottleneck. Build the decoded cache\n"
            "(scripts/v2/build_cache.py --res 512 --out cache_v2) and keep --fast-aug on,\n"
            "then re-profile. Do not start the run (project training rules)." % (util, GATE_UTIL))
    print(f"gate: PASS (stored profile: {util} % mean GPU utilisation, "
          f"{rec.get('seconds_per_epoch')} s/epoch)", flush=True)


# =============================================================================
# SECTION 10 -- training
# =============================================================================
def resolved_config(args, cache, n_train, n_val, n_test, steps_per_epoch):
    """Everything that decides what this run is, in one printable dict."""
    cfg = dict(RECIPE)
    cfg.update({
        "tag": args.tag, "seed": args.seed, "epochs": args.epochs,
        "fast_aug": bool(args.fast_aug),
        "cache": str(cache) if cache else None,
        "data_root": str(DATA_ROOT),
        "out": str(args.out),
        "device": args.device,
        "train_images": n_train, "val_images": n_val, "test_images": n_test,
        "items_per_epoch": n_train * RECIPE["crops_per_image"],
        "steps_per_epoch": steps_per_epoch,
    })
    return cfg


def train(args, cache):
    dev = torch.device(args.device)

    # ---- reproducibility, such as it is -------------------------------------
    # These fix the weight initialisation, the DataLoader shuffle and the CUDA
    # kernels.  They do NOT fix the augmentation (see CoralscapesV2Crops).
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # 7-class mode / no-CoralScapes mode (both off by default: the default path below
    # is the published 4-class CoralScapes recipe, line for line).
    seven = args.classes == 7
    use_cs = not args.no_coralscapes
    xkind, xlut = ("masks", FINE7_LUT) if seven else ("health", None)
    names = SEVEN_NAMES if seven else HEALTH_NAMES
    if not use_cs and not args.extra_train:
        raise SystemExit("--no-coralscapes needs at least one --extra-train export")

    train_ids = read_split_ids("train") if use_cs else []
    val_ids = read_split_ids("validation") if use_cs else []
    test_ids = read_split_ids("test") if use_cs else []

    parts = []
    if use_cs:
        ds = CoralscapesV2Crops("train", train_ids, RECIPE["crop"], res=RECIPE["res"],
                                cache=cache, fast_aug=args.fast_aug,
                                crops_per_image=RECIPE["crops_per_image"],
                                label_lut=HEALTH_TO_PARTIAL_LUT if seven else None)
        parts.append(ds)
    extras = []
    rare_kw = {}
    if getattr(args, "rare_crop_frac", 0.0) > 0:     # default off: ExtraCrops is built exactly as before
        rare_kw = {"rare_frac": args.rare_crop_frac,
                   "rare_ids": [SEVEN_NAMES.index(n) for n in args.rare_classes.split(",")]}
    if args.extra_train:
        for root in args.extra_train:
            e = ExtraCrops(root, RECIPE["crop"], fast_aug=args.fast_aug,
                           crops_per_image=RECIPE["crops_per_image"],
                           kind=xkind, label_lut=xlut, **rare_kw)
            print(f"extra train: {root}: {len(e.imgs)} photos, {len(e)} items", flush=True)
            if rare_kw:
                print("  rare-crop: pixel share " + " ".join(
                    f"{SEVEN_NAMES[c]}={100 * f:.2f}%" for c, f in e.rare_share.items())
                    + f", photos with a rare class {sum(bool(d) for d in e.rare_pix)}/{len(e.imgs)}",
                    flush=True)
            extras.append(e)
        ds = torch.utils.data.ConcatDataset(parts + extras)

    # Validation / test sets for every run that is not the default 4-class CoralScapes
    # one: the --val-dir exports (default: the --extra-train exports that have the split),
    # decoded once, pooled into one confusion matrix.
    alt_eval = seven or not use_cs
    def _export_sets(split, roots):
        out = []
        for r in roots:
            d = Path(r) / "processed" / f"{split}_512" / "images"
            if d.is_dir() and any(d.glob("*.png")):
                im, ms, _ = _load_export_split(r, split, xkind, xlut)
                out.append((str(r), im, ms))
        return out
    val_sets = (_export_sets(args.val_split, args.val_dir or args.extra_train)
                if alt_eval else [])
    test_sets = _export_sets("test", args.val_dir or args.extra_train) if alt_eval else []
    # Step mode on a small export (7-class / no-CoralScapes runs only): repeat the item list so one
    # DataLoader pass covers at least one validation block. Otherwise the workers drain and refill
    # every few steps -- E1_norfolk (41 photos, 10 steps per pass) ran at ~2x the profiled step
    # time. One shuffle over k copies is statistically the same as k shuffled passes.
    repeat = 1
    if args.total_steps > 0 and alt_eval:
        need = (args.eval_every or 447) * RECIPE["bs"]
        if len(ds) < need:
            repeat = -(-need // len(ds))
            ds = torch.utils.data.ConcatDataset([ds] * repeat)
    dl = DataLoader(ds, batch_size=RECIPE["bs"], shuffle=True,
                    num_workers=args.workers,
                    drop_last=True, pin_memory=dev.type == "cuda",
                    persistent_workers=(not args.smoke) and args.workers > 0)
    steps_per_epoch = len(dl)
    # Step-matched mode (--total-steps N): train for exactly N optimizer steps and
    # validate every --eval-every steps, whatever the dataset size, so arms with more
    # data get the same budget and the same validation cadence as the control. Off by
    # default: without --total-steps every line below behaves exactly as before.
    step_mode = args.total_steps > 0
    block = (args.eval_every or steps_per_epoch) if step_mode else steps_per_epoch

    cfg = resolved_config(args, cache, len(train_ids), len(val_ids), len(test_ids),
                          steps_per_epoch)
    if step_mode:
        cfg.update({"step_mode": True, "total_steps": args.total_steps,
                    "eval_every_steps": block, "workers": args.workers})
    if extras:
        cfg.update({"extra_train": [{"root": str(r), "photos": len(e.imgs), "items": len(e)}
                                    for r, e in zip(args.extra_train, extras)],
                    "items_per_epoch": len(ds)})
    lr = args.lr or RECIPE["lr"]
    if alt_eval or args.init or lr != RECIPE["lr"]:
        cfg.update({"classes": args.classes, "coralscapes": use_cs,
                    "target": "seven" if seven else RECIPE["target"],
                    "loss": ("partial_ce (set-valued CE on CoralScapes pixels, CE on exact "
                             "labels)") if seven else RECIPE["loss"],
                    "partial_sets": ({k: list(v) for k, v in PARTIAL_SETS.items()}
                                     if seven and use_cs else None),
                    "val_sets": [r for r, _, _ in val_sets] if alt_eval else "coralscapes",
                    "test_sets": [r for r, _, _ in test_sets] if alt_eval else "coralscapes",
                    "init": str(args.init or RECIPE["pretrained"]), "lr": lr,
                    "dataset_repeat": repeat})
    if getattr(args, "bleach_weight", 1.0) != 1.0:
        cfg.update({"bleach_weight": args.bleach_weight})
    if getattr(args, "select_metric", "miou") != "miou":
        cfg.update({"select_metric": args.select_metric})
    if rare_kw:
        cfg.update({"rare_crop": {"frac": args.rare_crop_frac, "classes": args.rare_classes.split(","),
                                  "applies_to": "--extra-train export crops (CoralScapes crops unchanged)"}})
    print("RESOLVED CONFIG " + json.dumps(cfg, indent=2), flush=True)

    model = build_model(num_classes=args.classes,
                        pretrained=(RECIPE["pretrained"] == "imagenet") and not args.init
                        ).to(dev)
    if args.init:
        init_from_checkpoint(model, args.init, names)
    nparam = sum(p.numel() for p in model.parameters())

    # ---- loss ----------------------------------------------------------------
    # Plain, UNWEIGHTED cross-entropy with ignore_index=255.  Two notes:
    #   * The v1 recipe has always been unweighted.  The repository used to print
    #     inverse-sqrt class weights for v1 runs, but never handed them to the loss.
    #   * `dice_w` is 1.0 in the published args, but `loss=ce` means no Dice term is
    #     built at all, so that number has no effect.  It is carried in the config
    #     only so the two argument dictionaries compare equal.
    crit = nn.CrossEntropyLoss(ignore_index=IGNORE_ID)
    if seven:
        # 7-class: one set-valued loss for every pixel (CE on exact 7-class labels, the
        # partial-label term on CoralScapes' 4-class pixels); see partial_label_loss.
        allowed = allowed_table(SEVEN_NAMES).to(dev)
        bw = float(getattr(args, "bleach_weight", 1.0))
        crit = lambda out, y: partial_label_loss(out, y, allowed, bleach_weight=bw)  # noqa: E731

    opt = torch.optim.AdamW(model.parameters(), lr=lr,
                            weight_decay=RECIPE["wd"])
    total_steps = (args.total_steps if step_mode
                   else max(args.epochs * steps_per_epoch, 1))
    # OneCycle: learning rate ramps from lr/25 up to lr over the first 10 % of the
    # run, then anneals down to ~lr/1e4.  Stepped once per BATCH, not per epoch.
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr,
                                                total_steps=total_steps, pct_start=0.1)
    # Mixed precision: the forward pass runs in fp16 where that is safe, the loss is
    # computed in fp32, and the scaler multiplies the loss by a large factor before
    # backward so small gradients do not underflow to zero in fp16.
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")

    print(f"=== {args.tag} === {RECIPE['model']} params={nparam/1e6:.4f}M "
          f"crop={RECIPE['crop']} bs={RECIPE['bs']} steps/ep={steps_per_epoch} "
          f"epochs={args.epochs} min_epochs={RECIPE['min_epochs']} "
          f"patience={RECIPE['patience']} loss="
          f"{'partial_label_loss(7 classes)' if seven else 'CrossEntropyLoss(unweighted)'} "
          f"device={dev}", flush=True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_pt = out_dir / f"{args.tag}.pt"
    last_pt = out_dir / f"{args.tag}_last.pt"

    hist, best, best_ep, bad = [], -1.0, 0, 0
    t_all = time.time()
    n_epochs = args.profile or args.epochs
    if step_mode and not args.profile:
        n_epochs = -(-args.total_steps // block)       # ceil: blocks of `block` steps
    eval_limit = 2 if args.smoke else None

    def _cycle():
        while True:
            for b in dl:
                yield b
    cyc = _cycle() if step_mode else None

    # --profile: time __getitem__ itself (decode + augment + normalise, one process),
    # the "loader ms/item" Project rule 1 asks for, on a fixed random sample.
    getitem_ms = None
    if args.profile:
        pick = np.random.default_rng(0).integers(0, len(ds), min(64, len(ds)))
        t_gi = time.perf_counter()
        for i in pick:
            ds[int(i)]
        getitem_ms = round((time.perf_counter() - t_gi) / len(pick) * 1000, 2)
        print(f"PROFILE getitem {getitem_ms} ms/item (main process, {len(pick)} items)",
              flush=True)

    def _validate():
        """(val mIoU, per-class IoU, pixel acc) for the non-default runs."""
        if val_sets:
            cm = ConfusionMatrix(len(names), ignore_index=IGNORE_ID)
            pf = make_predictor(model, dev)
            for _, ims, mss in val_sets:
                for im, gt in list(zip(ims, mss))[:eval_limit]:
                    cm.update(pf(im).cpu(), torch.from_numpy(gt.astype(np.int64)))
            return cm.results(names)
        if use_cs:       # 7-class, no export val split: CoralScapes val, lenient score
            return evaluate_lenient(make_predictor(model, dev), "validation", val_ids,
                                    res=RECIPE["eval_res"], cache=cache, limit=eval_limit)
        return None
    live_log = Path(args.out) / f"{args.tag}_val.jsonl" if args.persist_every_eval else None

    for ep in range(n_epochs):
        model.train()
        tl, n, nb, t0 = 0.0, 0, 0, time.time()
        smp = GpuSampler() if (args.profile and dev.type == "cuda") else None
        if smp is not None:
            smp.start()
        if step_mode:
            k = min(block, args.total_steps - ep * block)
            batches = (next(cyc) for _ in range(k))
        else:
            batches = dl
        for batch in batches:
            x, y, _ = batch
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast("cuda", enabled=dev.type == "cuda"):
                out = model(x)
            out = out.float()                     # the loss is always computed in fp32
            loss = crit(out, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            tl += loss.item() * x.shape[0]
            n += x.shape[0]
            nb += 1
            if args.smoke and nb >= 2:
                break

        # ---- --profile: measure, print, do not evaluate, do not save ----------
        if args.profile:
            secs = time.time() - t0
            gpu = (smp.stop() if smp is not None
                   else {"gpu_util_mean": None, "gpu_util_samples": 0,
                         "other_gpu_processes": None, "compute_apps_listed": False})
            prof = {"epoch": ep + 1, "of": n_epochs, "items": n, "batches": nb,
                    "seconds_per_epoch": round(secs, 1),
                    "ms_per_item_wall": round(secs / max(n, 1) * 1000, 3),
                    "loader_ms_per_item_per_worker":
                        round(secs / max(n, 1) * 1000 * max(args.workers, 1), 2),
                    "getitem_ms_per_item": getitem_ms,
                    **gpu,
                    "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)
                    if dev.type == "cuda" else 0}
            print("PROFILE " + json.dumps(prof), flush=True)
            if ep + 1 < n_epochs:
                continue
            util = prof.get("gpu_util_mean") or 0
            rec = {"key": gate_key(args), "measured": time.strftime("%Y-%m-%dT%H:%M:%S"),
                   **prof}
            gate_file(args.out).write_text(json.dumps(rec, indent=2), encoding="utf-8")
            ok = util >= GATE_UTIL
            print(("PROFILE_GATE_PASS " if ok else "PROFILE_GATE_FAIL ")
                  + f"mean GPU util {util} vs bar {GATE_UTIL} "
                    f"(stored in {gate_file(args.out)})", flush=True)
            return 0 if ok else 3

        # ---- validation, every epoch ----------------------------------------
        # Cadence matters: changing it changes which epoch is selected as "best"
        # and breaks comparability with every earlier run (project training rules).
        model.eval()
        if alt_eval:
            vr = _validate()
            if vr is None:       # nothing to validate on: "best" = the latest weights
                vr = {"miou": float("nan"), "pixel_acc": float("nan"),
                      "per_class_iou": {k: None for k in names}}
        else:
            vr = evaluate(make_predictor(model, dev), "validation", val_ids,
                          res=RECIPE["eval_res"], cache=cache, limit=eval_limit, desc="val")
        rec = {"epoch": ep + 1, "train_loss": round(tl / max(n, 1), 4),
               "lr": round(sched.get_last_lr()[0], 6),
               "val_miou": None if np.isnan(vr["miou"]) else round(float(vr["miou"]), 4),
               "val_pixel_acc": None if np.isnan(vr["pixel_acc"]) else round(float(vr["pixel_acc"]), 4),
               "val_per_class": {k: (None if v is None else round(float(v), 4))
                                 for k, v in vr["per_class_iou"].items()},
               "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)
               if dev.type == "cuda" else 0,
               "epoch_seconds": round(time.time() - t0, 1)}
        hist.append(rec)
        print(json.dumps(rec), flush=True)
        if live_log is not None:
            # --persist-every-eval: the val curve and the latest weights reach --out at
            # every validation, so a killed pod loses at most one block.
            with open(live_log, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            save_ckpt(last_pt, model, cfg, epoch=ep + 1, val_miou=rec["val_miou"],
                      history=hist)

        score = vr["miou"]
        if getattr(args, "select_metric", "miou") == "bleached":
            # --select-metric bleached (default off): best = highest validation BLEACHED IoU
            sb = vr["per_class_iou"].get("BLEACHED")
            score = float("nan") if sb is None else float(sb)
        if score > best or (alt_eval and np.isnan(score)):
            best, best_ep, bad = float(score), ep + 1, 0
            save_ckpt(best_pt, model, cfg, epoch=ep + 1, val_miou=best, history=hist)
        else:
            bad += 1
            if bad >= RECIPE["patience"] and (ep + 1) >= RECIPE["min_epochs"]:
                print(f"early stop at epoch {ep+1} (no val improvement for {bad} epochs)",
                      flush=True)
                break

    secs = round(time.time() - t_all, 1)

    # The LAST epoch's weights are kept as well.  `model` still holds them at this
    # point -- the next lines replace it with the best-val checkpoint.  Selection is
    # unchanged (best-val is what gets reported); this only stops the final epoch
    # from being thrown away, because best-val selection on the noisy 170-image
    # validation split has been seen to land on an under-annealed mid-schedule epoch.
    if hist:
        save_ckpt(last_pt, model, cfg, epoch=len(hist),
                  val_miou=hist[-1]["val_miou"], history=hist)
        print(f"saved last-epoch weights (epoch {len(hist)}, val "
              f"{'n/a' if hist[-1]['val_miou'] is None else format(hist[-1]['val_miou'], '.4f')})"
              f" -> {last_pt}", flush=True)

    # ---- final test score on the best-val checkpoint -------------------------
    model, _ = load_ckpt(best_pt, dev)
    extra_tests = {}
    if alt_eval:
        pf = make_predictor(model, dev)
        for root, ims, mss in test_sets:
            cm = ConfusionMatrix(len(names), ignore_index=IGNORE_ID)
            for im, gt in list(zip(ims, mss))[:eval_limit]:
                cm.update(pf(im).cpu(), torch.from_numpy(gt.astype(np.int64)))
            extra_tests[root] = class_stats(cm.results(names), names)
        if use_cs:
            r = (evaluate_lenient(pf, "test", test_ids, res=RECIPE["eval_res"], cache=cache,
                                  limit=eval_limit) if seven else
                 evaluate(pf, "test", test_ids, res=RECIPE["eval_res"], cache=cache,
                          limit=eval_limit, desc="test"))
            extra_tests["coralscapes_test" + ("_lenient4" if seven else "")] = class_stats(r)
        test_whole = next(iter(extra_tests.values()), None)
    else:
        r = evaluate(make_predictor(model, dev), "test", test_ids, res=RECIPE["eval_res"],
                     cache=cache, limit=eval_limit, desc="test")
        test_whole = class_stats(r)

    out = {"tag": args.tag, "model": RECIPE["model"], "params_M": round(nparam / 1e6, 4),
           "config": cfg, "best_val_miou": round(best, 4), "best_epoch": best_ep,
           "epochs_run": len(hist), "train_seconds": secs,
           "peak_vram_gb": max([h["peak_vram_gb"] for h in hist] or [0]),
           "history": hist, "checkpoint": str(best_pt), "last_checkpoint": str(last_pt),
           "test_whole": test_whole}
    if alt_eval:
        out["tests"] = extra_tests
    log = out_dir / f"train_{args.tag}.json"
    log.write_text(json.dumps(out, indent=2), encoding="utf-8")

    if alt_eval:
        print(f"DONE {args.tag}: val {best:.4f} (ep {best_ep}/{len(hist)}) | tests " + " | ".join(
            f"{Path(k).name} {v['miou']:.4f}" for k, v in extra_tests.items())
            + f" | {secs}s -> {log}", flush=True)
        return 0
    b = test_whole["bleached"]
    print(f"DONE {args.tag}: val {best:.4f} (ep {best_ep}/{len(hist)}) | test whole "
          f"{test_whole['miou']:.4f} | BLEACHED IoU {b['iou']:.4f} P {b['precision']:.4f} "
          f"R {b['recall']:.4f} area {b['pred_pixel_share']*100:.2f}% vs GT "
          f"{b['gt_pixel_share']*100:.2f}% (x{b['pred_over_gt_area']}) | {secs}s -> {log}",
          flush=True)
    return 0


# =============================================================================
# SECTION 11 -- --eval: score an existing checkpoint
# =============================================================================
def evaluate_dir(predict_fn, root, split="test", seven=False):
    """Score one split of an export directory (see _load_export_split); a 7-class model
    is scored on the export's masks/ in 7 classes."""
    imgs, msks, names = (_load_export_split(root, split, "masks", FINE7_LUT) if seven
                         else _load_export_split(root, split))
    cls = SEVEN_NAMES if seven else HEALTH_NAMES
    cm = ConfusionMatrix(len(cls), ignore_index=IGNORE_ID)
    for img, gt in zip(imgs, msks):
        cm.update(predict_fn(img).cpu(), torch.from_numpy(gt.astype(np.int64)))
    out = cm.results(cls)
    out["n_images"] = len(names)
    return out


def eval_only(pt, args, cache):
    dev = torch.device(args.device)
    model, ck = load_ckpt(pt, dev)
    nparam = sum(p.numel() for p in model.parameters())
    seven = int(ck.get("num_classes", NUM_CLASSES)) == 7
    if args.eval_dir:
        t0 = time.time()
        r = evaluate_dir(make_predictor(model, dev), args.eval_dir, args.eval_split, seven)
        st = class_stats(r, SEVEN_NAMES if seven else HEALTH_NAMES)
        print(json.dumps({"checkpoint": str(pt), "eval_dir": str(args.eval_dir),
                          "split": args.eval_split, "n_images": r["n_images"],
                          "seconds": round(time.time() - t0, 1), "test_whole": st}, indent=2),
              flush=True)
        print(f"EVAL_DIR {Path(pt).name} on {args.eval_dir}/{args.eval_split}: mIoU "
              f"{st['miou']:.4f} | per-class " + " ".join(
                  f"{k} {'nan' if v is None else format(v, '.4f')}"
                  for k, v in st["per_class_iou"].items()), flush=True)
        return 0
    test_ids = read_split_ids("test")
    eval_res = str((ck.get("args") or {}).get("eval_res") or ck.get("res") or "512")
    t0 = time.time()
    if seven:   # 4-class CoralScapes test, ROCK counted right on DEGRADED and OTHER
        r = evaluate_lenient(make_predictor(model, dev), "test", test_ids, res=eval_res,
                             cache=cache, limit=2 if args.smoke else None)
    else:
        r = evaluate(make_predictor(model, dev), "test", test_ids, res=eval_res, cache=cache,
                     limit=2 if args.smoke else None, desc="test", progress=args.progress)
    st = class_stats(r)
    b = st["bleached"]
    print(json.dumps({"checkpoint": str(pt), "model": ck.get("model"),
                      "params_M": round(nparam / 1e6, 4),
                      "trained_epoch": ck.get("epoch"),
                      "ckpt_val_miou": (None if ck.get("val_miou") is None
                                        else round(float(ck["val_miou"]), 4)),
                      "eval_res": eval_res, "device": dev.type,
                      "n_images": r["n_images"], "seconds": round(time.time() - t0, 1),
                      "test_whole": st}, indent=2), flush=True)
    print(f"EVAL {Path(pt).name}: test whole mIoU {st['miou']:.4f} | "
          f"BLEACHED IoU {b['iou']:.4f} P {b['precision']:.4f} R {b['recall']:.4f} | "
          f"per-class " + " ".join(f"{k} {v:.4f}" for k, v in st["per_class_iou"].items()),
          flush=True)
    return 0


# =============================================================================
# SECTION 12 -- command line
# =============================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0,
                    help="weight init / shuffle / CUDA seed. Does NOT make the run "
                         "bit-reproducible: the augmentation RNG is seeded from OS "
                         "entropy per item, exactly as in the published pipeline.")
    ap.add_argument("--epochs", type=int, default=RECIPE["epochs"],
                    help="default 20, the published value. Changing it changes the "
                         "OneCycle schedule, so the result is no longer the same recipe.")
    ap.add_argument("--out", default=str(DEFAULT_OUT),
                    help="directory for checkpoints, the run json and the gate record")
    ap.add_argument("--tag", default="",
                    help="default: standalone_r34_dappm_c512_s<seed>")
    ap.add_argument("--eval", default="", metavar="PT",
                    help="score this checkpoint on the 473-image test split and exit")
    ap.add_argument("--profile", type=int, default=0, metavar="N",
                    help="run N epochs, print s/epoch and mean GPU utilisation, write the "
                         "gate record and exit. Use 2: epoch 1 is not steady state.")
    ap.add_argument("--no-gate", action="store_true",
                    help="start training without a stored passing profile (project training rules)")
    ap.add_argument("--smoke", action="store_true",
                    help="CPU wiring test: 2 batches per epoch, 1 epoch, 2 images per "
                         "evaluation. Proves the graph, the data and the checkpoint path "
                         "work. It is not a measurement.")
    ap.add_argument("--cache", default="",
                    help="decoded cache directory ('' = $CORAL_V2_CACHE, then <segmentation>/cache_v2, "
                         "then plain PNG decoding)")
    ap.add_argument("--no-cache", action="store_true", help="ignore the decoded cache")
    ap.add_argument("--fast-aug", dest="fast_aug", action="store_true", default=True,
                    help="windowed scale augmentation + LUT brightness (DEFAULT, and what "
                         "the published run used)")
    ap.add_argument("--no-fast-aug", dest="fast_aug", action="store_false",
                    help="resize the whole frame before cropping: same draws, ~5x slower")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--progress", action="store_true", help="tqdm bar during --eval")
    ap.add_argument("--workers", type=int, default=0, metavar="N",
                    help="DataLoader workers (default 0 = the recipe's %d)" % RECIPE["workers"])
    ap.add_argument("--total-steps", dest="total_steps", type=int, default=0, metavar="N",
                    help="step-matched mode: train exactly N optimizer steps (OneCycle over N), "
                         "validating every --eval-every steps. 0 (default) = epoch mode. "
                         "The CoralScapes-only budget is 20 x 447 = 8940.")
    ap.add_argument("--eval-every", dest="eval_every", type=int, default=0, metavar="M",
                    help="step mode only: validate (and pick best) every M steps; default = "
                         "this dataset's steps per epoch. Use 447 to match the control's cadence.")
    ap.add_argument("--extra-train", dest="extra_train", action="append", default=[],
                    metavar="DIR", help="add the train split of an export directory "
                    "(data/EXPORT_FORMAT.md layout) to the training set; repeatable")
    ap.add_argument("--eval-dir", dest="eval_dir", default="", metavar="DIR",
                    help="with --eval: score an export directory's split instead of CoralScapes test")
    ap.add_argument("--eval-split", dest="eval_split", default="test",
                    choices=["train", "validation", "test"])
    # ---- 7-class / milestone options (all default-off: the default run is unchanged) ----
    ap.add_argument("--classes", type=int, default=NUM_CLASSES, choices=[4, 7],
                    help="head size. 4 (default) = the published health classes. 7 = "
                         "BLEACHED DEAD ALIVE ROCK OCEAN SAND OTHER: exports are read from "
                         "masks/ (fine ids), CoralScapes pixels get set-valued labels "
                         "(PARTIAL_SETS) and the partial-label loss; validation and the "
                         "final test use the exports' validation/test splits")
    ap.add_argument("--no-coralscapes", dest="no_coralscapes", action="store_true",
                    help="train on the --extra-train exports only")
    ap.add_argument("--val-dir", dest="val_dir", action="append", default=[], metavar="DIR",
                    help="7-class / no-CoralScapes runs: export(s) whose validation and test "
                         "splits pick the best checkpoint and give the final test (default: "
                         "the --extra-train exports); repeatable")
    ap.add_argument("--val-split", dest="val_split", default="validation",
                    choices=["train", "validation", "test"],
                    help="which split of the --val-dir exports picks the best checkpoint (e.g. 'test' "
                         "for a held-out-site export that has only a test split)")
    ap.add_argument("--init", default="", metavar="PT",
                    help="start from this checkpoint instead of ImageNet (fine-tuning); a "
                         "4-class head is mapped onto a 7-class one row by row")
    ap.add_argument("--lr", type=float, default=0.0,
                    help="peak OneCycle learning rate (default 0 = the recipe's 1e-3)")
    ap.add_argument("--persist-every-eval", dest="persist_every_eval", action="store_true",
                    help="write <tag>_last.pt and append <tag>_val.jsonl at every validation "
                         "(for pods that can be stopped mid-run)")
    ap.add_argument("--rare-crop-frac", dest="rare_crop_frac", type=float, default=0.0,
                    help="7-class runs: probability that an --extra-train crop is centred on a pixel of "
                         "a rare class present in the photo (class weight 1 / pixel share). Default 0 = off")
    ap.add_argument("--rare-classes", dest="rare_classes", default=RARE_DEFAULT,
                    help=f"--rare-crop-frac classes, comma-separated 7-class names (default {RARE_DEFAULT})")
    ap.add_argument("--crop", type=int, default=0,
                    help="training crop side (default 0 = the recipe's 512); multiple of 256")
    ap.add_argument("--bleach-weight", dest="bleach_weight", type=float, default=1.0,
                    help="7-class runs: loss weight on exact-BLEACHED pixels (default 1 = off)")
    ap.add_argument("--select-metric", dest="select_metric", default="miou", choices=["miou", "bleached"],
                    help="best checkpoint = highest validation mIoU (default) or BLEACHED IoU (7-class)")
    args = ap.parse_args(argv)
    if args.crop:
        if args.crop % 256:
            raise SystemExit("--crop must be a multiple of 256")
        RECIPE["crop"] = args.crop
    if (args.bleach_weight != 1.0 or args.select_metric != "miou") and args.classes != 7:
        raise SystemExit("--bleach-weight / --select-metric bleached need --classes 7")
    if args.rare_crop_frac > 0:
        if args.classes != 7 or not args.extra_train:
            raise SystemExit("--rare-crop-frac needs --classes 7 and at least one --extra-train export")
        bad = [n for n in args.rare_classes.split(",") if n not in SEVEN_NAMES]
        if bad or not 0 < args.rare_crop_frac <= 1:
            raise SystemExit(f"--rare-crop-frac in (0, 1] and --rare-classes from {SEVEN_NAMES}; got {bad}")

    _assert_standalone()

    if args.smoke:
        args.device = "cpu"
        args.epochs = 1
        args.tag = args.tag or f"standalone_r34_dappm_c512_s{args.seed}_smoke"
    args.tag = args.tag or f"standalone_r34_dappm_c512_s{args.seed}"
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.workers = 0 if args.smoke else (args.workers or RECIPE["workers"])

    # ---- resolve the cache ---------------------------------------------------
    cache = None
    if not args.no_cache:
        cand = Path(args.cache) if args.cache else DEFAULT_CACHE
        if cache_ok(cand, "train", RECIPE["res"]) or cache_ok(cand, "test", RECIPE["res"]):
            cache = str(cand)
    args.cache = cache
    hits = [s for s in ("train", "validation", "test")
            if cache and cache_ok(cache, s, RECIPE["res"])]
    print(f"data : {DATA_ROOT}", flush=True)
    print(f"cache: {cache or 'OFF -- decoding PNGs per crop (slow; the profile gate will fail)'}"
          + (f"  [complete for {', '.join(hits)}]" if hits else ""), flush=True)
    print(f"aug  : v1 {'fast (windowed resize + LUT brightness)' if args.fast_aug else 'slow (whole-frame resize)'}",
          flush=True)

    if args.eval:
        pt = Path(args.eval)
        if not pt.is_file():
            pt = REPO / args.eval
        if not pt.is_file():
            raise SystemExit(f"--eval: no such checkpoint: {args.eval}")
        return eval_only(pt, args, cache)

    # The gate applies to real training only: `--profile` is what MEASURES it, and
    # `--smoke` never touches the GPU.
    if not args.smoke and not args.profile:
        check_gate(args)
    return train(args, cache)


if __name__ == "__main__":
    sys.exit(main() or 0)
