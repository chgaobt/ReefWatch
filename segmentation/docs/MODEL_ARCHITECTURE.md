# The model: `fpn_resnet34_dappm`

> In this repository the whole network is defined in `train/standalone_train.py` (SECTION on the
> model). File paths quoted below (`scripts/quant/...`) refer to the original development
> repository. The milestone 1 model is identical except for its 1x1 classifier, which outputs
> 7 classes instead of 4 (BLEACHED, DEAD, ALIVE, ROCK, OCEAN, SAND, OTHER).

ResNet-34 encoder → **DAPPM** context block on the deepest feature map → FPN-lite decoder
→ 4 classes. It is the plain `fpn_resnet34` baseline used everywhere else in this project
with exactly one block added, so any difference in the numbers is attributable to that
block and nothing else.

Code: `scripts/quant/ctx_models.py` (`DAPPMFPNSeg`) and `scripts/quant/vitis_defs/dappm.py`
(`DAPPM`). Built through the one factory, `common.build_seg("fpn_resnet34_dappm", 4)`.

**Glossary.** *Stride* = how much smaller a feature map is than the input (stride 32 means
1/32 the width and height). *C2…C5* = the encoder's output maps at strides 4, 8, 16, 32.
*FPN* = Feature Pyramid Network, a decoder that adds coarse maps back onto fine ones.
*DAPPM* = Deep Aggregation Pyramid Pooling Module (Hong et al., DDRNet, arXiv 2101.06085
§3.3), a block that looks at several spatial scales at once. *DPU* = the Deep-learning
Processing Unit, the fixed-function accelerator on the Kria KV260 board this model is
meant to run on.

## Shape of the graph

At the training crop of 512×512 (and, in brackets, at the 512×1024 deployment frame):

| stage | what it is | stride | channels | map size |
|---|---|---:|---:|---|
| `enc` C2 | timm `resnet34`, `features_only`, `out_indices=(1,2,3,4)` | 4 | 64 | 128×128 (128×256) |
| `enc` C3 | | 8 | 128 | 64×64 (64×128) |
| `enc` C4 | | 16 | 256 | 32×32 (32×64) |
| `enc` C5 | | 32 | 512 | 16×16 (**16×32**) |
| `dappm` | context block, C5 → C5 (`cout == cin`, drop-in) | 32 | 512 | 16×16 (16×32) |
| `lat` | four 1×1 convolutions, every level → 64 channels | 4…32 | 64 | — |
| top-down | `interpolate(bilinear) + add`, C5→C4→C3→C2 | → 4 | 64 | 128×128 (128×256) |
| `smooth` | one 3×3 Conv-BN-ReLU | 4 | 64 | — |
| `head` | 1×1 convolution to the classes | 4 | **4** | — |
| output | one 4× bilinear resize back to input size | 1 | 4 | 512×512 (512×1024) |

Classes are the 4 "health" superclasses defined by `configs/health_label_mapping_v2.yaml`:
**LIVE, BLEACHED, DEGRADED, OTHER** (plus label 255 = ignore, excluded from the loss and
from every metric).

## The DAPPM block

Five parallel views of the same C5 map, aggregated in a cascade:

| branch | operation | receptive field on a 16×32 C5 |
|---|---|---|
| 0 | 1×1 Conv-BN-ReLU | 1 pixel of C5 |
| 1 | `AvgPool2d(k=5, s=2, p=2)` → 1×1 | 5×5 |
| 2 | `AvgPool2d(k=9, s=4, p=4)` → 1×1 | 9×9 |
| 3 | `AvgPool2d(k=17, s=8, p=8)` → 1×1 | 16×17 (clipped by the map) |
| 4 | `AdaptiveAvgPool2d(1)` → 1×1 | the whole map |

Each pooled branch is brought back to C5 resolution, added to the branch above it and
passed through a 3×3 `process` convolution (`y0 = b0`, `y_i = process_i(up(b_i) + y_{i-1})`).
The five results are concatenated, compressed by a 1×1 back to 512 channels, added to a
1×1 shortcut of the input, and ReLU'd. Branch width is 96 channels.

**Why it helps here.** Measured over 5 seeds, the head is a *recall-side* gain on the rare
BLEACHED class: recall 0.617 → 0.674 at flat precision (0.826 → 0.860). A 512×512 crop of
a reef gives a pixel very little local evidence of whether pale coral is bleached tissue or
sand; the pooled branches let the classifier see the surrounding metres of scene. The
capacity-matched ablation (`fpn_resnet34_dappm_nopool`: the identical block with every
pooling layer removed, same parameter count to the tensor) does **not** reproduce the gain,
which is what attributes it to the pooling rather than to the extra 1.09 M parameters.

## Cost

| | value |
|---|---|
| parameters | **22.47 M** (22.47284 M exactly; the plain `fpn_resnet34` baseline is 21.38 M) |
| multiply-accumulates at 512×1024 | **40.21 GMAC** (baseline 39.75) |
| training VRAM, crop 512 / batch 16, AMP | 2.07 GiB |
| training step | ~187 ms on an RTX 5050 laptop GPU |
| Vitis-AI partition | **1 DPU subgraph**, 222 ops; the only CPU op is the final 4× resize |
| estimated KV260 throughput | ~4.8 fps single-thread / ~10.9 fps multi-thread — **a MAC-scaling estimate, never measured on the board** |

## The DPU constraints this graph is written around

These are observed behaviours of the `vai_q_pytorch` / `vai_c_xir` toolchain on this
project's graphs (recorded in `artifacts_quant/reports/COMPILE_GATE_2_4.md` and summarised
in the project notes), not datasheet limits. Each one changed the block before it was trained.

1. **Bilinear upsampling by 2× compiles; by 4× or 8× it does not.** A single
   `nn.Upsample(scale_factor=8, mode='bilinear')` is left as `upsample-fix` and assigned to
   the CPU ("DPU does not support BILINEAR mode"), which splits the graph in two. So each
   pooled branch comes back up through a **cascade of 2× steps** (`dappm.up2_cascade`),
   which is parameter-free and DPU-native at every step. The one 4× resize that is left —
   the final output resize — is deliberately the graph's single CPU op, as it has been in
   every deployed build in this project.
2. **The pooled map's sides must be divisible by 8**, because those upsample factors are
   fixed rather than computed from the tensor shape. At stride 32 that makes **every input
   side a multiple of 256**. `DAPPM.forward` asserts it rather than trusting it. In
   practice: crop 256 → C5 8×8 ✓, crop 512 → 16×16 ✓, frame 512×1024 → 16×32 ✓; a 640×360
   camera frame must be padded to e.g. 768×512 first.
3. **Pooling kernels may be up to 256 px per side.** The `k=17` window and the global pool
   (a 16×32 kernel) on a 16×32 C5 map are far inside that. EfficientNet-B0's
   squeeze-and-excite broke on this hardware because it pooled a *stride-2* map (256×512
   here), not because pooling is forbidden.
4. **Convolution before BatchNorm, always.** `vai_q` folds BatchNorm into the convolution
   in front of it; a BatchNorm with no convolution in front is lowered as a *depthwise*
   convolution, which splits the graph and strands the following ReLU on the CPU. The DDRNet
   paper's blocks are pre-activation (`BN → ReLU → Conv`); every block here is reordered to
   `Conv → BN → ReLU`. That is also why there is a ReLU after the residual add, so the
   block's output has the same sign structure as the post-ReLU C5 map it replaces.
5. **`F.interpolate(size=x.shape[-2:])` can abort the compile outright**
   (`XCOM_OP_MUL_IO_WIDTH_NE`) when the source is a degenerate 1×N map. The global branch
   therefore uses `mode="nearest"` — and upsampling a 1×1 map is a broadcast, so every
   interpolation mode returns the identical tensor there.

## The recipe it is trained with

The project's plain **v1** control recipe, at crop 512. Reproduced verbatim by
`train_r34_dappm.py`; the published run's own argument block is
`artifacts_v2/logs/train_swp_r34_dappm_c512_s0.json`.

| | value | note |
|---|---|---|
| data | Coralscapes V2, 4 health classes | 1790 train images, 1024×512 decoded copies |
| crop | 512×512, 4 crops per image | 447 optimiser steps per epoch |
| augmentation (`--aug v1`) | random scale 0.75–1.5 → random crop → horizontal flip (p 0.5) → brightness gain 0.8–1.2 / bias ±20 (p 0.5) | nothing else |
| batch size | 16 | 512 crops at batch 32 do not fit in 8 GB |
| optimiser | AdamW, lr 1e-3, weight decay 1e-4 | |
| schedule | OneCycle, `pct_start=0.1` | |
| loss | plain cross-entropy, `ignore_index=255` | **unweighted** — the v1 recipe has always been |
| precision | AMP (mixed precision) on CUDA | |
| epochs | 20, no early stopping (patience = 20, floor 15) | |
| encoder init | ImageNet (`timm` `resnet34`) | |
| selection | best whole-image **validation** mIoU | the checkpoint kept is the best-val one |
| loader | `cache_v2` decoded memmaps + `--fast-aug` | throughput only; ≤2 grey levels of difference |

Recipe tuning on this family is worth ≤1 point and does not stack (`artifacts_v2/sweep/SWEEP_REPORT.md`).
Do not change the table above to try to improve the model; add a new sweep arm instead.
