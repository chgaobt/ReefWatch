# ReefWatch segmentation model (milestone 1)

A semantic segmentation model that labels every pixel of an underwater reef photo with one of
seven classes, built to find **coral bleaching** and to run on the Kria KV260 FPGA (DPU-legal ops only).

| | |
|---|---|
| model | `fpn_resnet34_dappm`: ResNet-34 encoder + DAPPM context block + FPN-lite decoder, 22.5 M params |
| classes | BLEACHED, DEAD, ALIVE, ROCK, OCEAN, SAND, OTHER |
| weights | [`weights/MILESTONE_R34_DAPPM_7c.pt`](weights/) (90 MB) |
| headline | **bleached-vs-not-bleached mIoU 0.708 ± 0.004** on 100 held-out photos from 5 reef regions (3 seeds) |

## Results

Benchmark: 100 CoralNet survey photos, 20 from each of Kenya, Red Sea, Chagos, Norfolk Island and
La Paz, all from **reef sites that were held out of training and model selection**, with hand-corrected
7-class masks ([benchmark/](benchmark/)). Scores are pooled over all 100 photos, mean ± sd over 3 training
seeds. The shipped checkpoint is seed 2. Reproduce with `python eval/evaluate.py weights/MILESTONE_R34_DAPPM_7c.pt`.

**Bleaching (the target task)**

| model | bleached-vs-rest mIoU | bleached IoU | precision | recall | F1 |
|---|---|---|---|---|---|
| **this model** (3 seeds) | **0.708 ± 0.004** | **0.544 ± 0.009** | 0.758 | 0.658 | 0.704 |
| same network trained on CoralScapes only (5 seeds) | 0.497 ± 0.025 | 0.174 ± 0.043 | 0.830 | 0.181 | 0.295 |

The CoralScapes-only model is cautious: when it calls a pixel bleached it is usually right, but it finds
fewer than 1 in 5 bleached pixels on reefs it has not seen. This model finds about 2 in 3. Bleached pixels
are 20 % of the benchmark.

Bleached IoU by region: Kenya 0.707, Red Sea 0.512, Chagos 0.315, Norfolk 0.510, La Paz 0.576.

**All classes (pooled mIoU)**

| scheme | mIoU |
|---|---|
| 7 classes | 0.474 ± 0.005 |
| 6 classes, DEAD and ROCK merged (dead coral vs rock is ambiguous even for people) | 0.549 ± 0.008 |
| 4-class health: LIVE / BLEACHED / DEGRADED (dead+rock) / OTHER | 0.564 ± 0.007 |

Full tables, per-class IoU and per-region numbers: [results/RESULTS.md](results/RESULTS.md).

## What is in here

| path | what |
|---|---|
| `train/standalone_train.py` | the whole training pipeline in one file: network, data loading, augmentation, loss, schedule, validation |
| `train/run_train.sh`, `run_train.ps1` | the exact milestone training command (config `P3_rare_E2_all_cs`) |
| `train/configs/` | the milestone config, written out |
| `eval/evaluate.py` | scores any checkpoint on the benchmark (bleaching + 7/6/4-class mIoU) |
| `benchmark/` | 100 hand-corrected test masks, the photo list, and `fetch_photos.py` to download the photos from CoralNet |
| `labeller/` | the region-labelling pipeline that produced the training labels (SAM regions + a vision-language model) |
| `data/` | how to get CoralScapes, and the format of our label exports |
| `results/` | benchmark scores per seed, training logs, validation curves, the original per-seed launch scripts |
| `docs/MODEL_ARCHITECTURE.md` | the network in detail, and the FPGA (DPU) constraints it respects |
| `weights/` | the milestone checkpoint |

## Quick start: run the model

```bash
pip install -r requirements.txt          # install a CUDA build of torch first (pytorch.org)
python benchmark/fetch_photos.py         # ~300 MB of photos from CoralNet, checksum-verified
python eval/evaluate.py weights/MILESTONE_R34_DAPPM_7c.pt
```
This prints the numbers above (about 2 s on a laptop GPU after loading). To use the model in your own code:
```python
import sys; sys.path.insert(0, "train")
import standalone_train as S
model, ck = S.load_ckpt("weights/MILESTONE_R34_DAPPM_7c.pt", device)
predict = S.make_predictor(model, device)          # predict(uint8 HxWx3, H and W multiples of 256) -> HxW class ids
```
Class ids out of the model: 0 BLEACHED, 1 DEAD, 2 ALIVE, 3 ROCK, 4 OCEAN, 5 SAND, 6 OTHER. Feed photos the
way the benchmark does: long side 1024 px, centre-padded to a multiple of 256 (see `eval/evaluate.py`).

## Reproducing the training

Training data:
1. **CoralScapes V2** (public, Red Sea, ~7 GB). `data/coralscapes/` downloads and decodes it. Its 4-class
   health labels are used as *partial* 7-class labels (LIVE→ALIVE, BLEACHED→BLEACHED,
   DEGRADED→{DEAD or ROCK}, OTHER→{ROCK, SAND, OCEAN or OTHER}), so the loss accepts any class in the set.
2. **Our 7-class labels of CoralNet survey photos**: 2,244 training photos from Kenya (870), Red Sea (577),
   Chagos (573), La Paz (135), Raja Ampat (48) and Norfolk Island (41), plus held-out-site validation sets.
   These were made locally with the pipeline in `labeller/` and are **not in this repository**. If you want
   to retrain, ask the team for the label exports; they go in `data/labels/` (format: `data/EXPORT_FORMAT.md`).

Then, from this folder:
```bash
bash train/run_train.sh profile     # 2-epoch throughput check (refuses to train until the GPU is the bottleneck)
bash train/run_train.sh 2           # one seed: 17,880 steps, ~31 min on an RTX 4090
```
Recipe: 512×512 crops, batch 16, AdamW, OneCycle (peak lr 1e-3), fp16, ImageNet init, validation every 447
steps on the held-out-site sets, best-validation checkpoint kept, rare-class crop sampling (`--rare-crop-frac 0.5`:
half the crops are centred on a DEAD, SAND, OCEAN or ALIVE pixel). The shipped model was picked from 39
screening runs (13 data mixes × 3 seeds) and two follow-up phases **by validation only**; the benchmark was
never used to choose anything. The logs of all three seeds are in `results/logs/`.

## Notes

- The benchmark is **test only**. Do not train, tune thresholds or pick checkpoints on it.
- Labels come from a model plus human correction, not from divers in the water: the benchmark masks were
  pre-filled by the labeller and then corrected by hand in our editor.
- One seed is not a result: compare setups by their mean over seeds.
