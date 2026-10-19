# Milestone 1 results

All numbers come from `eval/evaluate.py` on the 100-photo benchmark (`benchmark/`). They are pooled
pixel counts (one confusion matrix over all photos, IGNORE pixels excluded), mean ± sd over training seeds.
The raw per-seed outputs are in `benchmark_scores/`.

Models:
- **milestone**: config `P3_rare_E2_all_cs`, seeds 0, 1, 2. The shipped checkpoint `weights/MILESTONE_R34_DAPPM_7c.pt`
  is seed 2.
- **CoralScapes-only baseline**: the same network and recipe, 4-class head, trained on CoralScapes V2 only,
  seeds 0–4. This is our previous model.

## Bleaching

Binary task: BLEACHED vs everything else.
- bleached IoU = TP / (TP + FP + FN)
- not-bleached IoU = TN / (TN + FP + FN)
- binary mIoU = their mean

| model | seeds | binary mIoU | bleached IoU | not-bleached IoU | precision | recall | F1 |
|---|---|---|---|---|---|---|---|
| milestone | 3 | **0.708 ± 0.004** | **0.544 ± 0.009** | 0.872 ± 0.002 | 0.758 ± 0.018 | 0.658 ± 0.025 | 0.704 ± 0.007 |
| CoralScapes-only | 5 | 0.497 ± 0.025 | 0.174 ± 0.043 | 0.821 ± 0.006 | 0.830 ± 0.024 | 0.181 ± 0.046 | 0.295 ± 0.065 |

Per region, milestone model:

| region | bleached share of pixels | bleached IoU | binary mIoU | precision | recall | CoralScapes-only bleached IoU |
|---|---|---|---|---|---|---|
| Kenya | 22.9 % | 0.707 ± 0.047 | 0.808 ± 0.029 | 0.875 | 0.788 | 0.132 ± 0.050 |
| Red Sea | 16.1 % | 0.512 ± 0.036 | 0.703 ± 0.022 | 0.743 | 0.623 | 0.173 ± 0.015 |
| Chagos | 10.3 % | 0.315 ± 0.031 | 0.605 ± 0.016 | 0.515 | 0.451 | 0.112 ± 0.034 |
| Norfolk Island | 21.2 % | 0.510 ± 0.020 | 0.669 ± 0.016 | 0.640 | 0.715 | 0.125 ± 0.047 |
| La Paz | 28.7 % | 0.576 ± 0.007 | 0.711 ± 0.004 | 0.934 | 0.600 | 0.268 ± 0.077 |
| pooled | 20.2 % | 0.544 ± 0.009 | 0.708 ± 0.004 | 0.758 | 0.658 | 0.174 ± 0.043 |

Chagos is the weak region. It also has the least bleaching in the benchmark (10 % of pixels), so each miss
weighs more.

## All classes

Pooled mIoU, milestone model (3 seeds):

| scheme | pooled | Kenya | Red Sea | Chagos | Norfolk | La Paz |
|---|---|---|---|---|---|---|
| 7 classes | 0.474 ± 0.005 | 0.430 | 0.502 | 0.389 | 0.338 | 0.422 |
| 6 classes (DEAD+ROCK merged) | 0.549 ± 0.008 | 0.487 | 0.583 | 0.451 | 0.396 | 0.496 |
| 4-class health | 0.564 ± 0.007 | 0.567 | 0.609 | 0.480 | 0.473 | 0.595 |

The CoralScapes-only baseline scores 0.241 ± 0.012 on the 4-class health scheme. It has no 7-class head.

Per-class IoU (7 classes, pooled):

| BLEACHED | DEAD | ALIVE | ROCK | OCEAN | SAND | OTHER |
|---|---|---|---|---|---|---|
| 0.544 | 0.098 | 0.513 | 0.561 | 0.697 | 0.526 | 0.380 |

DEAD is the hard class: dead coral and bare rock look alike, and the labels themselves disagree on them. That is
why the 6-class lenient scheme is also reported. Merged, DEAD+ROCK scores 0.632.

## How the model was chosen

Selection used validation sets only. These are photos from held-out reef sites of each region; they are
separate from the benchmark photos, which come from other held-out sites.

| stage | what | best |
|---|---|---|
| screening | 13 data mixes × 3 seeds (single region, all regions, leave-one-region-out, ± CoralScapes, CoralScapes-init fine-tune) | all regions + CoralScapes, val mIoU 0.468 |
| phase 2 | the same at 2× training steps | 0.472 |
| phase 3 | + rare-class crop sampling, 2× steps | **0.478** (seeds 0.481 / 0.461 / 0.493) |

The shipped checkpoint is phase 3, seed 2: validation mIoU 0.493, best epoch 29 of 40.

Also tried, none adopted for the milestone checkpoint:
- adding Unreal Engine simulator frames with synthetic bleaching. On real photos this matched the real-data
  controls; on simulator frames it raised bleaching precision.
- adding labelled photos of the source reefs of our 3D reef scans
- a bleach-weighted loss
- 1536 px training resolution (half the speed)
- more labelled photos for the two smallest regions

## Logs

`logs/seed{0,1,2}/` contains, for each training seed:
- `train.log`: the full console log (data counts, resolved config, one line per validation)
- `val.jsonl`: the validation curve
- `train_<run>.json`: the run record (config, per-epoch metrics, final held-out-site test scores)
- `run_on_pod.sh`: the exact command that was run on the cloud GPU

The profile check that gates training measured 50.5 s/epoch at 95.4 % GPU utilisation on an RTX 4090.
