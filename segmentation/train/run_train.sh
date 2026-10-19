#!/usr/bin/env bash
# Train the milestone 1 model (7-class R34-DAPPM, config P3_rare_E2_all_cs).
# Run from the segmentation/ folder:
#   bash train/run_train.sh profile      # 2-epoch throughput check; must pass before training
#   bash train/run_train.sh 2            # train seed 2 (the shipped checkpoint); also run 0 and 1
# Needs: data/coralscapes_v2 (see data/README.md) and the 7-class label exports in
# data/labels/ (sonnet7_<location> and sonnet7_<location>_heldout; available from the team).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CORAL_DATA_V2="${CORAL_DATA_V2:-$ROOT/data/coralscapes_v2}"
export CORAL_V2_CACHE="${CORAL_V2_CACHE:-$ROOT/cache_v2}"
export PYTHONIOENCODING=utf-8
LABELS="${LABELS:-$ROOT/data/labels}"
PY="${PY:-python}"
WORKERS="${WORKERS:-8}"

TRAIN_LOCS=(kenya red_sea chagos la_paz norfolk raja)   # raja is train-only (no held-out split)
VAL_LOCS=(kenya red_sea chagos la_paz norfolk)

ARGS=(--classes 7 --total-steps 17880 --eval-every 447 --persist-every-eval
      --rare-crop-frac 0.5 --val-split test --workers "$WORKERS")
for l in "${TRAIN_LOCS[@]}"; do ARGS+=(--extra-train "$LABELS/sonnet7_$l"); done
for l in "${VAL_LOCS[@]}"; do ARGS+=(--val-dir "$LABELS/sonnet7_${l}_heldout"); done

MODE="${1:-2}"; shift || true
mkdir -p "$ROOT/runs"
if [ "$MODE" = "profile" ]; then
  exec "$PY" "$ROOT/train/standalone_train.py" --profile 2 --out "$ROOT/runs" "${ARGS[@]}" "$@"
fi
SEED="$MODE"
exec "$PY" "$ROOT/train/standalone_train.py" --seed "$SEED" --tag "P3_rare_E2_all_cs_s$SEED" \
     --out "$ROOT/runs" "${ARGS[@]}" "$@"
