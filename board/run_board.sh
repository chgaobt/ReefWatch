#!/bin/sh
# On the KV260 (Kria Ubuntu 22.04 or PetaLinux): start the board program from this directory.
#   ./run_board.sh                                   pipe test (backend none), map + status port on
#   ./run_board.sh --backend vart --map-hz 1         DPU inference, one-way (Unreal -sendonly)
#   ./run_board.sh --backend vart --map-hz 1 --reply DPU, masks + map sent back (Unreal normal mode)
#   ./run_board.sh --backend vart --xmodel models/coral_milestone_7c_512x1024.xmodel
# Ctrl+C stops it; outputs go to out/board_<date>_<time>/ (see README.md).
# PYTHON=/path/to/python3 ./run_board.sh ...   uses another interpreter (e.g. where vart/xir live).
cd "$(dirname "$0")" || exit 1
exec "${PYTHON:-python3}" -m reefwatch_board "$@"
