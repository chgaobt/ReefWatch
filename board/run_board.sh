#!/bin/bash
# On the KV260 (Kria Ubuntu 22.04 or PetaLinux): start the board program from this directory.
#   ./run_board.sh                                   pipe test (backend none), map + status port on
#   ./run_board.sh --backend vart --map-hz 1         DPU inference, one-way (Unreal -sendonly)
#   ./run_board.sh --backend vart --map-hz 1 --reply DPU, masks + map sent back (Unreal normal mode)
#   ./run_board.sh --backend vart --xmodel models/coral_milestone_7c_512x1024.xmodel
# Ctrl+C stops it; outputs go to out/board_<date>_<time>/ (see README.md).
# PYTHON=/path/to/python3 ./run_board.sh ...   uses another interpreter (e.g. where vart/xir live).
#
# Kria Ubuntu with the Vitis AI 3.5 / DPU-PYNQ install (SETUP_UBUNTU.md route A): vart, xir and
# pynq_dpu live in the PYNQ venv, so its two profile scripts are sourced here when present. That
# route programs the FPGA from Python and needs root: sudo ./run_board.sh --backend vart ...
cd "$(dirname "$0")" || exit 1
for f in /etc/profile.d/pynq_venv.sh /etc/profile.d/xrt_setup.sh; do
    [ -z "$PYTHON" ] && [ -f "$f" ] && . "$f"
done
exec "${PYTHON:-python3}" -m reefwatch_board "$@"
