#!/bin/bash
source /workspace/milestone/env_pods.sh 2>/dev/null || source /workspace/milestone/env_04d7n4uk1biscq.sh
cd /workspace/milestone/runs/P3_rare_E2_all_cs_s1
cp /workspace/milestone/profile/standalone_profile_gate.json . 2>/dev/null
echo $$ > PID
echo ${MILESTONE_POD:-04d7n4uk1biscq} > POD
python /workspace/milestone/code/standalone_train.py --classes 7 --seed 1 --tag P3_rare_E2_all_cs_s1 --out . --cache /root/cache_v2 --workers 12 --total-steps 17880 --eval-every 447 --persist-every-eval --extra-train /workspace/milestone/exports/sonnet7_kenya__74541dee --extra-train /workspace/milestone/exports/sonnet7_red_sea__55360602 --extra-train /workspace/milestone/exports/sonnet7_chagos__af41443c --extra-train /workspace/milestone/exports/sonnet7_la_paz__5cc16a1e --extra-train /workspace/milestone/exports/sonnet7_norfolk__e03a6a40 --extra-train /workspace/milestone/exports/sonnet7_raja__4981ad5d --val-dir /workspace/milestone/exports/sonnet7_kenya_heldout__53d13367 --val-dir /workspace/milestone/exports/sonnet7_red_sea_heldout__dbf8f97e --val-dir /workspace/milestone/exports/sonnet7_chagos_heldout__2788aaaf --val-dir /workspace/milestone/exports/sonnet7_la_paz_heldout__3d175828 --val-dir /workspace/milestone/exports/sonnet7_norfolk_heldout__6894e499 --val-split test --rare-crop-frac 0.5 > train.log 2>&1
echo $? > EXIT
