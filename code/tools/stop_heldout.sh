#!/bin/bash
# Stop the held-out pipeline (GPU 0 occupied by a foreign process -> cascading OOM) incl. its students.
for p in $(pgrep -f "bash run_v6g10h.sh"); do kill $p; done
sleep 1
for p in $(pgrep -f "distill_train.py --profile v6g10h"); do kill $p; done
for p in $(pgrep -f "eval_student.py --student /root/gpufree-data/hnu_v2/runs/students_v6g10h"); do kill $p; done
sleep 5
echo "[$(date +%H:%M:%S)] stopped: GPU 0 occupied by a foreign process (cascading OOM); restart with a free-memory check" \
  >> /root/gpufree-data/hnu_v2/runs/night/v6g10h.log
pgrep -af "run_v6g10h|v6g10h_" | cut -c1-100
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
