#!/bin/bash
# C-2: PPO fine-tuning of the v5 bridge-scene students (all five levels, identical settings), then evaluation.
# Waits for the v5 DAgger students; at most two jobs per GPU.
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
N=$H/runs/night
cd $H/code
export HF_HUB_OFFLINE=1 HNU_BRIDGE_ROOT=$H/data/ref_bridge_video_lift
CR=$(cat $H/runs/bridge_eval/bridge_video_lift_perc/checkpoint.txt)
S=$H/runs/students_v5
O=$H/runs/students_v5_ppo
R=$H/runs/results_v5_ppo
mkdir -p $O $R
log() { echo "[$(date +%H:%M:%S)] $*" >> $N/v5_ppo.log; }
for P in Z0 Z1 Z2 Z3 Z4; do
  for i in $(seq 1 900); do [ -f $S/$P/student_4000.pt ] && break; sleep 60; done
done
sleep 120
log "PPO fine-tuning"
ppo() {  # P gpu
  [ -f $O/$1/student_ppo.pt ] && return
  CUDA_VISIBLE_DEVICES=$2 $PY student_ppo.py --student $S/$1/student_4000.pt --latent-dir $H/data/z_tables_v5/$1 \
    --critic-from $CR --iters 1500 --out $O/$1 > $N/v5_ppo_$1.log 2>&1
  for X in "" "--push"; do
    CUDA_VISIBLE_DEVICES=$2 $PY eval_student.py --student $O/$1/student_ppo.pt --latent-dir $H/data/z_tables_v5/$1 \
      --per-clip 8 $X --out $R --tag _$1 >> $N/v5_ppo_F_$1.log 2>&1
  done
}
( ppo Z0 0; ppo Z4 0 ) & ( ppo Z1 0 ) & ( ppo Z2 1; ppo Z3 1 ) & wait
for P in Z0 Z1 Z2 Z3 Z4; do ppo $P 1; done  # retry
log "PPO finished"
