#!/bin/bash
# v5 DAgger round 2 (students stay pure DAgger): continue every level's v5 student for 2000 iterations with
# DART-style executed-action noise (sigma 0.1 on half of the envs; teacher labels clean) and a lower learning rate,
# so the student is also trained on recovery states.  Identical settings for L0-L4.  Then evaluate.
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
N=$H/runs/night
cd $H/code
export HF_HUB_OFFLINE=1 HNU_BRIDGE_ROOT=$H/data/ref_bridge_video_lift HNU_TB_ROOT=$H/runs/logs/rsl_rl/students_v5_d2
ck() { cat $H/runs/expert_eval/$1/checkpoint.txt; }
CKD=$(cat $H/runs/bridge_eval/bridge_video_lift_perc/checkpoint.txt)
CK="--ckpt-ThreeStepPoleRun $(ck ThreeStepPoleRun) --ckpt-FiveStepPoleRun $(ck FiveStepPoleRun) --ckpt-CrouchedTraverse $CKD"
S=$H/runs/students_v5
O=$H/runs/students_v5_d2
R=$H/runs/results_v5_d2
mkdir -p $O $R
log() { echo "[$(date +%H:%M:%S)] $*" >> $N/v5_d2.log; }
for P in Z0 Z1 Z2 Z3 Z4; do
  for i in $(seq 1 900); do [ -f $S/$P/student_4000.pt ] && break; sleep 60; done
done
for i in $(seq 1 120); do pgrep -f 'v5_reschedul[e]' > /dev/null || break; sleep 60; done
log "DAgger round 2"
d2() {  # P gpu
  [ -f $O/$1/student_2000.pt ] || \
  CUDA_VISIBLE_DEVICES=$2 $PY distill_train.py --profile v5d2_$1 $CK --num-envs 4096 --iters 2000 --beta-iters 1 --lr 2e-4 \
    --action-noise 0.1 --seed 2000 --scene bridge --z-offsets 0,5,10,15,20,25,30,40,50 --hidden 1024,512,256 \
    --init-from $S/$1/student_4000.pt --latent-dir $H/data/z_tables_v5/$1 --out $O/$1 > $N/v5_d2_$1.log 2>&1
  for X in "" "--push"; do
    [ -f $O/$1/student_2000.pt ] && CUDA_VISIBLE_DEVICES=$2 $PY eval_student.py --student $O/$1/student_2000.pt \
      --latent-dir $H/data/z_tables_v5/$1 --per-clip 8 $X --out $R --tag _$1 >> $N/v5_d2_F_$1.log 2>&1
  done
}
( d2 Z0 0; d2 Z4 0 ) & ( d2 Z1 0 ) & ( d2 Z2 1; d2 Z3 1 ) & wait
for P in Z0 Z1 Z2 Z3 Z4; do [ -f $R/eval_student_$P.json ] || d2 $P 1; done
log "DAgger round 2 finished"
