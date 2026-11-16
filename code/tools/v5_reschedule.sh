#!/bin/bash
# Students that died with CUDA OOM at start-up (5 concurrent jobs): reschedule at <= 2 jobs per GPU.
#   now   : v5 Z2, Z3 on GPU1
#   later : v5 Z4 and v4d-bridge Z0 on GPU0 once v5 Z0/Z1 finish; v4d-bridge Z2 on GPU1 once v5 Z2/Z3 finish
# then re-run run_v5.sh (skips finished stages -> eval + language) and evaluate the two v4d baselines.
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
N=$H/runs/night
cd $H/code
pkill -f 'run_v5.s[h]'
export HF_HUB_OFFLINE=1 HNU_BRIDGE_ROOT=$H/data/ref_bridge_video_lift
ck() { cat $H/runs/expert_eval/$1/checkpoint.txt; }
CKD=$(cat $H/runs/bridge_eval/bridge_video_lift_perc/checkpoint.txt)
CK="--ckpt-ThreeStepPoleRun $(ck ThreeStepPoleRun) --ckpt-FiveStepPoleRun $(ck FiveStepPoleRun) --ckpt-CrouchedTraverse $CKD"
v5() {  # P gpu
  rm -rf $H/runs/students_v5/$1
  HNU_TB_ROOT=$H/runs/logs/rsl_rl/students_v5 CUDA_VISIBLE_DEVICES=$2 $PY distill_train.py --profile v5_$1 $CK --num-envs 4096 \
    --iters 4000 --seed 1000 --scene bridge --z-offsets 0,5,10,15,20,25,30,40,50 --hidden 1024,512,256 \
    --latent-dir $H/data/z_tables_v5/$1 --out $H/runs/students_v5/$1 > $N/v5_E_$1.log 2>&1
}
v4d() {  # P gpu
  rm -rf $H/runs/students_bridge/$1
  HNU_TB_ROOT=$H/runs/logs/rsl_rl/students_bridge CUDA_VISIBLE_DEVICES=$2 $PY distill_train.py --profile bridge_$1 $CK \
    --num-envs 4096 --iters 2500 --seed 1000 --scene bridge --latent-dir $H/data/z_tables_v4d/$1 --out $H/runs/students_bridge/$1 \
    > $N/bridge_E_$1.log 2>&1
  for X in "" "--push"; do
    CUDA_VISIBLE_DEVICES=$2 $PY eval_student.py --student $H/runs/students_bridge/$1/student_2500.pt \
      --latent-dir $H/data/z_tables_v4d/$1 --per-clip 8 $X --out $H/runs/results_bridge --tag _$1 >> $N/bridge_F_$1.log 2>&1
  done
}
wait_file() { for i in $(seq 1 600); do [ -f $1 ] && return; sleep 60; done; }
( v5 Z2 1; v4d Z2 1 ) &
( v5 Z3 1 ) &
( wait_file $H/runs/students_v5/Z0/student_4000.pt; v5 Z4 0 ) &
( wait_file $H/runs/students_v5/Z1/student_4000.pt; sleep 60; v4d Z0 0 ) &
wait
nohup bash run_v5.sh > /dev/null 2>&1 < /dev/null
echo "[$(date +%H:%M:%S)] reschedule finished" >> $N/v5.log
