#!/bin/bash
# Baseline "No latent" students (Table II): same environment, teachers, DAgger schedule, latency model and evaluation as
# the v6g10s L0-L4 students, but the student gets NO z_skill -- only the action one-hot and the normalised clip
# progress (obs group student_nolatent).  3 RL seeds.  Starts after q_exec_A.sh has finished (GPU memory).
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
N=$H/runs/night
V=v6g10s
cd $H/code
export HF_HUB_OFFLINE=1 HNU_BRIDGE_ROOT=$H/data/ref_bridge_video_lift HNU_STAKE_H=0.10 HNU_GEOM_W1=$H/data/ref_stakes_s93_h10 \
  HNU_GEOM_W2=$H/data/ref_stakes_s83_h10 HNU_STAKE_TOL=0.17 HNU_START_FRAC=0.25 HNU_DR_LEVEL=1 HNU_DR_LAG_SYNC=1 \
  HNU_DR_LAG_SENSORS=1 PYTHONPATH=$H/code
log() { echo "[$(date +%H:%M:%S)] $*" >> $N/nolatent.log; }
log "waiting for q_exec_A"
for i in $(seq 1 3000); do grep -q finished $N/q_exec_A.log 2>/dev/null && break; sleep 60; done
CK=$(cat $N/${V}_teachers.txt)
COMMON="--scene geom --z-offsets 0,5,10,15,20,25,30,40,50 --hidden 1024,512,256 --hist 20 --lin-vel --obs-group student_nolatent"
LD=$H/data/z_tables_v5/Z0  # required by the env (z terms exist in other groups); not an input of this student
one() {  # seed gpu
  local s=$1 G=$2 S=$H/runs/students_${V}_nolat_s$1 O=$H/runs/students_${V}_nolat_d2_s$1
  [ -f $S/student_4000.pt ] || HNU_DR=1 HNU_TB_ROOT=$H/runs/logs/rsl_rl/students_${V}_nolat_s$s CUDA_VISIBLE_DEVICES=$G \
    $PY distill_train.py --profile ${V}_nolat $CK --num-envs 4096 --iters 4000 --seed $((1000 * s)) $COMMON \
    --latent-dir $LD --out $S > $N/${V}_nolat_s${s}_E.log 2>&1
  [ -f $S/student_4000.pt ] || { log "seed $s round 1 failed"; return; }
  [ -f $O/student_2000.pt ] || HNU_DR=1 HNU_TB_ROOT=$H/runs/logs/rsl_rl/students_${V}_nolat_d2_s$s CUDA_VISIBLE_DEVICES=$G \
    $PY distill_train.py --profile ${V}_nolat_d2 $CK --num-envs 4096 --iters 2000 --beta-iters 1 --lr 2e-4 --action-noise 0.1 \
    --fail-weight 2 --seed $((2000 * s)) $COMMON --init-from $S/student_4000.pt --latent-dir $LD --out $O > $N/${V}_nolat_d2_s$s.log 2>&1
  [ -f $O/student_2000.pt ] || { log "seed $s round 2 failed"; return; }
  for DRV in 0 1; do for X in "" "--push"; do
    HNU_DR=$DRV HNU_START_FRAC=0 CUDA_VISIBLE_DEVICES=$G $PY eval_student.py --student $O/student_2000.pt --latent-dir $LD \
      --per-clip 8 $X --out $H/runs/results_${V}_s${s}_dr$DRV --tag _NL >> $N/${V}_nolat_eval_s$s.log 2>&1
  done; done
  log "seed $s done"
}
( one 1 0; one 3 0 ) & one 2 1 & wait
log "finished"
