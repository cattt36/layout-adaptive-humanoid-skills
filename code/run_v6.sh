#!/bin/bash
# v6 = v5 + student input improvements (identical for L0-L4; deployment chain unchanged):
#   * estimated base linear velocity (state-estimator stand-in, +-0.2 m/s noise)
#   * proprioceptive history 5 -> 20 steps (0.4 s)
#   * DAgger round 2 additionally up-weights states that fail within 0.5 s (x3); labels remain the teacher's
# Bridge scene, RL seeds 1-3, DAgger 4000 + round 2 2000; evaluation clean / push.  Starts when v5 seed 3 is done.
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
N=$H/runs/night
V=v6
cd $H/code
export HF_HUB_OFFLINE=1 HNU_BRIDGE_ROOT=$H/data/ref_bridge_video_lift
ck() { cat $H/runs/expert_eval/$1/checkpoint.txt; }
CKD=$(cat $H/runs/bridge_eval/bridge_video_lift_perc/checkpoint.txt)
CK="--ckpt-ThreeStepPoleRun $(ck ThreeStepPoleRun) --ckpt-FiveStepPoleRun $(ck FiveStepPoleRun) --ckpt-CrouchedTraverse $CKD"
COMMON="--scene bridge --z-offsets 0,5,10,15,20,25,30,40,50 --hidden 1024,512,256 --hist 20 --lin-vel"
log() { echo "[$(date +%H:%M:%S)] $*" >> $N/$V.log; }
for i in $(seq 1 1500); do grep -q finished $N/v5_seeds.log 2>/dev/null && break; sleep 60; done
log "start"
one() {  # P gpu seed
  local P=$1 G=$2 s=$3 S=$H/runs/students_${V}_s$3 O=$H/runs/students_${V}_d2_s$3
  [ -f $S/$P/student_4000.pt ] || HNU_TB_ROOT=$H/runs/logs/rsl_rl/students_${V}_s$s CUDA_VISIBLE_DEVICES=$G $PY distill_train.py \
    --profile ${V}_$P $CK --num-envs 4096 --iters 4000 --seed $((1000 * s)) $COMMON --latent-dir $H/data/z_tables_v5/$P \
    --out $S/$P > $N/${V}_s${s}_E_$P.log 2>&1
  [ -f $S/$P/student_4000.pt ] || return
  [ -f $O/$P/student_2000.pt ] || HNU_TB_ROOT=$H/runs/logs/rsl_rl/students_${V}_d2_s$s CUDA_VISIBLE_DEVICES=$G $PY distill_train.py \
    --profile ${V}d2_$P $CK --num-envs 4096 --iters 2000 --beta-iters 1 --lr 2e-4 --action-noise 0.1 --fail-weight 2 \
    --seed $((2000 * s)) $COMMON --init-from $S/$P/student_4000.pt --latent-dir $H/data/z_tables_v5/$P --out $O/$P \
    > $N/${V}_d2_s${s}_$P.log 2>&1
  for X in "" "--push"; do
    [ -f $O/$P/student_2000.pt ] && CUDA_VISIBLE_DEVICES=$G $PY eval_student.py --student $O/$P/student_2000.pt \
      --latent-dir $H/data/z_tables_v5/$P --per-clip 8 $X --out $H/runs/results_${V}_d2_s$s --tag _$P >> $N/${V}_F_s${s}_$P.log 2>&1
  done
}
for s in 1 2 3; do
  log "seed $s"
  ( one Z0 0 $s; one Z3 0 $s ) & ( one Z1 0 $s ) & ( one Z2 1 $s; one Z4 1 $s ) & wait
  for P in Z0 Z1 Z2 Z3 Z4; do one $P 1 $s; done
done
$PY tools/bridge_err.py $H/runs/results_${V}_d2_s1 $H/runs/results_${V}_d2_s2 $H/runs/results_${V}_d2_s3 > $N/${V}_summary.txt 2>&1
log "finished"
