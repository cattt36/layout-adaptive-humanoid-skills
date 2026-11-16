#!/bin/bash
# Restricted look-ahead (hypothesis: when the student cannot see the future of the reference, the timing information
# a level's z_skill carries -- human touchdown / loading in L3 / L4 -- matters for execution).
# Identical to the v6g10s students (teachers, DAgger 4000 + 2000, latency model, evaluation) except that the
# student receives ONLY z_skill(t) (z offsets "0" instead of 0..50).  L0-L4 x 3 RL seeds.  Starts after the
# no-latent baseline.
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
N=$H/runs/night
V=v6g10s
W=${V}_la0
cd $H/code
export HF_HUB_OFFLINE=1 HNU_BRIDGE_ROOT=$H/data/ref_bridge_video_lift HNU_STAKE_H=0.10 HNU_GEOM_W1=$H/data/ref_stakes_s93_h10 \
  HNU_GEOM_W2=$H/data/ref_stakes_s83_h10 HNU_STAKE_TOL=0.17 HNU_START_FRAC=0.25 HNU_DR_LEVEL=1 HNU_DR_LAG_SYNC=1 \
  HNU_DR_LAG_SENSORS=1 PYTHONPATH=$H/code
log() { echo "[$(date +%H:%M:%S)] $*" >> $N/$W.log; }
log "waiting for the no-latent baseline"
for i in $(seq 1 3000); do grep -q finished $N/nolatent.log 2>/dev/null && break; sleep 60; done
CK=$(cat $N/${V}_teachers.txt)
COMMON="--scene geom --z-offsets 0 --hidden 1024,512,256 --hist 20 --lin-vel"
one() {  # P gpu seed
  local P=$1 G=$2 s=$3 S=$H/runs/students_${W}_s$3 O=$H/runs/students_${W}_d2_s$3
  [ -f $S/$P/student_4000.pt ] || HNU_DR=1 HNU_TB_ROOT=$H/runs/logs/rsl_rl/students_${W}_s$s CUDA_VISIBLE_DEVICES=$G \
    $PY distill_train.py --profile ${W}_$P $CK --num-envs 4096 --iters 4000 --seed $((1000 * s)) $COMMON \
    --latent-dir $H/data/z_tables_v5/$P --out $S/$P > $N/${W}_s${s}_E_$P.log 2>&1
  [ -f $S/$P/student_4000.pt ] || { log "$P s$s round 1 failed"; return; }
  [ -f $O/$P/student_2000.pt ] || HNU_DR=1 HNU_TB_ROOT=$H/runs/logs/rsl_rl/students_${W}_d2_s$s CUDA_VISIBLE_DEVICES=$G \
    $PY distill_train.py --profile ${W}d2_$P $CK --num-envs 4096 --iters 2000 --beta-iters 1 --lr 2e-4 --action-noise 0.1 \
    --fail-weight 2 --seed $((2000 * s)) $COMMON --init-from $S/$P/student_4000.pt --latent-dir $H/data/z_tables_v5/$P \
    --out $O/$P > $N/${W}_d2_s${s}_$P.log 2>&1
  [ -f $O/$P/student_2000.pt ] || { log "$P s$s round 2 failed"; return; }
  for DRV in 0 1; do for X in "" "--push"; do
    HNU_DR=$DRV HNU_START_FRAC=0 CUDA_VISIBLE_DEVICES=$G $PY eval_student.py --student $O/$P/student_2000.pt \
      --latent-dir $H/data/z_tables_v5/$P --per-clip 8 $X --out $H/runs/results_${W}_s${s}_dr$DRV --tag _$P >> $N/${W}_F_s${s}_$P.log 2>&1
  done; done
  log "$P s$s done"
}
JOBS=(); for s in 1 2 3; do for P in Z0 Z1 Z2 Z3 Z4; do JOBS+=("$P $s"); done; done
queue() { for i in "${!JOBS[@]}"; do [ $((i % 2)) = $1 ] && one ${JOBS[$i]%% *} $1 ${JOBS[$i]##* }; done; }
queue 0 & queue 1 & wait
log "finished"
