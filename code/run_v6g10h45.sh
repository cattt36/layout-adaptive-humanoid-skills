#!/bin/bash
# Item 2 (decided before any held-out result): RL seeds 4 and 5 for L0-L4 of the held-out batch v6g10h
# + the explicit-reference baseline (REF, obs group student_ref, seeds 1-3) in ONE free-memory queue,
# so that every level has 5 seeds.  Identical settings to run_v6g10h.sh.  Starts after v6g10h; shares the GPUs with
# the explicit-reference baseline through the free-memory check.
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
N=$H/runs/night
V=v6g10h
cd $H/code
export HF_HUB_OFFLINE=1 HNU_BRIDGE_ROOT=$H/data/ref_bridge_video_lift HNU_STAKE_H=0.10 HNU_GEOM_W1=$H/data/ref_stakes_s93_h10 \
  HNU_GEOM_W2=$H/data/ref_stakes_s83_h10 HNU_STAKE_TOL=0.17 HNU_START_FRAC=0.25 HNU_DR_LEVEL=1 HNU_DR_LAG_SYNC=1 \
  HNU_DR_LAG_SENSORS=1 PYTHONPATH=$H/code
ALLOW=$H/data/clip_split_v5/clips_train.txt
log() { echo "[$(date +%H:%M:%S)] $*" >> $N/${V}_s45.log; }
log "waiting for $V"
for i in $(seq 1 5000); do grep -q finished $N/$V.log 2>/dev/null && break; sleep 60; done
CK=$(cat $N/${V}_teachers.txt)
ZO="--z-offsets 0,5,10,15,20,25,30,40,50"
BASE="--scene geom --hidden 1024,512,256 --hist 20 --lin-vel"
free_mb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $1 | tr -d " "; }
one() {  # P gpu seed
  local P=$1 G=$2 s=$3 S=$H/runs/students_${V}_s$3 O=$H/runs/students_${V}_d2_s$3 LD=$H/data/z_tables_v5/$1 EXTRA=""
  if [ $P = REF ]; then LD=$H/data/z_tables_v5/Z0; EXTRA="--obs-group student_ref"; fi
  [ -f $S/$P/student_4000.pt ] || HNU_CLIPS_ALLOW=$ALLOW HNU_DR=1 HNU_TB_ROOT=$H/runs/logs/rsl_rl/students_${V}_s$s \
    CUDA_VISIBLE_DEVICES=$G $PY distill_train.py --profile ${V}_$P $CK --num-envs 4096 --iters 4000 --seed $((1000 * s)) \
    $BASE $ZO $EXTRA --latent-dir $LD --out $S/$P > $N/${V}_s${s}_E_$P.log 2>&1
  [ -f $S/$P/student_4000.pt ] || { log "$P s$s round 1 failed"; return; }
  [ -f $O/$P/student_2000.pt ] || HNU_CLIPS_ALLOW=$ALLOW HNU_DR=1 HNU_TB_ROOT=$H/runs/logs/rsl_rl/students_${V}_d2_s$s \
    CUDA_VISIBLE_DEVICES=$G $PY distill_train.py --profile ${V}d2_$P $CK --num-envs 4096 --iters 2000 --beta-iters 1 --lr 2e-4 \
    --action-noise 0.1 --fail-weight 2 --seed $((2000 * s)) $BASE $ZO $EXTRA --init-from $S/$P/student_4000.pt \
    --latent-dir $LD --out $O/$P > $N/${V}_d2_s${s}_$P.log 2>&1
  [ -f $O/$P/student_2000.pt ] || { log "$P s$s round 2 failed"; return; }
  for DRV in 0 1; do for X in "" "--push"; do
    HNU_DR=$DRV HNU_START_FRAC=0 CUDA_VISIBLE_DEVICES=$G $PY eval_student.py --student $O/$P/student_2000.pt \
      --latent-dir $LD --per-clip 8 $X --out $H/runs/results_${V}_s${s}_dr$DRV --tag _$P >> $N/${V}_F_s${s}_$P.log 2>&1
  done; done
  log "$P s$s done"
}
# one queue for the explicit-reference baseline (REF, seeds 1-3; replaces run_ref.sh) and the seeds 4-5 of L0-L4
JOBS=("REF 1" "REF 2" "REF 3"); for s in 4 5; do for P in Z0 Z1 Z2 Z3 Z4; do JOBS+=("$P $s"); done; done
Q=$N/${V}_s45_queue; echo 0 > $Q.next; : > $Q.lock
worker() {
  while true; do
    while [ "$(free_mb $1)" -lt 17000 ]; do sleep 90; done
    local i=$(flock $Q.lock bash -c "i=\$(cat $Q.next); echo \$((i + 1)) > $Q.next; echo \$i")
    [ $i -ge ${#JOBS[@]} ] && return
    one ${JOBS[$i]%% *} $1 ${JOBS[$i]##* }
  done
}
worker 0 & worker 1 & wait
for j in "${JOBS[@]}"; do g=1; [ "$(free_mb 0)" -ge 17000 ] && g=0; one ${j%% *} $g ${j##* }; done
n=$(ls $H/runs/students_${V}_d2_s[45]/Z*/student_2000.pt $H/runs/students_${V}_d2_s[123]/REF/student_2000.pt 2>/dev/null | wc -l)
log "students done: $n of 13 (REF x3 + L0-L4 seeds 4-5)"
[ "$n" = 13 ] && log "finished"
