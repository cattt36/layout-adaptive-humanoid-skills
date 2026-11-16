#!/bin/bash
# Strong baseline "explicit reference" students (obs group student_ref): same deployable proprioception and
# obstacle features, but the explicit reference command of the motion (reference joint positions / velocities)
# instead of z_skill -- a motion-tracking interface like HOVER / OmniH2O students.  HELD-OUT protocol as v6g10h
# (train-subject clips only, test-subject clips reported), 3 RL seeds.  Starts after v6g10h.
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
N=$H/runs/night
V=v6g10h
cd $H/code
export HF_HUB_OFFLINE=1 HNU_BRIDGE_ROOT=$H/data/ref_bridge_video_lift HNU_STAKE_H=0.10 HNU_GEOM_W1=$H/data/ref_stakes_s93_h10 \
  HNU_GEOM_W2=$H/data/ref_stakes_s83_h10 HNU_STAKE_TOL=0.17 HNU_START_FRAC=0.25 HNU_DR_LEVEL=1 HNU_DR_LAG_SYNC=1 \
  HNU_DR_LAG_SENSORS=1 PYTHONPATH=$H/code
ALLOW=$H/data/clip_split_v5/clips_train.txt
log() { echo "[$(date +%H:%M:%S)] $*" >> $N/${V}_ref.log; }
log "waiting for $V"
for i in $(seq 1 5000); do grep -q finished $N/$V.log 2>/dev/null && break; sleep 60; done
CK=$(cat $N/${V}_teachers.txt)
BASE="--scene geom --z-offsets 0,5,10,15,20,25,30,40,50 --hidden 1024,512,256 --hist 20 --lin-vel --obs-group student_ref"
LD=$H/data/z_tables_v5/Z0  # required by the env; not an input of this student
free_mb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $1 | tr -d " "; }
one() {  # seed gpu
  local s=$1 G=$2 S=$H/runs/students_${V}_s$1/REF O=$H/runs/students_${V}_d2_s$1/REF
  while [ "$(free_mb $G)" -lt 17000 ]; do sleep 60; done
  [ -f $S/student_4000.pt ] || HNU_CLIPS_ALLOW=$ALLOW HNU_DR=1 HNU_TB_ROOT=$H/runs/logs/rsl_rl/students_${V}_s$s \
    CUDA_VISIBLE_DEVICES=$G $PY distill_train.py --profile ${V}_REF $CK --num-envs 4096 --iters 4000 --seed $((1000 * s)) \
    $BASE --latent-dir $LD --out $S > $N/${V}_s${s}_E_REF.log 2>&1
  [ -f $S/student_4000.pt ] || { log "REF s$s round 1 failed"; return; }
  [ -f $O/student_2000.pt ] || HNU_CLIPS_ALLOW=$ALLOW HNU_DR=1 HNU_TB_ROOT=$H/runs/logs/rsl_rl/students_${V}_d2_s$s \
    CUDA_VISIBLE_DEVICES=$G $PY distill_train.py --profile ${V}d2_REF $CK --num-envs 4096 --iters 2000 --beta-iters 1 --lr 2e-4 \
    --action-noise 0.1 --fail-weight 2 --seed $((2000 * s)) $BASE --init-from $S/student_4000.pt --latent-dir $LD \
    --out $O > $N/${V}_d2_s${s}_REF.log 2>&1
  [ -f $O/student_2000.pt ] || { log "REF s$s round 2 failed"; return; }
  for DRV in 0 1; do for X in "" "--push"; do
    HNU_DR=$DRV HNU_START_FRAC=0 CUDA_VISIBLE_DEVICES=$G $PY eval_student.py --student $O/student_2000.pt --latent-dir $LD \
      --per-clip 8 $X --out $H/runs/results_${V}_s${s}_dr$DRV --tag _REF >> $N/${V}_F_s${s}_REF.log 2>&1
  done; done
  log "REF s$s done"
}
( one 1 0; one 3 0 ) & one 2 1 & wait
log "finished"
