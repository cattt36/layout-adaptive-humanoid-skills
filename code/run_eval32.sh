#!/bin/bash
# Item 1: re-evaluate every held-out student (L0-L4, No-latent, explicit reference; all seeds) and the teacher bank
# with 32 trials per clip instead of 8 (lower evaluation noise; identical for all models).  Results:
# results_v6g10h32_s<seed>_dr{0,1} (tags as before).  Starts after the explicit-reference baseline.
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
N=$H/runs/night
V=v6g10h
W=${V}32
cd $H/code
export HF_HUB_OFFLINE=1 HNU_BRIDGE_ROOT=$H/data/ref_bridge_video_lift HNU_STAKE_H=0.10 HNU_GEOM_W1=$H/data/ref_stakes_s93_h10 \
  HNU_GEOM_W2=$H/data/ref_stakes_s83_h10 HNU_STAKE_TOL=0.17 HNU_DR_LEVEL=1 HNU_DR_LAG_SYNC=1 HNU_DR_LAG_SENSORS=1 PYTHONPATH=$H/code
log() { echo "[$(date +%H:%M:%S)] $*" >> $N/$W.log; }
log "waiting for seeds 4-5 + explicit-reference baseline (v6g10h_s45)"
for i in $(seq 1 8000); do grep -q finished $N/${V}_s45.log 2>/dev/null && break; sleep 60; done
CK=$(cat $N/${V}_teachers.txt)
free_mb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $1 | tr -d " "; }
for DRV in 0 1; do for X in "" "--push"; do
  HNU_DR=$DRV HNU_START_FRAC=0 CUDA_VISIBLE_DEVICES=0 $PY eval_student.py --teacher $CK --scene geom --z-offsets 0,5,10,15,20,25,30,40,50 \
    --latent-dir $H/data/z_tables_v5/Z4 --per-clip 32 $X --out $H/runs/results_${W}_bank_teacher_dr$DRV --tag _teacher >> $N/${W}_teacher.log 2>&1
done; done
log "teacher bank evaluated"
JOBS=(); for s in 1 2 3 4 5; do for P in Z0 Z1 Z2 Z3 Z4 NL REF; do
  [ -f $H/runs/students_${V}_d2_s$s/$P/student_2000.pt ] && JOBS+=("$P $s"); done; done
evalone() {  # P seed gpu
  local P=$1 s=$2 G=$3 LD=$H/data/z_tables_v5/$1
  [ $P = NL ] || [ $P = REF ] && LD=$H/data/z_tables_v5/Z0
  for DRV in 0 1; do for X in "" "--push"; do
    local f=$H/runs/results_${W}_s${s}_dr$DRV/eval_student_${P}$([ -n "$X" ] && echo _push).json
    [ -f $f ] || HNU_DR=$DRV HNU_START_FRAC=0 CUDA_VISIBLE_DEVICES=$G $PY eval_student.py \
      --student $H/runs/students_${V}_d2_s$s/$P/student_2000.pt --latent-dir $LD --per-clip 32 $X \
      --out $H/runs/results_${W}_s${s}_dr$DRV --tag _$P >> $N/${W}_eval.log 2>&1
  done; done
  log "$P s$s evaluated"
}
Q=$N/${W}_queue; echo 0 > $Q.next; : > $Q.lock
worker() {
  while true; do
    while [ "$(free_mb $1)" -lt 9000 ]; do sleep 60; done
    local i=$(flock $Q.lock bash -c "i=\$(cat $Q.next); echo \$((i + 1)) > $Q.next; echo \$i")
    [ $i -ge ${#JOBS[@]} ] && return
    evalone ${JOBS[$i]} $1
  done
}
worker 0 & worker 1 & wait
log "finished (${#JOBS[@]} students)"
