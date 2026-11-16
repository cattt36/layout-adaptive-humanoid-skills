#!/bin/bash
# Item 3: LOW-DATA distillation (data efficiency).  Identical to v6g10h (held-out protocol, teachers, scene, latency,
# DAgger 4000 + 2000) but the students are trained on ONE training subject only (subject_1 of the train pool, fixed
# before any result: data/clip_split_v5/clips_train_subject1.txt, 22 of 47 train clips).  Evaluated on the same
# unseen test-subject clips, 32 trials per clip.  L0-L4 x 3 RL seeds.  Starts after the 32-trial re-evaluation.
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
N=$H/runs/night
V=v6g10q
cd $H/code
export HF_HUB_OFFLINE=1 HNU_BRIDGE_ROOT=$H/data/ref_bridge_video_lift HNU_STAKE_H=0.10 HNU_GEOM_W1=$H/data/ref_stakes_s93_h10 \
  HNU_GEOM_W2=$H/data/ref_stakes_s83_h10 HNU_STAKE_TOL=0.17 HNU_START_FRAC=0.25 HNU_DR_LEVEL=1 HNU_DR_LAG_SYNC=1 \
  HNU_DR_LAG_SENSORS=1 PYTHONPATH=$H/code
ALLOW=$H/data/clip_split_v5/clips_train_subject1.txt
log() { echo "[$(date +%H:%M:%S)] $*" >> $N/$V.log; }
log "waiting for the 32-trial re-evaluation (v6g10h32)"
for i in $(seq 1 8000); do grep -q finished $N/v6g10h32.log 2>/dev/null && break; sleep 60; done
CK=$(cat $N/v6g10h_teachers.txt)
echo "$CK" > $N/${V}_teachers.txt
for DRV in 0 1; do
  [ -e $H/runs/results_${V}_bank_teacher_dr$DRV ] || ln -s $H/runs/results_v6g10h32_bank_teacher_dr$DRV $H/runs/results_${V}_bank_teacher_dr$DRV
done
log "students: training clips $(wc -l < $ALLOW) (one train subject)"
ZO="--z-offsets 0,5,10,15,20,25,30,40,50"
BASE="--scene geom --hidden 1024,512,256 --hist 20 --lin-vel"
free_mb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $1 | tr -d " "; }
one() {  # P gpu seed
  local P=$1 G=$2 s=$3 S=$H/runs/students_${V}_s$3 O=$H/runs/students_${V}_d2_s$3 LD=$H/data/z_tables_v5/$1
  [ -f $S/$P/student_4000.pt ] || HNU_CLIPS_ALLOW=$ALLOW HNU_DR=1 HNU_TB_ROOT=$H/runs/logs/rsl_rl/students_${V}_s$s \
    CUDA_VISIBLE_DEVICES=$G $PY distill_train.py --profile ${V}_$P $CK --num-envs 4096 --iters 4000 --seed $((1000 * s)) \
    $BASE $ZO --latent-dir $LD --out $S/$P > $N/${V}_s${s}_E_$P.log 2>&1
  [ -f $S/$P/student_4000.pt ] || { log "$P s$s round 1 failed"; return; }
  [ -f $O/$P/student_2000.pt ] || HNU_CLIPS_ALLOW=$ALLOW HNU_DR=1 HNU_TB_ROOT=$H/runs/logs/rsl_rl/students_${V}_d2_s$s \
    CUDA_VISIBLE_DEVICES=$G $PY distill_train.py --profile ${V}d2_$P $CK --num-envs 4096 --iters 2000 --beta-iters 1 --lr 2e-4 \
    --action-noise 0.1 --fail-weight 2 --seed $((2000 * s)) $BASE $ZO --init-from $S/$P/student_4000.pt \
    --latent-dir $LD --out $O/$P > $N/${V}_d2_s${s}_$P.log 2>&1
  [ -f $O/$P/student_2000.pt ] || { log "$P s$s round 2 failed"; return; }
  for DRV in 0 1; do for X in "" "--push"; do
    HNU_DR=$DRV HNU_START_FRAC=0 CUDA_VISIBLE_DEVICES=$G $PY eval_student.py --student $O/$P/student_2000.pt \
      --latent-dir $LD --per-clip 32 $X --out $H/runs/results_${V}_s${s}_dr$DRV --tag _$P >> $N/${V}_F_s${s}_$P.log 2>&1
  done; done
  log "$P s$s done"
}
JOBS=(); for s in 1 2 3; do for P in Z0 Z1 Z2 Z3 Z4; do JOBS+=("$P $s"); done; done
Q=$N/${V}_queue; echo 0 > $Q.next; : > $Q.lock
worker() {
  while true; do
    while [ "$(free_mb $1)" -lt 17000 ]; do sleep 60; done
    local i=$(flock $Q.lock bash -c "i=\$(cat $Q.next); echo \$((i + 1)) > $Q.next; echo \$i")
    [ $i -ge ${#JOBS[@]} ] && return
    one ${JOBS[$i]%% *} $1 ${JOBS[$i]##* }
  done
}
worker 0 & worker 1 & wait
for j in "${JOBS[@]}"; do g=1; [ "$(free_mb 0)" -ge 17000 ] && g=0; one ${j%% *} $g ${j##* }; done
n=$(ls $H/runs/students_${V}_d2_s*/*/student_2000.pt 2>/dev/null | wc -l)
log "students done: $n of 15"
[ "$n" = 15 ] && log "finished"
