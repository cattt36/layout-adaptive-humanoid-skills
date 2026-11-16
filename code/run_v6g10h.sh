#!/bin/bash
# v6g10h: HELD-OUT protocol for every closed-loop result.  Identical to v6g10s (teachers, unified geometry scene,
# 0.10 m stakes, sensor-only 0-20 ms latency, DAgger 4000 + round 2 2000, evaluation) except that the students are
# TRAINED ONLY ON THE TRAIN-SUBJECT CLIPS of the latent subject split (data/clip_split_v5/clips_train.txt; HNU_CLIPS_ALLOW).
# Validation-subject clips: deployment-seed selection (select_seed.py).  Test-subject clips: all reported closed-loop
# precision / success.  Students: L0-L4 x 3 RL seeds + "No latent" (task one-hot + clip progress) x 3 RL seeds.
# Teachers are not retrained: they only label states of training clips, so test clips never reach a student
# (the teacher row is an optimistic upper bound: the teachers were trained on all clips).
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
N=$H/runs/night
V=v6g10h
cd $H/code
export HF_HUB_OFFLINE=1 HNU_BRIDGE_ROOT=$H/data/ref_bridge_video_lift HNU_STAKE_H=0.10 HNU_GEOM_W1=$H/data/ref_stakes_s93_h10 \
  HNU_GEOM_W2=$H/data/ref_stakes_s83_h10 HNU_STAKE_TOL=0.17 HNU_START_FRAC=0.25 HNU_DR_LEVEL=1 HNU_DR_LAG_SYNC=1 \
  HNU_DR_LAG_SENSORS=1 PYTHONPATH=$H/code
ALLOW=$H/data/clip_split_v5/clips_train.txt
log() { echo "[$(date +%H:%M:%S)] $*" >> $N/$V.log; }
log "start (restart with free-memory scheduler)"
CK=$(cat $N/v6g10s_teachers.txt)
echo "$CK" > $N/${V}_teachers.txt
for DRV in 0 1; do  # same teachers, same evaluation as v6g10s
  [ -e $H/runs/results_${V}_bank_teacher_dr$DRV ] || ln -s $H/runs/results_v6g10s_bank_teacher_dr$DRV $H/runs/results_${V}_bank_teacher_dr$DRV
done
log "students: training clips $(wc -l < $ALLOW) (train subjects); teachers: $CK"
ZO="--z-offsets 0,5,10,15,20,25,30,40,50"
BASE="--scene geom --hidden 1024,512,256 --hist 20 --lin-vel"
one() {  # P gpu seed   (P = Z0..Z4 or NL)
  local P=$1 G=$2 s=$3 S=$H/runs/students_${V}_s$3 O=$H/runs/students_${V}_d2_s$3 LD=$H/data/z_tables_v5/$1 EXTRA=""
  if [ $P = NL ]; then LD=$H/data/z_tables_v5/Z0; EXTRA="--obs-group student_nolatent"; fi
  [ -f $S/$P/student_4000.pt ] || HNU_CLIPS_ALLOW=$ALLOW HNU_DR=1 HNU_TB_ROOT=$H/runs/logs/rsl_rl/students_${V}_s$s \
    CUDA_VISIBLE_DEVICES=$G $PY distill_train.py --profile ${V}_$P $CK --num-envs 4096 --iters 4000 --seed $((1000 * s)) \
    $BASE $ZO $EXTRA --latent-dir $LD --out $S/$P > $N/${V}_s${s}_E_$P.log 2>&1
  [ -f $S/$P/student_4000.pt ] || { log "$P s$s round 1 failed"; return; }
  [ -f $O/$P/student_2000.pt ] || HNU_CLIPS_ALLOW=$ALLOW HNU_DR=1 HNU_TB_ROOT=$H/runs/logs/rsl_rl/students_${V}_d2_s$s \
    CUDA_VISIBLE_DEVICES=$G $PY distill_train.py --profile ${V}d2_$P $CK --num-envs 4096 --iters 2000 --beta-iters 1 --lr 2e-4 \
    --action-noise 0.1 --fail-weight 2 --seed $((2000 * s)) $BASE $ZO $EXTRA --init-from $S/$P/student_4000.pt \
    --latent-dir $LD --out $O/$P > $N/${V}_d2_s${s}_$P.log 2>&1
  [ -f $O/$P/student_2000.pt ] || { log "$P s$s round 2 failed"; return; }
  for DRV in 0 1; do for X in "" "--push"; do  # evaluation on ALL clips (split labels kept); test subjects reported
    HNU_DR=$DRV HNU_START_FRAC=0 CUDA_VISIBLE_DEVICES=$G $PY eval_student.py --student $O/$P/student_2000.pt \
      --latent-dir $LD --per-clip 8 $X --out $H/runs/results_${V}_s${s}_dr$DRV --tag _$P >> $N/${V}_F_s${s}_$P.log 2>&1
  done; done
  log "$P s$s done"
}
JOBS=(); for s in 1 2 3; do for P in Z0 Z1 Z2 Z3 Z4; do JOBS+=("$P $s"); done; done
for s in 1 2 3; do JOBS+=("NL $s"); done
# dynamic scheduler: a GPU takes the next job only when it has >= 17 GB free (GPU 0 may be occupied by a foreign
# process); the job index is shared through a counter file under flock
Q=$N/${V}_queue; echo 0 > $Q.next; : > $Q.lock
free_mb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $1 | tr -d " "; }
worker() {  # gpu
  while true; do
    while [ "$(free_mb $1)" -lt 17000 ]; do sleep 60; done
    local i=$(flock $Q.lock bash -c "i=\$(cat $Q.next); echo \$((i + 1)) > $Q.next; echo \$i")
    [ $i -ge ${#JOBS[@]} ] && return
    one ${JOBS[$i]%% *} $1 ${JOBS[$i]##* }
  done
}
worker 0 & worker 1 & wait
for j in "${JOBS[@]}"; do  # retry anything that failed, on whichever GPU has room
  g=1; [ "$(free_mb 0)" -ge 17000 ] && g=0
  while [ "$(free_mb $g)" -lt 17000 ]; do sleep 60; done
  one ${j%% *} $g ${j##* }
done
for T in CrouchedTraverse ThreeStepPoleRun FiveStepPoleRun; do
  for DRV in 0 1; do
    HNU_TASK=$T $PY tools/exec_quality_table.py $H/runs/results_${V}_bank_teacher_dr$DRV $H/runs/results_${V}_s1_dr$DRV \
      $H/runs/results_${V}_s2_dr$DRV $H/runs/results_${V}_s3_dr$DRV > $N/${V}_table_${T}_dr$DRV.txt 2>&1
  done
done
n=$(ls $H/runs/students_${V}_d2_s*/*/student_2000.pt 2>/dev/null | wc -l)
log "students done: $n of 18"
$PY tools/select_seed.py $H/runs/results_${V} > $N/${V}_selection.txt 2>&1
[ "$n" = 18 ] && log "finished"
