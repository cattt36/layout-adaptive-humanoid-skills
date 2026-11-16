#!/bin/bash
# v6g10t: TRIAL-LEVEL held-out protocol.  Cross-subject transfer failed for every student type
# (0 % on test subjects, v6g10h / pilot v6g10p), so: per action and subject, trials split train / val / test
# (data/clip_split_trial, seed 2026); students train on the train trials only, deployment seeds are selected on
# val trials, every reported closed-loop number comes from test trials (never seen in training).
# z tables from the trial-split latents (latent_v5t: the test trials are never seen by the latent either).
# Students: L0-L4 x 3 RL seeds, No-latent x 3, explicit reference x 3.  z augmentation for all z students
# (uniform noise +-0.3 on every z sample, random time shift +-3 steps; training only).  DAgger 4000 + 2000.
# Evaluation: 32 trials per clip, nominal / randomised x clean / push, split labels from the trial split.
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
N=$H/runs/night
V=v6g10t
cd $H/code
export HF_HUB_OFFLINE=1 HNU_BRIDGE_ROOT=$H/data/ref_bridge_video_lift HNU_STAKE_H=0.10 HNU_GEOM_W1=$H/data/ref_stakes_s93_h10 \
  HNU_GEOM_W2=$H/data/ref_stakes_s83_h10 HNU_STAKE_TOL=0.17 HNU_START_FRAC=0.25 HNU_DR_LEVEL=1 HNU_DR_LAG_SYNC=1 \
  HNU_DR_LAG_SENSORS=1 PYTHONPATH=$H/code HNU_SPLIT_JSON=$H/data/clip_split_trial/split.json
ALLOW=$H/data/clip_split_trial/clips_train.txt
AUG="HNU_Z_NOISE=0.3 HNU_Z_JITTER=3"
log() { echo "[$(date +%H:%M:%S)] $*" >> $N/$V.log; }
CK=$(cat $N/v6g10h_teachers.txt)
echo "$CK" > $N/${V}_teachers.txt
log "waiting for the trial-split latents / z tables (run_latent_t.sh)"
for i in $(seq 1 3000); do grep -q "z tables exported" $N/latent_v5t.log 2>/dev/null && break; sleep 60; done
log "start: training trials $(wc -l < $ALLOW), val $(wc -l < $H/data/clip_split_trial/clips_val.txt), test $(wc -l < $H/data/clip_split_trial/clips_test.txt)"
# success criteria: PRIMARY = task (no fall, motion completed; missed / edge stakes, recovered stumbles and
# bridge contact are not failures) -> results_${V}_*; SECONDARY = reference-tracking terminations -> results_${V}trk_*
for CR in task track; do
  R=$H/runs/results_${V}; [ $CR = track ] && R=$H/runs/results_${V}trk
  for DRV in 0 1; do for X in "" "--push"; do  # teacher bank (upper bound: the teachers were trained on all clips)
    [ -f ${R}_bank_teacher_dr$DRV/eval_teacher_teacher$([ -n "$X" ] && echo _push).json ] && continue
    HNU_EVAL_CRIT=$CR HNU_DR=$DRV HNU_START_FRAC=0 CUDA_VISIBLE_DEVICES=0 $PY eval_student.py --teacher $CK --scene geom \
      --z-offsets 0,5,10,15,20,25,30,40,50 --latent-dir $H/data/z_tables_v5/Z4 --per-clip 32 $X --out ${R}_bank_teacher_dr$DRV \
      --tag _teacher >> $N/${V}_teacher.log 2>&1
  done; done
done
log "teacher bank evaluated"
ZO="--z-offsets 0,5,10,15,20,25,30,40,50"
BASE="--scene geom --hidden 1024,512,256 --hist 20 --lin-vel"
free_mb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $1 | tr -d " "; }
one() {  # P gpu seed  (P = Z0..Z4, NL, REF)
  local P=$1 G=$2 s=$3 S=$H/runs/students_${V}_s$3 O=$H/runs/students_${V}_d2_s$3 LD=$H/data/z_tables_v5t/$1 EXTRA=""
  [ $P = NL ] && { LD=$H/data/z_tables_v5t/Z0; EXTRA="--obs-group student_nolatent"; }
  [ $P = REF ] && { LD=$H/data/z_tables_v5t/Z0; EXTRA="--obs-group student_ref"; }
  [ -f $S/$P/student_4000.pt ] || env $AUG HNU_CLIPS_ALLOW=$ALLOW HNU_DR=1 HNU_TB_ROOT=$H/runs/logs/rsl_rl/students_${V}_s$s \
    CUDA_VISIBLE_DEVICES=$G $PY distill_train.py --profile ${V}_$P $CK --num-envs 4096 --iters 4000 --seed $((1000 * s)) \
    $BASE $ZO $EXTRA --latent-dir $LD --out $S/$P > $N/${V}_s${s}_E_$P.log 2>&1
  [ -f $S/$P/student_4000.pt ] || { log "$P s$s round 1 failed"; return; }
  [ -f $O/$P/student_2000.pt ] || env $AUG HNU_CLIPS_ALLOW=$ALLOW HNU_DR=1 HNU_TB_ROOT=$H/runs/logs/rsl_rl/students_${V}_d2_s$s \
    CUDA_VISIBLE_DEVICES=$G $PY distill_train.py --profile ${V}d2_$P $CK --num-envs 4096 --iters 2000 --beta-iters 1 --lr 2e-4 \
    --action-noise 0.1 --fail-weight 2 --seed $((2000 * s)) $BASE $ZO $EXTRA --init-from $S/$P/student_4000.pt \
    --latent-dir $LD --out $O/$P > $N/${V}_d2_s${s}_$P.log 2>&1
  [ -f $O/$P/student_2000.pt ] || { log "$P s$s round 2 failed"; return; }
  for CR in task track; do
    local R=$H/runs/results_${V}; [ $CR = track ] && R=$H/runs/results_${V}trk
    for DRV in 0 1; do for X in "" "--push"; do
      HNU_EVAL_CRIT=$CR HNU_DR=$DRV HNU_START_FRAC=0 CUDA_VISIBLE_DEVICES=$G $PY eval_student.py --student $O/$P/student_2000.pt \
        --latent-dir $LD --per-clip 32 $X --out ${R}_s${s}_dr$DRV --tag _$P >> $N/${V}_F_s${s}_$P.log 2>&1
    done; done
  done
  log "$P s$s done"
}
JOBS=(); for s in 1 2 3; do for P in Z0 Z4 Z1 Z2 Z3; do JOBS+=("$P $s"); done; done
for s in 1 2 3; do JOBS+=("REF $s" "NL $s"); done
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
for T in CrouchedTraverse ThreeStepPoleRun FiveStepPoleRun; do for DRV in 0 1; do
  HNU_TASK=$T $PY tools/exec_quality_table.py $H/runs/results_${V}_bank_teacher_dr$DRV $H/runs/results_${V}_s1_dr$DRV \
    $H/runs/results_${V}_s2_dr$DRV $H/runs/results_${V}_s3_dr$DRV > $N/${V}_table_${T}_dr$DRV.txt 2>&1
done; done
n=$(ls $H/runs/students_${V}_d2_s*/*/student_2000.pt 2>/dev/null | wc -l)
log "students done: $n of 21"
$PY tools/select_seed.py $H/runs/results_${V} > $N/${V}_selection.txt 2>&1
[ "$n" = 21 ] && log "finished"
