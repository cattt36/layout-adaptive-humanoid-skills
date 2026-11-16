#!/bin/bash
# v6g10k: FINAL student redistillation, supersedes v6g10j (stopped, students discarded).  Same frozen
# teachers, same trial-level protocol (train 53 / val 9 / test 17 trials), trial-split latents z_tables_v5t.
# Training only (identical for every level and baseline):
#   * TEMPO augmentation: ThreeStepPoleRun / FiveStepPoleRun train trials also replayed at 0.90 / 0.95 / 1.05 / 1.10x
#     (tools/make_timescale.py; z of the copies re-encoded by the trained latent encoder, export_z_tables_ts.py);
#     the copies share 50 % of their action's sampling mass (HNU_TS_SHARE), the balance between actions is unchanged.
#     Motivation: unseen trials fail where their tempo is outside the training range (val analysis, tools/w2_fail.sh,
#     runs/ts_check); the teacher bank stays >= 94 % on ThreeStepPoleRun / FiveStepPoleRun within 0.9-1.15x.  CrouchedTraverse not augmented
#     (teacher 79-83 % at 0.9 / 1.1x).
#   * stake layout randomisation +-8 cm (HNU_STAKE_JITTER), CrouchedTraverse sampled 1.5x, z augmentation (as v6g10j)
# Students: L0-L4 x 5 RL seeds, No-latent x 3, explicit reference x 3 (31).  DAgger 4000 + 2000.
# Evaluation unchanged: ORIGINAL clips / z tables / stake layouts, 32 trials per clip, nominal / randomised x clean /
# push; PRIMARY task criterion -> results_${V}_*, tracking criterion -> results_${V}trk_*.
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
N=$H/runs/night
V=v6g10k
cd $H/code
export HF_HUB_OFFLINE=1 HNU_BRIDGE_ROOT=$H/data/ref_bridge_video_lift HNU_STAKE_H=0.10 HNU_GEOM_W1=$H/data/ref_stakes_s93_h10 \
  HNU_GEOM_W2=$H/data/ref_stakes_s83_h10 HNU_STAKE_TOL=0.17 HNU_START_FRAC=0.25 HNU_DR_LEVEL=1 HNU_DR_LAG_SYNC=1 \
  HNU_DR_LAG_SENSORS=1 PYTHONPATH=$H/code HNU_SPLIT_JSON=$H/data/clip_split_trial/split.json
ALLOW=$H/data/clip_split_trial/clips_train_aug.txt
AUG="HNU_Z_NOISE=0.3 HNU_Z_JITTER=3 HNU_STAKE_JITTER=0.08 HNU_TASK_WEIGHT=CrouchedTraverse:1.5 HNU_TS_SHARE=0.5 HNU_GEOM_W1=$H/data/aug_ts/ref_stakes_s93_h10 HNU_GEOM_W2=$H/data/aug_ts/ref_stakes_s83_h10"
log() { echo "[$(date +%H:%M:%S)] $*" >> $N/$V.log; }
CK=$(cat $N/v6g10t_teachers.txt)
echo "$CK" > $N/${V}_teachers.txt
log "start: training clips $(wc -l < $ALLOW) (53 trials + tempo copies), val $(wc -l < $H/data/clip_split_trial/clips_val.txt), test $(wc -l < $H/data/clip_split_trial/clips_test.txt); training aug: $AUG"
# teacher bank: identical teachers and evaluation as v6g10t (re-evaluated with the corrected fall thresholds) -> reuse
for CR in task track; do
  R=results_${V}; T0=results_v6g10t; [ $CR = track ] && { R=results_${V}trk; T0=results_v6g10ttrk; }
  for DRV in 0 1; do
    [ -d $H/runs/${R}_bank_teacher_dr$DRV ] || cp -r $H/runs/${T0}_bank_teacher_dr$DRV $H/runs/${R}_bank_teacher_dr$DRV
  done
done
log "teacher bank results copied from v6g10t"
ZO="--z-offsets 0,5,10,15,20,25,30,40,50"
BASE="--scene geom --hidden 1024,512,256 --hist 20 --lin-vel"
free_mb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $1 | tr -d " "; }
one() {  # P gpu seed  (P = Z0..Z4, NL, REF)
  local P=$1 G=$2 s=$3 S=$H/runs/students_${V}_s$3 O=$H/runs/students_${V}_d2_s$3 LD=$H/data/z_tables_v5t/$1 LT=$H/data/z_tables_v5t_aug/$1 EXTRA=""
  [ $P = NL ] && { LD=$H/data/z_tables_v5t/Z0; LT=$H/data/z_tables_v5t_aug/Z0; EXTRA="--obs-group student_nolatent"; }
  [ $P = REF ] && { LD=$H/data/z_tables_v5t/Z0; LT=$H/data/z_tables_v5t_aug/Z0; EXTRA="--obs-group student_ref"; }
  [ -f $S/$P/student_4000.pt ] || env $AUG HNU_CLIPS_ALLOW=$ALLOW HNU_DR=1 HNU_TB_ROOT=$H/runs/logs/rsl_rl/students_${V}_s$s \
    CUDA_VISIBLE_DEVICES=$G $PY distill_train.py --profile ${V}_$P $CK --num-envs 4096 --iters 4000 --seed $((1000 * s)) \
    $BASE $ZO $EXTRA --latent-dir $LT --out $S/$P > $N/${V}_s${s}_E_$P.log 2>&1
  [ -f $S/$P/student_4000.pt ] || { log "$P s$s round 1 failed"; return; }
  [ -f $O/$P/student_2000.pt ] || env $AUG HNU_CLIPS_ALLOW=$ALLOW HNU_DR=1 HNU_TB_ROOT=$H/runs/logs/rsl_rl/students_${V}_d2_s$s \
    CUDA_VISIBLE_DEVICES=$G $PY distill_train.py --profile ${V}d2_$P $CK --num-envs 4096 --iters 2000 --beta-iters 1 --lr 2e-4 \
    --action-noise 0.1 --fail-weight 2 --seed $((2000 * s)) $BASE $ZO $EXTRA --init-from $S/$P/student_4000.pt \
    --latent-dir $LT --out $O/$P > $N/${V}_d2_s${s}_$P.log 2>&1
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
# order: complete 3-seed comparison (L0-L4 + baselines) first, then seeds 4 and 5 of L0-L4
JOBS=(); for s in 1 2 3; do for P in Z0 Z4 Z1 Z2 Z3; do JOBS+=("$P $s"); done; done
for s in 1 2 3; do JOBS+=("REF $s" "NL $s"); done
for s in 4 5; do for P in Z0 Z4 Z1 Z2 Z3; do JOBS+=("$P $s"); done; done
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
for j in "${JOBS[@]}"; do g=1; [ "$(free_mb 0)" -ge 17000 ] && g=0; one ${j%% *} $g ${j##* }; done  # retry failures
for T in CrouchedTraverse ThreeStepPoleRun FiveStepPoleRun; do for DRV in 0 1; do
  HNU_TASK=$T $PY tools/exec_quality_table.py $H/runs/results_${V}_bank_teacher_dr$DRV $H/runs/results_${V}_s1_dr$DRV \
    $H/runs/results_${V}_s2_dr$DRV $H/runs/results_${V}_s3_dr$DRV $H/runs/results_${V}_s4_dr$DRV \
    $H/runs/results_${V}_s5_dr$DRV > $N/${V}_table_${T}_dr$DRV.txt 2>&1
done; done
n=$(ls $H/runs/students_${V}_d2_s*/*/student_2000.pt 2>/dev/null | wc -l)
log "students done: $n of 31"
$PY tools/select_seed.py $H/runs/results_${V} > $N/${V}_selection.txt 2>&1
[ "$n" = 31 ] && log "finished"
