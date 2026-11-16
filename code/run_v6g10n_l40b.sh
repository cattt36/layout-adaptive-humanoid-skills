#!/bin/bash
# v6g10n on the rented L40 server: EVERY student of the study -- fold 1 (= split_v2) incl. baselines and
# seeds 4-5, and the CV folds 2-4 (runs/preregistration_cv.md) -- is trained and evaluated here, on the same hardware.
# Recipe identical to run_v6g10n.sh (teachers, augmentation, DAgger 4000 + 2000, evaluation); per fold only the data
# differ: training trials + tempo copies (clip_split_cv/fold<k>/clips_train_aug.txt, aug_ts_f<k>), the z tables of the
# fold's own encoders (z_tables_cv_f<k>; encoders and z tables made on the 4090 server) and the fold's split labels.
# Names: fold 1 keeps the v6g10n names (students_v6g10n_s<s>, results_v6g10n_s<s>_dr*), folds 2-4 use v6g10n_f<k>.
# Order: teacher bank (re-evaluated on L40) -> PRIMARY (folds 1-4 x L0-L4 x seeds 1-3, seed-major, L0 / L4 first) ->
# fold-1 baselines (REF / NL x 3) -> fold-1 seeds 4-5.  A job whose fold data are not ready yet is skipped and picked up
# later (claim = atomic mkdir), so no GPU waits for the data.  Slots per GPU: HNU_SLOTS (default 1).
# run_v6g10n_l40b.sh (~21:00, user decision): same recipe; started at the batch boundary after fold-1 seed 2
# L0-L4, with the diagnostic jobs FIRST: fold-1 REF / NL seed 1 and the perception-channel ablation "NT" (L0 / L4,
# seeds 1-2: identical to the main students except that the time-to-step entries of the student's stake features are
# zeroed, HNU_STAKE_NO_TTS=1, in training AND evaluation; names v6g10nnt).  Then the remaining primary jobs, REF / NL
# seeds 2-3, seeds 4-5.  Finished students / evaluations are skipped by one(), nothing is redone.
# (user decision): L4 (Z4) dropped from the study -- no further Z4 job; NT compares L0 with L3.
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
N=$H/runs/night
V=v6g10n
NGPU=$(nvidia-smi -L | wc -l)
SLOTS=${HNU_SLOTS:-1}
mkdir -p $N
cd $H/code
export HF_HUB_OFFLINE=1 HNU_BRIDGE_ROOT=$H/data/ref_bridge_video_lift HNU_STAKE_H=0.10 HNU_GEOM_W1=$H/data/ref_stakes_s93_h10 \
  HNU_GEOM_W2=$H/data/ref_stakes_s83_h10 HNU_STAKE_TOL=0.17 HNU_START_FRAC=0.25 HNU_DR_LEVEL=1 HNU_DR_LAG_SYNC=1 \
  HNU_DR_LAG_SENSORS=1 PYTHONPATH=$H/code HNU_STAKE_SHAPE=mesh16 HNU_POST_SHAPE=mesh12
aug() {  # fold -> training augmentation (identical for every fold except the geometry dirs with the fold's copies)
  echo "HNU_Z_NOISE=0.3 HNU_Z_JITTER=3 HNU_STAKE_JITTER=0 HNU_STAKE_JITTER_ALONG=0.03 HNU_STAKE_JITTER_LAT=0.06 HNU_STAKE_STRAIGHT=0.3 HNU_STAKE_CONTACT=1 HNU_BRIDGE_JITTER_XY=0.05 HNU_BRIDGE_JITTER_ZLO=-0.05 HNU_BRIDGE_JITTER_ZHI=0.06 HNU_BRIDGE_POST_JITTER=0.10 HNU_RESET_SCALE=1.5 HNU_PUSH_SCALE=1.0 HNU_TASK_WEIGHT=CrouchedTraverse:1.5 HNU_TS_SHARE=0.5 HNU_GEOM_W1=$H/data/aug_ts_f$1/ref_stakes_s93_h10 HNU_GEOM_W2=$H/data/aug_ts_f$1/ref_stakes_s83_h10 HNU_BRIDGE_ROOT=$H/data/aug_ts_f$1/ref_bridge_video_lift"
}
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $N/$V.log; }
ready() {  # fold data present and consistent (make_aug_ts consistency check passed when the data were made)
  [ -s $H/data/clip_split_cv/fold$1/clips_train_aug.txt ] && [ -f $H/data/z_tables_cv_f$1/ready ] && [ -f $H/data/z_tables_cv_f$1_aug/ready ]
}
CK=$(cat $N/${V}_teachers.txt)
log "start (L40 server, $NGPU GPUs x $SLOTS slots): teachers $CK"
# teacher bank on this hardware (same protocol as on the 4090 server)
for CR in task track; do
  R=$H/runs/results_${V}; [ $CR = track ] && R=$H/runs/results_${V}trk
  for DRV in 0 1; do for X in "" "--push"; do
    [ -f ${R}_bank_teacher_dr$DRV/eval_teacher_teacher$([ -n "$X" ] && echo _push).json ] && continue
    HNU_SPLIT_JSON=$H/data/clip_split_trial/split_v2.json HNU_EVAL_CRIT=$CR HNU_DR=$DRV HNU_START_FRAC=0 CUDA_VISIBLE_DEVICES=0 \
      $PY eval_student.py --teacher $CK --scene geom --z-offsets 0,5,10,15,20,25,30,40,50 --latent-dir $H/data/z_tables_v5t/Z4 \
      --per-clip 32 $X --out ${R}_bank_teacher_dr$DRV --tag _teacher >> $N/${V}_teacher.log 2>&1
  done; done
done
log "teacher bank evaluated"
ZO="--z-offsets 0,5,10,15,20,25,30,40,50"
BASE="--scene geom --hidden 1024,512,256 --hist 20 --lin-vel"
one() {  # fold P gpu seed  (P = Z0..Z4, NL, REF, NT0..NT4; baselines and NT only in fold 1)
  local F=$1 P=$2 G=$3 s=$4 VS=$V NTE=""
  if [ ${P:0:2} = NT ]; then P=Z${P:2}; VS=${V}nt; NTE="HNU_STAKE_NO_TTS=1"; fi  # perception-channel ablation
  [ $F != 1 ] && VS=${VS}_f$F
  local S=$H/runs/students_${VS}_s$s O=$H/runs/students_${VS}_d2_s$s LD=$H/data/z_tables_cv_f$F/$P LT=$H/data/z_tables_cv_f${F}_aug/$P
  local ALLOW=$H/data/clip_split_cv/fold$F/clips_train_aug.txt AUG=$(aug $F) EXTRA=""
  [ $P = NL ] && { LD=$H/data/z_tables_cv_f$F/Z0; LT=$H/data/z_tables_cv_f${F}_aug/Z0; EXTRA="--obs-group student_nolatent"; }
  [ $P = REF ] && { LD=$H/data/z_tables_cv_f$F/Z0; LT=$H/data/z_tables_cv_f${F}_aug/Z0; EXTRA="--obs-group student_ref"; }
  [ -f $S/$P/student_4000.pt ] || env $AUG $NTE HNU_CLIPS_ALLOW=$ALLOW HNU_DR=1 HNU_TB_ROOT=$H/runs/logs/rsl_rl/students_${VS}_s$s \
    CUDA_VISIBLE_DEVICES=$G $PY distill_train.py --profile ${VS}_$P $CK --num-envs 4096 --iters 4000 --seed $((1000 * s)) \
    $BASE $ZO $EXTRA --latent-dir $LT --out $S/$P > $N/${VS}_s${s}_E_$P.log 2>&1
  [ -f $S/$P/student_4000.pt ] || { log "fold $F $P s$s round 1 failed"; return; }
  [ -f $O/$P/student_2000.pt ] || env $AUG $NTE HNU_CLIPS_ALLOW=$ALLOW HNU_DR=1 HNU_TB_ROOT=$H/runs/logs/rsl_rl/students_${VS}_d2_s$s \
    CUDA_VISIBLE_DEVICES=$G $PY distill_train.py --profile ${VS}d2_$P $CK --num-envs 4096 --iters 2000 --beta-iters 1 --lr 2e-4 \
    --action-noise 0.1 --fail-weight 2 --seed $((2000 * s)) $BASE $ZO $EXTRA --init-from $S/$P/student_4000.pt \
    --latent-dir $LT --out $O/$P > $N/${VS}_d2_s${s}_$P.log 2>&1
  [ -f $O/$P/student_2000.pt ] || { log "fold $F $P s$s round 2 failed"; return; }
  for CR in task track; do
    local R=$H/runs/results_${VS}; [ $CR = track ] && R=$H/runs/results_${V}$([ -n "$NTE" ] && echo nt)trk$([ $F != 1 ] && echo _f$F)
    for DRV in 0 1; do for X in "" "--push"; do
      [ -f ${R}_s${s}_dr$DRV/eval_student_${P}$([ -n "$X" ] && echo _push).json ] && continue
      env $NTE HNU_SPLIT_JSON=$H/data/clip_split_cv/fold$F/split.json HNU_EVAL_CRIT=$CR HNU_DR=$DRV HNU_START_FRAC=0 \
        CUDA_VISIBLE_DEVICES=$G $PY eval_student.py --student $O/$P/student_2000.pt \
        --latent-dir $LD --per-clip 32 $X --out ${R}_s${s}_dr$DRV --tag _$P >> $N/${VS}_F_s${s}_$P.log 2>&1
    done; done
  done
  log "fold $F $P s$s done${NTE:+ (NT: no time-to-step)}"
}
JOBS=("1 REF 1" "1 NL 1" "1 NT0 1" "1 NT3 1" "1 NT0 2" "1 NT3 2")                                    # diagnostics first
for s in 1 2 3; do for F in 1 2 3 4; do for P in Z0 Z3 Z1 Z2; do JOBS+=("$F $P $s"); done; done; done  # primary (60)
for s in 2 3; do JOBS+=("1 REF $s" "1 NL $s"); done                                                    # fold-1 baselines
for s in 4 5; do for P in Z0 Z3 Z1 Z2; do JOBS+=("1 $P $s"); done; done                              # fold-1 seeds 4-5
Q=$N/${V}_l40_claims  # claims of THIS run (reset at start: finished parts are skipped by one(), nothing is redone)
rm -rf $Q; mkdir -p $Q
worker() {  # gpu
  while true; do
    local left=0 took=0
    for i in "${!JOBS[@]}"; do
      [ -d $Q/$i ] && continue
      left=1
      set -- ${JOBS[$i]}
      ready $1 || continue
      mkdir $Q/$i 2> /dev/null || continue
      one $1 $2 $G $3
      took=1
      break
    done
    [ $left = 0 ] && return
    [ $took = 0 ] && sleep 300
  done
}
for G in $(seq 0 $((NGPU - 1))); do for k in $(seq 1 $SLOTS); do
  G=$G worker & sleep 20
done; done
wait
for j in "${JOBS[@]}"; do set -- $j; one $1 $2 0 $3; done  # retry failures (sequential, GPU 0)
n=$(ls $H/runs/students_${V}_d2_s*/{Z0,Z1,Z2,Z3,REF,NL}/student_2000.pt $H/runs/students_${V}_f*_d2_s*/{Z0,Z1,Z2,Z3}/student_2000.pt \
  $H/runs/students_${V}nt_d2_s*/{Z0,Z3}/student_2000.pt 2>/dev/null | wc -l)  # L4 dropped
log "students done: $n of ${#JOBS[@]}"
for T in CrouchedTraverse ThreeStepPoleRun FiveStepPoleRun; do for DRV in 0 1; do
  HNU_TASK=$T $PY tools/exec_quality_table.py $H/runs/results_${V}_bank_teacher_dr$DRV $H/runs/results_${V}_s1_dr$DRV \
    $H/runs/results_${V}_s2_dr$DRV $H/runs/results_${V}_s3_dr$DRV $H/runs/results_${V}_s4_dr$DRV \
    $H/runs/results_${V}_s5_dr$DRV > $N/${V}_table_${T}_dr$DRV.txt 2>&1
done; done
$PY tools/select_seed.py $H/runs/results_${V} > $N/${V}_selection.txt 2>&1
[ "$n" = ${#JOBS[@]} ] && log "finished"
