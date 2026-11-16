#!/bin/bash
# 8.3.2 for one new student pair (L0 / L3) of fold FOLD, RL seed SEED -- parametrised copy of tools/deploy_v2_4090.sh
# (user decision: L0 / L3 x 4 folds x 3 seeds = 24 students, like 8.2-8.5).  Same settings:
#   sim2sim (CPU MuJoCo, 0-20 ms, criterion v2, 8 trials): base, fixed 20 ms, perception bias / noise, layout shift,
#   independent stake offsets, beam height, start heading, odometry drift; new conservative items (re-planned):
#   stake spacing x0.90-1.10, start -> first stake +-5/10 cm, bridge post gap +-3/5 cm
#   experiment C (mjlab, GPU): obstacle features frozen 0 / 0.25 / 0.5 / 1.0 s
# Chain = library z + planner plan4 of the fold.  Exports deploy_exports/v6g10xm_f<F>_s<S>/L0, L3.
#   FOLD=2 SEED=1 bash tools/deploy_v2_run.sh        log runs/night/deploy_v2.log
F=${FOLD:-1} S=${SEED:-1}
FS=""; [ $F != 1 ] && FS=_f$F
EXP=v6g10xm_f${F}_s$S
W=${W:-96}
PY=/root/gpufree-data/venvs/mjlab/bin/python
PYD=/root/gpufree-data/venvs/deploy/bin/python
H=/root/gpufree-data/hnu_v2
N=$H/runs/night
D=$H/data
cd $H/code
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HNU_BRIDGE_ROOT=$D/ref_bridge_video_lift HNU_STAKE_H=0.10 \
  HNU_GEOM_W1=$D/ref_stakes_s93_h10 HNU_GEOM_W2=$D/ref_stakes_s83_h10 HNU_STAKE_TOL=0.17 HNU_START_FRAC=0.25 \
  HNU_DR_LEVEL=1 HNU_DR_LAG_SYNC=1 HNU_DR_LAG_SENSORS=1 PYTHONPATH=$H/code HNU_STAKE_SHAPE=mesh16 HNU_POST_SHAPE=mesh12
log() { echo "[$(date '+%m-%d %H:%M:%S')] $EXP: $*" >> $N/deploy_v2.log; }
log "start"
# ---- data of the fold: library, planner tables, perturbed layouts
[ -d $D/z_lib_f$F/geo/Z3 ] || HNU_LEVELS=Z0,Z1,Z2,Z3 $PY tools/make_z_library.py $F >> $N/deploy_v2.log 2>&1
[ -d $D/z_lib_f$F/plan4/Z3 ] && [ -d $D/geom_plan4_f$F ] || $PY tools/make_z_plan.py $F plan4 >> $N/deploy_v2.log 2>&1
[ -d $D/z_pert_f$F/post+0.05/Z3 ] || $PY tools/make_layout_perturb.py $F >> $N/deploy_v2.log 2>&1 || { log "perturb FAILED"; exit 1; }
GPD="$D/geom_plan4_f$F/ref_stakes_s93_h10/ThreeStepPoleRun,$D/geom_plan4_f$F/ref_stakes_s83_h10/FiveStepPoleRun,$D/ref_bridge_video_lift/CrouchedTraverse"
pert_dirs() { local G=$D/geom_pert_f$F/$1; echo "$G/ref_stakes_s93_h10/ThreeStepPoleRun,$G/ref_stakes_s83_h10/FiveStepPoleRun,$G/ref_bridge_video_lift/CrouchedTraverse"; }
ST=$H/runs/students_v6g10xm${FS}_d2_s$S
# ---- exports
OUT=$H/deploy_exports/$EXP; mkdir -p $OUT
for P in Z0 Z3; do
  E=$OUT/L${P#Z}
  [ -f $E/policy.onnx ] || CUDA_VISIBLE_DEVICES=0 $PY deploy/export_student.py --student $ST/$P/student_2000.pt \
    --latent-dir $D/z_tables_cv_f$F/$P --out $E > $OUT/L${P#Z}.export.log 2>&1
  [ -f $E/parity.txt ] || $PYD deploy/parity_check.py $E > $E/parity.txt 2>&1
  log "export L${P#Z}: $([ -f $E/policy.onnx ] && echo ok || echo FAILED); parity: $(tail -n 1 $E/parity.txt | cut -c1-100)"
done
# ---- experiment C (GPU, background)
(
  [ -f $H/runs/freeze_table.json ] || $PY tools/make_freeze_table.py >> $N/deploy_v2.log 2>&1
  for P in Z0 Z3; do for d in 0 0.25 0.5 1.0; do
    O=$H/runs/results_v6g10xmC${d}${FS}_s${S}_dr0
    [ -f $O/eval_student_$P.json ] && continue
    HNU_GEOM_W1=$D/geom_plan4_f$F/ref_stakes_s93_h10 HNU_GEOM_W2=$D/geom_plan4_f$F/ref_stakes_s83_h10 \
      HNU_SPLIT_JSON=$D/clip_split_cv/fold$F/split.json HNU_PERC_FREEZE=$d HNU_PERC_FREEZE_TABLE=$H/runs/freeze_table.json \
      HNU_EVAL_CRIT=task HNU_DR=0 HNU_START_FRAC=0 CUDA_VISIBLE_DEVICES=1 nice $PY eval_student.py \
      --student $ST/$P/student_2000.pt --latent-dir $D/z_lib_f$F/plan4/$P --per-clip 32 --out $O --tag _$P >> $N/deploy_v2_eval.log 2>&1
  done; done
  log "experiment C done"
) &
# ---- sim2sim sweeps
run() {  # export_dir level name args...
  local EX=$1 P=$2 NAME=$3; shift 3
  mkdir -p $EX/sweeps
  [ -f $EX/sweeps/$NAME.json ] && return
  local MD=$GPD LD=$D/z_lib_f$F/plan4/$P LAT="0-20"
  case $NAME in spacing*|start-*|start+*|post*) MD=$(pert_dirs $NAME); LD=$D/z_pert_f$F/$NAME/$P;; esac
  [ $NAME = lat20 ] && LAT=20
  $PYD deploy/sim2sim.py $EX --latency $LAT --crit task --trials 8 --workers $W --motion-dirs $MD --latent-dir $LD "$@" \
    --out $EX/sweeps/$NAME.json >> $EX/sweeps/sweeps.log 2>&1
  [ -f $EX/sweeps/$NAME.json ] || log "$(basename $EX) $NAME FAILED"
}
for L in L3 L0; do
  EX=$OUT/$L P=Z${L#L}
  run $EX $P base
  run $EX $P lat20
  for s in 0.90 0.95 1.05 1.10; do run $EX $P spacing$s --tasks ThreeStepPoleRun,FiveStepPoleRun; done
  for d in -0.10 -0.05 +0.05 +0.10; do run $EX $P start$d --tasks ThreeStepPoleRun,FiveStepPoleRun; done
  for g in -0.05 -0.03 +0.03 +0.05; do run $EX $P post$g --tasks CrouchedTraverse; done
  for b in 0.02 0.05 0.08 0.12; do run $EX $P perc_bias$b --perc-bias $b; done
  for n in 0.02 0.05 0.10; do run $EX $P perc_noise$n --perc-noise $n; done
  for b in 0.02 0.05 0.08 0.12; do run $EX $P shift$b --obst-shift $b; done
  for j in 0.02 0.05 0.08; do run $EX $P stake_jitter$j --tasks ThreeStepPoleRun,FiveStepPoleRun --stake-jitter $j; done
  for dz in -0.05 -0.03 0.03 0.05; do run $EX $P bridge_dz$dz --tasks CrouchedTraverse --bridge-dz $dz; done
  for y in 3 5 10; do run $EX $P yaw$y --start-yaw $y; done
  for r in 0.01 0.02 0.05 0.10; do run $EX $P drift$r --drift $r; done
  log "$L sweeps done ($(ls $EX/sweeps/*.json | wc -l) settings)"
done
wait
log "all done"
