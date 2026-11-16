#!/bin/bash
# baseline with the support-post bases of the real bridge (HNU_POST_BASE=1: floor bar 100 x 10 x 3 cm +
# 2 braces per post, hnu_tasks/bridge.py).  Real-robot video: the deployed L3 tripped over a bar.
#  GPU 0: deployed student (v6g10xm fold 1 seed 5 L3), condition B (library z + plan4), DR 0 / 1, with bases
#  GPU 1: teacher bank v6g10n, own reference, with bases (+ floor-touch metric for the stake tasks)
#  GPU 2: export of the deployed student with the bases in scene.mjb -> deploy_exports/v6g10xm_f1_s5_base/L3
#  then CPU sim2sim (deploy venv) on the planned bridge layouts D1 / D2 / N3, conditions A / B / C as 
# Out: runs/base_baseline/, robot_sim_check/out_base/; log runs/night/base_baseline.log
PY=/root/gpufree-data/venvs/mjlab/bin/python
PYD=/root/gpufree-data/venvs/deploy/bin/python
H=/root/gpufree-data/hnu_v2; D=$H/data; N=$H/runs/night; O=$H/runs/base_baseline
RS=/root/gpufree-data/robot_sim_check
cd $H/code
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HNU_BRIDGE_ROOT=$D/ref_bridge_video_lift HNU_STAKE_H=0.10 HNU_STAKE_TOL=0.17 \
  HNU_DR_LEVEL=1 HNU_DR_LAG_SYNC=1 HNU_DR_LAG_SENSORS=1 PYTHONPATH=$H/code HNU_STAKE_SHAPE=mesh16 HNU_POST_SHAPE=mesh12 \
  HNU_START_FRAC=0 HNU_EVAL_CRIT=task HNU_POST_BASE=1 HNU_SPLIT_JSON=$D/clip_split_cv/fold1/split.json
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $N/base_baseline.log; }
mkdir -p $O
ST=$H/runs/students_v6g10xm_d2_s5/Z3/student_2000.pt
GB="HNU_GEOM_W1=$D/geom_plan4_f1/ref_stakes_s93_h10 HNU_GEOM_W2=$D/geom_plan4_f1/ref_stakes_s83_h10"
GA="HNU_GEOM_W1=$D/ref_stakes_s93_h10 HNU_GEOM_W2=$D/ref_stakes_s83_h10"
log "start"
(
  for DRV in 0 1; do
    [ -f $O/student_B_dr$DRV/eval_student_Z3.json ] && continue
    env $GB HNU_DR=$DRV CUDA_VISIBLE_DEVICES=0 $PY eval_student.py --student $ST --latent-dir $D/z_lib_f1/plan4/Z3 \
      --per-clip 32 --out $O/student_B_dr$DRV --tag _Z3 >> $O/student.log 2>&1
  done
  log "student done"
) &
(
  [ -f $O/teacher_dr0/eval_teacher_teacher.json ] || env $GA HNU_DR=0 HNU_EVAL_FLOOR=1 CUDA_VISIBLE_DEVICES=1 $PY eval_student.py \
    --teacher $(cat $N/v6g10n_teachers.txt) --scene geom --z-offsets 0,5,10,15,20,25,30,40,50 \
    --latent-dir $D/z_tables_v5t/Z4 --per-clip 32 --out $O/teacher_dr0 --tag _teacher >> $O/teacher.log 2>&1
  log "teacher done"
) &
(
  E=$H/deploy_exports/v6g10xm_f1_s5_base/L3; mkdir -p $E
  [ -f $E/policy.onnx ] || env $GA CUDA_VISIBLE_DEVICES=2 $PY deploy/export_student.py --student $ST \
    --latent-dir $D/z_tables_cv_f1/Z3 --out $E > $E/../L3.export.log 2>&1
  [ -f $E/parity.txt ] || $PYD deploy/parity_check.py $E > $E/parity.txt 2>&1
  log "export: $([ -f $E/policy.onnx ] && echo ok || echo FAILED); parity: $(tail -n 1 $E/parity.txt | cut -c1-100)"
  mkdir -p $RS/out_base
  for L in $RS/planned_layouts/D*_CrouchedTraverse_* $RS/planned_layouts/N3_*; do
    n=$(basename $L)
    for c in A B C; do
      case $c in
        A) X="";;
        B) X="--dr";;
        C) X="--dr --perc-noise 0.02 --obst-shift 0.03 --stake-jitter 0.03 --start-yaw 5 --init-off 0.02";;
      esac
      [ -f $RS/out_base/${n}_$c.json ] && continue
      $PYD deploy/sim2sim.py $E --latency 0-20 --crit task --trials 100 --workers 64 --tasks CrouchedTraverse \
        --motion-dirs $L --latent-dir $L $X --out $RS/out_base/${n}_$c.json > $RS/out_base/${n}_$c.log 2>&1
      log "sim2sim $n $c: $(tail -1 $RS/out_base/${n}_$c.log)"
    done
  done
) &
wait
log "finished"
