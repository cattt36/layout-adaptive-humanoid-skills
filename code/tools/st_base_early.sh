#!/bin/bash
# run the S2k evaluation / export / sim2sim (and the deployed sim2sim) while S4k is still training;
# tools/st_base.sh skips outputs that already exist.  Functions copied from tools/st_base.sh.
PY=/root/gpufree-data/venvs/mjlab/bin/python
PYD=/root/gpufree-data/venvs/deploy/bin/python
H=/root/gpufree-data/hnu_v2; D=$H/data; N=$H/runs/night; R=$H/runs/st_base
RS=/root/gpufree-data/robot_sim_check
cd $H/code
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HNU_BRIDGE_ROOT=$D/ref_bridge_video_lift HNU_STAKE_H=0.10 \
  HNU_GEOM_W1=$D/ref_stakes_s93_h10 HNU_GEOM_W2=$D/ref_stakes_s83_h10 HNU_STAKE_TOL=0.17 HNU_START_FRAC=0.25 HNU_DR_LEVEL=1 \
  HNU_DR_LAG_SYNC=1 HNU_DR_LAG_SENSORS=1 PYTHONPATH=$H/code HNU_STAKE_SHAPE=mesh16 HNU_POST_SHAPE=mesh12
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $N/st_base.log; }
G=$H/runs/tf_base_gate.md

CUR=$(cat $N/v6g10n_teachers.txt)
CK=$(cat $N/v6g10n_base_teachers.txt)
SPL=$D/clip_split_cv/fold1/split.json
SRC=$H/runs/students_v6g10xm_d2_s5/Z3/student_2000.pt
ev() {  # student-pt name gpu
  local ST=$1 NM=$2 G=$3
  for B in 1 0; do for DRV in 0 1; do
    local BE=""; [ $B = 1 ] && BE="HNU_POST_BASE=1"
    local OA=$R/${NM}_A_b${B}_dr$DRV OB=$R/${NM}_B_b${B}_dr$DRV
    [ -f $OA/eval_student_Z3.json ] || env $BE HNU_SPLIT_JSON=$SPL HNU_EVAL_CRIT=task HNU_DR=$DRV HNU_START_FRAC=0 \
      HNU_EVAL_FLOOR=1 CUDA_VISIBLE_DEVICES=$G $PY eval_student.py --student $ST --latent-dir $D/z_tables_cv_f1/Z3 \
      --per-clip 32 --out $OA --tag _Z3 >> $R/eval.log 2>&1
    [ -f $OB/eval_student_Z3.json ] || env $BE HNU_GEOM_W1=$D/geom_plan4_f1/ref_stakes_s93_h10 \
      HNU_GEOM_W2=$D/geom_plan4_f1/ref_stakes_s83_h10 HNU_SPLIT_JSON=$SPL HNU_EVAL_CRIT=task HNU_DR=$DRV HNU_START_FRAC=0 \
      HNU_EVAL_FLOOR=1 CUDA_VISIBLE_DEVICES=$G $PY eval_student.py --student $ST --latent-dir $D/z_lib_f1/plan4/Z3 \
      --per-clip 32 --out $OB --tag _Z3 >> $R/eval.log 2>&1
  done; done
  log "eval $NM done"
}
s2s() {  # export-dir out-subdir
  local E=$1 OUT=$RS/$2
  mkdir -p $OUT
  for L in $RS/v6/in/planned_layouts/*/; do  # v6 layouts (S2 = 012; robot_sim_check/planned_layouts still has the old S2 = 002)
    L=${L%/}; local n=$(basename $L) t=$(basename $L | cut -d_ -f2)
    for c in A B C; do
      local X=""
      case $c in B) X="--dr";; C) X="--dr --perc-noise 0.02 --obst-shift 0.03 --stake-jitter 0.03 --start-yaw 5 --init-off 0.02";; esac
      [ -f $OUT/${n}_$c.json ] && continue
      $PYD deploy/sim2sim.py $E --latency 0-20 --crit task --trials 100 --workers 100 --tasks $t \
        --motion-dirs $L --latent-dir $L $X --out $OUT/${n}_$c.json > $OUT/${n}_$c.log 2>&1
    done
  done
  log "sim2sim $2 done"
}
mkdir -p $R
( ev $H/runs/students_v6g10xm_base_S2k_s5/Z3/student_2000.pt S2k 0 ) &
E=$H/deploy_exports/v6g10xm_base_S2k_s5/L3; mkdir -p $E
[ -f $E/policy.onnx ] || HNU_POST_BASE=1 CUDA_VISIBLE_DEVICES=2 $PY deploy/export_student.py \
  --student $H/runs/students_v6g10xm_base_S2k_s5/Z3/student_2000.pt --latent-dir $D/z_tables_cv_f1/Z3 --out $E > $E/../export.log 2>&1
[ -f $E/parity.txt ] || $PYD deploy/parity_check.py $E > $E/parity.txt 2>&1
log "early: export S2k: $(tail -n 1 $E/parity.txt | cut -c1-80)"
s2s $H/deploy_exports/v6g10xm_f1_s5_base/L3 out_base_deployed
s2s $E out_base_S2k
wait
log "early: S2k part done"
