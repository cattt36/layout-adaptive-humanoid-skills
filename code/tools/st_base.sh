#!/bin/bash
# student stage for the real-robot deployment (after tools/tf_base_gate.sh): the deployed L3 student
# (v6g10xm fold 1, RL seed 5, students_v6g10xm_d2_s5/Z3) fine-tuned by DAgger with the new teacher bank (CT with the
# post bases, 5PR without floor scuff), scene with the post bases (bar height 2.5-4 cm).  Same data / augmentation as
# run_v6g10xm_l40.sh (fold-1 TRAINING clips only, cross-trial z mix), round-2 settings (lr 2e-4, action noise 0.1,
# fail weight 2).  Variants: S2k 2000 its, S4k 4000 its.
# Evaluation (fold-1 split, test clips reported): condition A (own z, human clock) and B (library z + plan4), DR 0 / 1,
# with and without bases, floor-touch metric; reference = the deployed student (runs/base_baseline + results_v6g10xmB_s5).
# Then export (bases in scene.mjb) + CPU sim2sim on the 9 planned layouts (A / B / C as ).
# usage: bash tools/st_base.sh        log runs/night/st_base.log
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
CTC=${CT_OVERRIDE:-$(grep CT_CHOICE $G | awk '{print $2}')}; W2C=$(grep W2_CHOICE $G | awk '{print $2}')  # CT_OVERRIDE: manual pick
CUR=$(cat $N/v6g10n_teachers.txt)
get() { echo $CUR | sed 's/ /\n/g' | grep -A1 -- "--ckpt-$1" | tail -1; }
last() { ls $H/runs/logs/rsl_rl/hnu_track_$1/*_$2/model_*.pt 2>/dev/null | sort -V | tail -1; }
# a task whose candidates all failed keeps the current teacher (reported in the log)
CT=$(get CrouchedTraverse); W2=$(get FiveStepPoleRun)
[ -n "$CTC" ] && [ "$CTC" != None ] && CT=$(last CrouchedTraverse base_$CTC)
[ -n "$W2C" ] && [ "$W2C" != None ] && W2=$(last FiveStepPoleRun scuff_$W2C)
CK="--ckpt-ThreeStepPoleRun $(get ThreeStepPoleRun) --ckpt-FiveStepPoleRun $W2 --ckpt-CrouchedTraverse $CT"
echo "$CK" > $N/v6g10n_base_teachers.txt
log "start: CT choice $CTC, 5PR choice $W2C -> $CK"
AUG="HNU_Z_NOISE=0.3 HNU_Z_JITTER=3 HNU_STAKE_JITTER=0 HNU_STAKE_JITTER_ALONG=0.03 HNU_STAKE_JITTER_LAT=0.06 HNU_STAKE_STRAIGHT=0.3 HNU_STAKE_CONTACT=1 HNU_BRIDGE_JITTER_XY=0.05 HNU_BRIDGE_JITTER_ZLO=-0.05 HNU_BRIDGE_JITTER_ZHI=0.06 HNU_BRIDGE_POST_JITTER=0.10 HNU_RESET_SCALE=1.5 HNU_PUSH_SCALE=1.0 HNU_TASK_WEIGHT=CrouchedTraverse:1.5 HNU_TS_SHARE=0.5 HNU_GEOM_W1=$D/aug_xzmix_f1/ref_stakes_s93_h10 HNU_GEOM_W2=$D/aug_xzmix_f1/ref_stakes_s83_h10 HNU_BRIDGE_ROOT=$D/aug_xzmix_f1/ref_bridge_video_lift HNU_POST_BASE=1 HNU_BASE_DZ_LO=-0.005 HNU_BASE_DZ_HI=0.01"
ALLOW=$D/clip_split_cv/fold1/clips_train_xzmix.txt
LT=$D/z_tables_cv_f1_xzmix/Z3
SPL=$D/clip_split_cv/fold1/split.json
SRC=$H/runs/students_v6g10xm_d2_s5/Z3/student_2000.pt
BASE="--scene geom --hidden 1024,512,256 --hist 20 --lin-vel --z-offsets 0,5,10,15,20,25,30,40,50"
train() {  # variant iters gpu
  local V=$1 IT=$2 G=$3 O=$H/runs/students_v6g10xm_base_$1_s5/Z3
  [ -f $O/student_$IT.pt ] || env $AUG HNU_CLIPS_ALLOW=$ALLOW HNU_DR=1 HNU_TB_ROOT=$H/runs/logs/rsl_rl/students_v6g10xm_base_$V \
    CUDA_VISIBLE_DEVICES=$G $PY distill_train.py --profile v6g10xmbase${V}_Z3 $CK --num-envs 4096 --iters $IT --beta-iters 1 \
    --lr 2e-4 --action-noise 0.1 --fail-weight 2 --seed 30005 $BASE --init-from $SRC --latent-dir $LT --out $O \
    > $N/st_base_$V.log 2>&1
  log "student $V: $([ -f $O/student_$IT.pt ] && echo ok || echo FAILED)"
}
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
train S2k 2000 0 &
sleep 30
train S4k 4000 1 &
( ev $SRC deployed 2 ) &   # reference: the deployed student, same protocol
wait
( ev $H/runs/students_v6g10xm_base_S2k_s5/Z3/student_2000.pt S2k 0 ) &
( ev $H/runs/students_v6g10xm_base_S4k_s5/Z3/student_4000.pt S4k 1 ) &
for V in S2k S4k; do
  IT=${V%k}; IT=$((${IT#S} * 1000))
  E=$H/deploy_exports/v6g10xm_base_${V}_s5/L3; mkdir -p $E
  [ -f $E/policy.onnx ] || HNU_POST_BASE=1 CUDA_VISIBLE_DEVICES=2 $PY deploy/export_student.py \
    --student $H/runs/students_v6g10xm_base_${V}_s5/Z3/student_$IT.pt --latent-dir $D/z_tables_cv_f1/Z3 --out $E > $E/../export.log 2>&1
  [ -f $E/parity.txt ] || $PYD deploy/parity_check.py $E > $E/parity.txt 2>&1
  log "export $V: $(tail -n 1 $E/parity.txt | cut -c1-80)"
done
s2s $H/deploy_exports/v6g10xm_f1_s5_base/L3 out_base_deployed   # deployed student, all layouts, with bases
s2s $H/deploy_exports/v6g10xm_base_S2k_s5/L3 out_base_S2k
s2s $H/deploy_exports/v6g10xm_base_S4k_s5/L3 out_base_S4k
wait
log "finished"
