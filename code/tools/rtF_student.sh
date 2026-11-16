#!/bin/bash
# real-robot retraining, stage 2 (after tools/rtF_teachers.sh):
#  1. teacher check (all clips, 16 rollouts, criterion task / v2): start as before (reference velocities) vs at rest +
#     1 s hold vs stepping + 1 s hold; current teachers vs rtF teachers  -> runs/rtF/teacher_*.json
#  2. DAgger of the deployed L3 student (model) with the rtF teachers, 3000 its (checkpoints 2000 / 3000):
#     st_base.sh recipe (fold-1 training clips, cross-trial z mix) + start from rest / stepping + hold, HNU_DR_X,
#     DR level 2 (PD +-20 %, 0-40 ms sensor latency), mocap velocity estimate, per-episode mocap pose bias
#     (+-4 cm, +-4 cm height, +-5 deg; 30 % clean), cut bars, 40 cm stakes
#  3. export (scene: cut bars + 40 cm stakes, meta as Deployment_E) + parity
#  4. sim2sim acceptance on the 9 planned layouts, 64 trials: tools/rtF_accept.sh
PY=/root/gpufree-data/venvs/mjlab/bin/python; PYD=/root/gpufree-data/venvs/deploy/bin/python
H=/root/gpufree-data/hnu_v2; D=$H/data; N=$H/runs/night; R=$H/runs/rtF
cd $H/code
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $N/rtF.log; }
last() { ls $H/runs/logs/rsl_rl/hnu_track_$1/*_$2/model_*.pt 2>/dev/null | sort -V | tail -1; }
MID='{"bar_len": 0.30, "brace_d": 0.10, "brace_h": 0.10}'
END='{"bar_len": 0.40, "brace_d": 0.15, "brace_h": 0.15}'
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HNU_STAKE_H=0.10 HNU_STAKE_R=0.20 HNU_STAKE_TOL=0.17 HNU_DR_LAG_SYNC=1 \
  HNU_DR_LAG_SENSORS=1 PYTHONPATH=$H/code HNU_STAKE_SHAPE=mesh16 HNU_POST_SHAPE=mesh12 \
  HNU_STEP_STATES=$D/step_states/step_states_realF.npz HNU_POST_BASE=1 HNU_BASE_DIMS=1.0,0.10,0.04,0.39,0.37,0.03 \
  HNU_BASE_MID="$MID" HNU_BASE_END="$END"
mkdir -p $R
CUR=$(cat $N/v6g10n_teachers.txt)
TRUN=${TRUN:-rtFb}  # teachers: rtF + continuation rtFb (real stepping states, hip / knee effort)
NEWT="--ckpt-ThreeStepPoleRun $(last ThreeStepPoleRun $TRUN) --ckpt-FiveStepPoleRun $(last FiveStepPoleRun $TRUN) --ckpt-CrouchedTraverse $(last CrouchedTraverse $TRUN)"
echo "$NEWT" > $N/rtF_teachers.txt
log "stage 2 start, teachers: $NEWT"
# ---- 1. teacher check
tev() {  # tag bank start gpu
  local T=$1 BK=$2 S=$3 G=$4 V=""
  case $S in ref) V="HNU_REST_FRAC=0";; rest) V="HNU_REST_FRAC=1 HNU_STEP_FRAC=0";; step) V="HNU_REST_FRAC=1 HNU_STEP_FRAC=1";; esac
  [ -f $R/teacher_$T/eval_teacher_$S.json ] && return
  env $V HNU_HOLD=1.0,1.0 HNU_HOLD_P=1 HNU_REST_XY=0.05 HNU_REST_YAW=10 HNU_DR=0 HNU_EVAL_CRIT=task HNU_START_FRAC=0 \
    HNU_BRIDGE_ROOT=$D/ref_bridge_video_lift HNU_GEOM_W1=$D/ref_stakes_s93_h10 HNU_GEOM_W2=$D/ref_stakes_s83_h10 \
    HNU_SPLIT_JSON=$D/clip_split_trial/split_v2.json CUDA_VISIBLE_DEVICES=$G $PY eval_student.py --teacher $BK --scene geom \
    --z-offsets 0,5,10,15,20,25,30,40,50 --latent-dir $D/z_tables_v5t/Z4 --per-clip 16 --out $R/teacher_$T --tag _$S >> $R/teval.log 2>&1
}
( for S in ref rest step; do tev cur "$CUR" $S 3; done ) &
( for S in ref rest step; do tev new "$NEWT" $S 4; done ) &
# ---- 2. student
AUG="HNU_Z_NOISE=0.3 HNU_Z_JITTER=3 HNU_STAKE_JITTER=0 HNU_STAKE_JITTER_ALONG=0.03 HNU_STAKE_JITTER_LAT=0.06 HNU_STAKE_STRAIGHT=0.3
     HNU_STAKE_CONTACT=1 HNU_BRIDGE_JITTER_XY=0.05 HNU_BRIDGE_JITTER_ZLO=-0.05 HNU_BRIDGE_JITTER_ZHI=0.06 HNU_BRIDGE_POST_JITTER=0.10
     HNU_RESET_SCALE=1.5 HNU_PUSH_SCALE=1.0 HNU_TASK_WEIGHT=CrouchedTraverse:1.5 HNU_TS_SHARE=0.5
     HNU_GEOM_W1=$D/aug_xzmix_f1/ref_stakes_s93_h10 HNU_GEOM_W2=$D/aug_xzmix_f1/ref_stakes_s83_h10
     HNU_BRIDGE_ROOT=$D/aug_xzmix_f1/ref_bridge_video_lift HNU_BASE_DZ_LO=-0.005 HNU_BASE_DZ_HI=0.01"
NEWS="HNU_START_FRAC=0.4 HNU_REST_FRAC=0.6 HNU_STEP_FRAC=0.5 HNU_HOLD=0.2,4.0 HNU_HOLD_P=0.8 HNU_REST_XY=0.10 HNU_REST_YAW=15
      HNU_DR=1 HNU_DR_X=1 HNU_DR_X_HIPKNEE=1 HNU_DR_LEVEL=2 HNU_VEL_EST=1 HNU_PERC_BIAS=0.04,0.04,5,0.3"
SRC=$H/runs/students_v6g10xm_d2_s5/Z3/student_2000.pt
O=$H/runs/students_rtF_s5/Z3
IT=${IT:-3000}
[ -f $O/student_$IT.pt ] || env $AUG $NEWS HNU_CLIPS_ALLOW=$D/clip_split_cv/fold1/clips_train_xzmix.txt \
  HNU_TB_ROOT=$H/runs/logs/rsl_rl/students_rtF CUDA_VISIBLE_DEVICES=0 $PY distill_train.py --profile rtF_Z3 $NEWT \
  --num-envs 4096 --iters $IT --beta-iters 1 --lr 2e-4 --action-noise 0.1 --fail-weight 2 --seed 30005 --scene geom \
  --hidden 1024,512,256 --hist 20 --lin-vel --z-offsets 0,5,10,15,20,25,30,40,50 --init-from $SRC \
  --latent-dir $D/z_tables_cv_f1_xzmix/Z3 --out $O > $N/rtF_student.log 2>&1
log "student: $([ -f $O/student_$IT.pt ] && echo ok || echo FAILED)"
# ---- 3. export (checkpoints 2000 and IT)
for C in 2000 $IT; do
  E=$H/deploy_exports/rtF_s5_c$C/L3; mkdir -p $E
  [ -f $E/policy.onnx ] || HNU_BRIDGE_ROOT=$D/ref_bridge_video_lift HNU_GEOM_W1=$D/ref_stakes_s93_h10 \
    HNU_GEOM_W2=$D/ref_stakes_s83_h10 HNU_START_FRAC=0.25 HNU_DR_LEVEL=1 \
    CUDA_VISIBLE_DEVICES=1 $PY deploy/export_student.py --student $O/student_$C.pt \
    --latent-dir $D/z_tables_cv_f1/Z3 --out $E > $E/../export.log 2>&1
  $PYD - $E/policy_meta.json << 'EOF'
import json, sys
f = sys.argv[1]; m = json.load(open(f))
cut = lambda L: dict(bar_len=round(2 * L, 3), brace_d=round(max(L - 0.05, 0.06), 3), brace_h=round(max(L - 0.05, 0.06), 3))
m["bridge"]["post_base"].update(mid=cut(0.15), end=cut(0.20))
m["start"] = dict(hold_s=1.0, note="after switching on the policy, give frame 0 for hold_s seconds "
                  "(student trained 0.2-4 s, from rest or from a stepping-in-place controller), then advance the frame every 20 ms")
json.dump(m, open(f, "w"), indent=1)
print(m["bridge"]["post_base"], m["stakes"])
EOF
  [ -f $E/parity.txt ] || $PYD deploy/parity_check.py $E > $E/parity.txt 2>&1
  log "export c$C: $(tail -n 1 $E/parity.txt | cut -c1-60)"
done
wait
log "stage 2 done"
bash tools/rtF_accept.sh
