#!/bin/bash
# evening, the user's preferred start: hang on the rope in the frame-0 pose -> set down -> switch the policy
# on (clock held at frame 0) -> release the rope -> stand still -> start the motion.
#  1. teachers rtFc: 800 its from rtFb + rope release (HNU_ROPE: pull U(0, 0.5) x body weight on the torso,
#     released over 0.3-1.5 s, 60 % of the frame-0-pose rest starts) + set down tilted (+-3 deg roll / pitch)
#  2. student rtFr: DAgger from the running rtF student at 1000 its, 2000 its with the rtFc teachers (same
#     recipe as tools/rtF_student.sh + rope) -> export rtFr_s5_c2000 (total 3000 its)
#  3. acceptance incl. RP (rope 30 % BW released in 1 s, tilt 3 deg, 2.5 s hold) and RPX (+ all deviations)
PY=/root/gpufree-data/venvs/mjlab/bin/python; PYD=/root/gpufree-data/venvs/deploy/bin/python
H=/root/gpufree-data/hnu_v2; D=$H/data; N=$H/runs/night
cd $H/code
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $N/rtF.log; }
last() { ls $H/runs/logs/rsl_rl/hnu_track_$1/*_$2/model_*.pt 2>/dev/null | sort -V | tail -1; }
ROPE_SPEC="0.5,0.3,1.5,0.6"
RUN=rtFc SRCRUN=rtFb IT=800 HOLD=0.2,4.0 STEPS=$D/step_states/step_states_realF.npz HIPKNEE=1 ROPE=$ROPE_SPEC REST_RP=3 \
  G_CT=1 G_W1=2 G_W2=3 bash tools/rtF_teachers2.sh > $N/rtF_teachers3.out 2>&1
NEWT="--ckpt-ThreeStepPoleRun $(last ThreeStepPoleRun rtFc) --ckpt-FiveStepPoleRun $(last FiveStepPoleRun rtFc) --ckpt-CrouchedTraverse $(last CrouchedTraverse rtFc)"
echo "$NEWT" > $N/rtFc_teachers.txt
SRC=$H/runs/students_rtF_s5/Z3/student_1000.pt
while [ ! -f $SRC ]; do sleep 60; done
log "rope student start from $SRC, teachers $NEWT"
MID='{"bar_len": 0.30, "brace_d": 0.10, "brace_h": 0.10}'
END='{"bar_len": 0.40, "brace_d": 0.15, "brace_h": 0.15}'
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HNU_STAKE_H=0.10 HNU_STAKE_R=0.20 HNU_STAKE_TOL=0.17 HNU_DR_LAG_SYNC=1 \
  HNU_DR_LAG_SENSORS=1 PYTHONPATH=$H/code HNU_STAKE_SHAPE=mesh16 HNU_POST_SHAPE=mesh12 \
  HNU_STEP_STATES=$D/step_states/step_states_realF.npz HNU_POST_BASE=1 HNU_BASE_DIMS=1.0,0.10,0.04,0.39,0.37,0.03 \
  HNU_BASE_MID="$MID" HNU_BASE_END="$END"
AUG="HNU_Z_NOISE=0.3 HNU_Z_JITTER=3 HNU_STAKE_JITTER=0 HNU_STAKE_JITTER_ALONG=0.03 HNU_STAKE_JITTER_LAT=0.06 HNU_STAKE_STRAIGHT=0.3
     HNU_STAKE_CONTACT=1 HNU_BRIDGE_JITTER_XY=0.05 HNU_BRIDGE_JITTER_ZLO=-0.05 HNU_BRIDGE_JITTER_ZHI=0.06 HNU_BRIDGE_POST_JITTER=0.10
     HNU_RESET_SCALE=1.5 HNU_PUSH_SCALE=1.0 HNU_TASK_WEIGHT=CrouchedTraverse:1.5 HNU_TS_SHARE=0.5
     HNU_GEOM_W1=$D/aug_xzmix_f1/ref_stakes_s93_h10 HNU_GEOM_W2=$D/aug_xzmix_f1/ref_stakes_s83_h10
     HNU_BRIDGE_ROOT=$D/aug_xzmix_f1/ref_bridge_video_lift HNU_BASE_DZ_LO=-0.005 HNU_BASE_DZ_HI=0.01"
NEWS="HNU_START_FRAC=0.4 HNU_REST_FRAC=0.6 HNU_STEP_FRAC=0.5 HNU_HOLD=0.2,4.0 HNU_HOLD_P=0.8 HNU_REST_XY=0.10 HNU_REST_YAW=15
      HNU_DR=1 HNU_DR_X=1 HNU_DR_X_HIPKNEE=1 HNU_DR_LEVEL=2 HNU_VEL_EST=1 HNU_PERC_BIAS=0.04,0.04,5,0.3
      HNU_ROPE=$ROPE_SPEC HNU_REST_RP=3"
O=$H/runs/students_rtFr_s5/Z3
[ -f $O/student_2000.pt ] || env $AUG $NEWS HNU_CLIPS_ALLOW=$D/clip_split_cv/fold1/clips_train_xzmix.txt \
  HNU_TB_ROOT=$H/runs/logs/rsl_rl/students_rtFr CUDA_VISIBLE_DEVICES=1 $PY distill_train.py --profile rtFr_Z3 $NEWT \
  --num-envs 4096 --iters 2000 --beta-iters 1 --lr 2e-4 --action-noise 0.1 --fail-weight 2 --seed 30006 --scene geom \
  --hidden 1024,512,256 --hist 20 --lin-vel --z-offsets 0,5,10,15,20,25,30,40,50 --init-from $SRC \
  --latent-dir $D/z_tables_cv_f1_xzmix/Z3 --out $O > $N/rtFr_student.log 2>&1
log "rope student: $([ -f $O/student_2000.pt ] && echo ok || echo FAILED)"
E=$H/deploy_exports/rtFr_s5_c2000/L3; mkdir -p $E
[ -f $E/policy.onnx ] || HNU_BRIDGE_ROOT=$D/ref_bridge_video_lift HNU_GEOM_W1=$D/ref_stakes_s93_h10 \
  HNU_GEOM_W2=$D/ref_stakes_s83_h10 HNU_START_FRAC=0.25 HNU_DR_LEVEL=1 CUDA_VISIBLE_DEVICES=1 $PY deploy/export_student.py \
  --student $O/student_2000.pt --latent-dir $D/z_tables_cv_f1/Z3 --out $E > $E/../export.log 2>&1
$PYD - $E/policy_meta.json << 'EOF'
import json, sys
f = sys.argv[1]; m = json.load(open(f))
cut = lambda L: dict(bar_len=round(2 * L, 3), brace_d=round(max(L - 0.05, 0.06), 3), brace_h=round(max(L - 0.05, 0.06), 3))
m["bridge"]["post_base"].update(mid=cut(0.15), end=cut(0.20))
m["start"] = dict(hold_s=1.5, note="hang on the rope in the frame-0 pose, set down, switch the policy on with "
                  "frame 0, release the rope within ~1 s, stand (frame 0 held, trained up to 4 s), then advance the frame "
                  "every 20 ms; also from a stepping walking controller")
json.dump(m, open(f, "w"), indent=1)
EOF
[ -f $E/parity.txt ] || $PYD deploy/parity_check.py $E > $E/parity.txt 2>&1
log "export rope r2000: $(tail -n 1 $E/parity.txt | cut -c1-60)"
# wait until the main chain's acceptance has started its own models, then run everything (missing cells only)
while ! grep -q "stage 2 done" $N/rtF.log; do sleep 60; done
sleep 30
while pgrep -f "tools/rtF_accept.sh" > /dev/null; do sleep 60; done
bash tools/rtF_accept.sh mE c2000 c3000 r2000
log "rope stage done"
