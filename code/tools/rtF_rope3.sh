#!/bin/bash
# , user: the rope will be lowered / released slowly on the real robot, "at least 10 s".
# Slow-release round on top of the rope round (rtF_rope2.sh):
#  - teachers rtFd: 600 its from rtFc, rope pull U(0, 0.9) x body weight released over U(1, 12) s (80 % of the
#    frame-0-pose rest starts), clock held until 0.3-2 s after the release, episodes 25 s (HNU_EP_LEN)
#  - student rtFs on GPU 4: 1500 its from the latest rope student (rtFr) with the same start mix
#  - export rtFs_s5_c1500 + parity + acceptance (incl. RP10: 80 % BW released over 10 s, 12 s hold)
PY=/root/gpufree-data/venvs/mjlab/bin/python; PYD=/root/gpufree-data/venvs/deploy/bin/python
H=/root/gpufree-data/hnu_v2; D=$H/data; N=$H/runs/night
cd $H/code
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $N/rtF.log; }
last() { ls $H/runs/logs/rsl_rl/hnu_track_$1/*_$2/model_*.pt 2>/dev/null | sort -V | tail -1; }
ROPE_SPEC="0.9,1.0,12.0,0.8"
RUN=rtFd SRCRUN=rtFc IT=600 HOLD=0.2,4.0 STEPS=$D/step_states/step_states_realF.npz HIPKNEE=1 ROPE=$ROPE_SPEC REST_RP=3 \
  EP_LEN=25 G_CT=1 G_W1=2 G_W2=3 bash tools/rtF_teachers2.sh > $N/rtF_teachers4.out 2>&1
NEWT="--ckpt-ThreeStepPoleRun $(last ThreeStepPoleRun rtFd) --ckpt-FiveStepPoleRun $(last FiveStepPoleRun rtFd) --ckpt-CrouchedTraverse $(last CrouchedTraverse rtFd)"
echo "$NEWT" > $N/rtFd_teachers.txt
SRC=$(ls $H/runs/students_rtFr_s5/Z3/student_*.pt | sort -V | tail -1)
log "rope3 (slow release) student start from $SRC, teachers $NEWT"
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
NEWS="HNU_START_FRAC=0.4 HNU_REST_FRAC=0.7 HNU_STEP_FRAC=0.25 HNU_HOLD=0.2,4.0 HNU_HOLD_P=0.8 HNU_REST_XY=0.10 HNU_REST_YAW=15
      HNU_DR=1 HNU_DR_X=1 HNU_DR_X_HIPKNEE=1 HNU_DR_LEVEL=2 HNU_VEL_EST=1 HNU_PERC_BIAS=0.04,0.04,5,0.3
      HNU_ROPE=0.5,0.3,1.5,0.8 HNU_REST_RP=3"
NEWS="$(echo $NEWS | sed 's/HNU_ROPE=0.5,0.3,1.5,0.8/HNU_ROPE=0.9,1.0,12.0,0.8/') HNU_EP_LEN=25"
O=$H/runs/students_rtFs_s5/Z3
[ -f $O/student_1500.pt ] || env $AUG $NEWS HNU_CLIPS_ALLOW=$D/clip_split_cv/fold1/clips_train_xzmix.txt \
  HNU_TB_ROOT=$H/runs/logs/rsl_rl/students_rtFs CUDA_VISIBLE_DEVICES=4 $PY distill_train.py --profile rtFs_Z3 $NEWT \
  --num-envs 4096 --iters 1500 --beta-iters 1 --lr 2e-4 --action-noise 0.1 --fail-weight 2 --seed 30007 --scene geom \
  --hidden 1024,512,256 --hist 20 --lin-vel --z-offsets 0,5,10,15,20,25,30,40,50 --init-from $SRC \
  --latent-dir $D/z_tables_cv_f1_xzmix/Z3 --out $O > $N/rtFs_student.log 2>&1
log "rope3 student: $([ -f $O/student_1500.pt ] && echo ok || echo FAILED)"
E=$H/deploy_exports/rtFs_s5_c1500/L3; mkdir -p $E
[ -f $E/policy.onnx ] || HNU_BRIDGE_ROOT=$D/ref_bridge_video_lift HNU_GEOM_W1=$D/ref_stakes_s93_h10 \
  HNU_GEOM_W2=$D/ref_stakes_s83_h10 HNU_START_FRAC=0.25 HNU_DR_LEVEL=1 CUDA_VISIBLE_DEVICES=4 $PY deploy/export_student.py \
  --student $O/student_1500.pt --latent-dir $D/z_tables_cv_f1/Z3 --out $E > $E/../export.log 2>&1
$PYD - $E/policy_meta.json << 'PEOF'
import json, sys
f = sys.argv[1]; m = json.load(open(f))
cut = lambda L: dict(bar_len=round(2 * L, 3), brace_d=round(max(L - 0.05, 0.06), 3), brace_h=round(max(L - 0.05, 0.06), 3))
m["bridge"]["post_base"].update(mid=cut(0.15), end=cut(0.20))
m["start"] = dict(hold_s="rope release time + 1-2 s", note="hang on the rope in the frame-0 pose, set the feet "
                  "down, switch the policy on with frame 0, lower / release the rope slowly (trained: up to 90 % body weight "
                  "on the rope at the switch, released over 1-12 s), keep frame 0 until 1-2 s after the rope is slack, then "
                  "advance the frame every 20 ms")
json.dump(m, open(f, "w"), indent=1)
PEOF
[ -f $E/parity.txt ] || $PYD deploy/parity_check.py $E > $E/parity.txt 2>&1
log "rope3 export c1500: $(tail -n 1 $E/parity.txt | cut -c1-60)"
while pgrep -f "tools/rtF_accept.sh" > /dev/null; do sleep 60; done
bash tools/rtF_accept.sh mE r2000 s1500
log "rope3 done"
