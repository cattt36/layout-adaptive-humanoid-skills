#!/bin/bash
# real-robot round 3 (user decisions ): proprioception only -- no obstacle features, no base linear
# velocity (both inputs constant 0, slots kept); waist roll / pitch locked (faulty real waist); CT swing-foot scuff
# penalty in the teacher.  Paper models / results unchanged.
#  1. wait for the teachers rtGw (tools/rtF_teachers2.sh: 800 its from rtFd, HNU_WAIST_LOCK=1,
#     HNU_SWING_SCUFF_W=-1, rope 0.9 BW released over 1-12 s, 25 s episodes)
#  2. student rtGb: DAgger from Deployment_F (rtFs c1500), 2500 its, HNU_OBST_ZERO / HNU_LINVEL_ZERO /
#     HNU_WAIST_LOCK, start placement +-5 cm / +-5 deg (the blind student cannot see the start error)
#  3. export (meta: obstacle_input / lin_vel_input "zero", waist_lock, post bases, start) + parity
#  4. acceptance: 9 planned layouts x conditions (64 trials) + every retrieval candidate of every layout (45 = 5 per
#     layout, the candidate tables of the chain) in the closest-to-real condition (32 trials)
PY=/root/gpufree-data/venvs/mjlab/bin/python; PYD=/root/gpufree-data/venvs/deploy/bin/python
H=/root/gpufree-data/hnu_v2; D=$H/data; N=$H/runs/night; RS=/root/gpufree-data/robot_sim_check
cd $H/code
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $N/rtF.log; }
last() { ls $H/runs/logs/rsl_rl/hnu_track_$1/*_$2/model_*.pt 2>/dev/null | sort -V | tail -1; }
while ! grep -q "teachers rtGw done" $N/rtF.log; do sleep 60; done
NEWT="--ckpt-ThreeStepPoleRun $(last ThreeStepPoleRun rtGw) --ckpt-FiveStepPoleRun $(last FiveStepPoleRun rtGw) --ckpt-CrouchedTraverse $(last CrouchedTraverse rtGw)"
echo "$NEWT" > $N/rtGw_teachers.txt
SRC=$H/runs/students_rtFs_s5/Z3/student_1500.pt
log "rtG blind student start from $SRC, teachers $NEWT"
MID='{"bar_len": 0.30, "brace_d": 0.10, "brace_h": 0.10}'
END='{"bar_len": 0.40, "brace_d": 0.15, "brace_h": 0.15}'
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HNU_STAKE_H=0.10 HNU_STAKE_R=0.20 HNU_STAKE_TOL=0.17 HNU_DR_LAG_SYNC=1 \
  HNU_DR_LAG_SENSORS=1 PYTHONPATH=$H/code HNU_STAKE_SHAPE=mesh16 HNU_POST_SHAPE=mesh12 \
  HNU_STEP_STATES=$D/step_states/step_states_realF.npz HNU_POST_BASE=1 HNU_BASE_DIMS=1.0,0.10,0.04,0.39,0.37,0.03 \
  HNU_BASE_MID="$MID" HNU_BASE_END="$END" HNU_WAIST_LOCK=1 HNU_OBST_ZERO=1 HNU_LINVEL_ZERO=1
AUG="HNU_Z_NOISE=0.3 HNU_Z_JITTER=3 HNU_STAKE_JITTER=0 HNU_STAKE_JITTER_ALONG=0.03 HNU_STAKE_JITTER_LAT=0.06 HNU_STAKE_STRAIGHT=0.3
     HNU_STAKE_CONTACT=1 HNU_BRIDGE_JITTER_XY=0.05 HNU_BRIDGE_JITTER_ZLO=-0.05 HNU_BRIDGE_JITTER_ZHI=0.06 HNU_BRIDGE_POST_JITTER=0.10
     HNU_RESET_SCALE=1.5 HNU_PUSH_SCALE=1.0 HNU_TASK_WEIGHT=CrouchedTraverse:1.5 HNU_TS_SHARE=0.5
     HNU_GEOM_W1=$D/aug_xzmix_f1/ref_stakes_s93_h10 HNU_GEOM_W2=$D/aug_xzmix_f1/ref_stakes_s83_h10
     HNU_BRIDGE_ROOT=$D/aug_xzmix_f1/ref_bridge_video_lift HNU_BASE_DZ_LO=-0.005 HNU_BASE_DZ_HI=0.01"
NEWS="HNU_START_FRAC=0.4 HNU_REST_FRAC=0.7 HNU_STEP_FRAC=0.25 HNU_HOLD=0.2,4.0 HNU_HOLD_P=0.8 HNU_REST_XY=0.05 HNU_REST_YAW=5
      HNU_DR=1 HNU_DR_X=1 HNU_DR_X_HIPKNEE=1 HNU_DR_LEVEL=2 HNU_ROPE=0.9,1.0,12.0,0.8 HNU_REST_RP=3 HNU_EP_LEN=25"
O=$H/runs/students_rtGb_s5/Z3
IT=2500
[ -f $O/student_$IT.pt ] || env $AUG $NEWS HNU_CLIPS_ALLOW=$D/clip_split_cv/fold1/clips_train_xzmix.txt \
  HNU_TB_ROOT=$H/runs/logs/rsl_rl/students_rtGb CUDA_VISIBLE_DEVICES=0 $PY distill_train.py --profile rtGb_Z3 $NEWT \
  --num-envs 4096 --iters $IT --beta-iters 1 --lr 2e-4 --action-noise 0.1 --fail-weight 2 --seed 30008 --scene geom \
  --hidden 1024,512,256 --hist 20 --lin-vel --z-offsets 0,5,10,15,20,25,30,40,50 --init-from $SRC \
  --latent-dir $D/z_tables_cv_f1_xzmix/Z3 --out $O > $N/rtGb_student.log 2>&1
log "rtG student: $([ -f $O/student_$IT.pt ] && echo ok || echo FAILED)"
for C in 1500 $IT; do
  E=$H/deploy_exports/rtGb_s5_c$C/L3; mkdir -p $E
  [ -f $E/policy.onnx ] || HNU_BRIDGE_ROOT=$D/ref_bridge_video_lift HNU_GEOM_W1=$D/ref_stakes_s93_h10 \
    HNU_GEOM_W2=$D/ref_stakes_s83_h10 HNU_START_FRAC=0.25 HNU_DR_LEVEL=1 CUDA_VISIBLE_DEVICES=1 $PY deploy/export_student.py \
    --student $O/student_$C.pt --latent-dir $D/z_tables_cv_f1/Z3 --out $E > $E/../export.log 2>&1
  $PYD - $E/policy_meta.json << 'PEOF'
import json, sys
f = sys.argv[1]; m = json.load(open(f))
cut = lambda L: dict(bar_len=round(2 * L, 3), brace_d=round(max(L - 0.05, 0.06), 3), brace_h=round(max(L - 0.05, 0.06), 3))
m["bridge"]["post_base"].update(mid=cut(0.15), end=cut(0.20))
m["obstacle_input"] = "zero"
m["lin_vel_input"] = "zero"
m["waist_lock"] = ["waist_roll_joint", "waist_pitch_joint"]
m["start"] = dict(hold_s="rope release time + 1-2 s", note="proprioception only (no mocap while the policy "
                  "runs): hang on the rope in the frame-0 pose at the planned start (pelvis +-5 cm, heading +-3 deg), set "
                  "the feet down, switch the policy on with frame 0, release the rope over ~10 s (trained 1-12 s, up to 90 % "
                  "body weight at the switch), keep frame 0 until 1-2 s after the rope is slack, then advance the frame "
                  "every 20 ms.  Waist roll / pitch get the constant target 0 (action_scale 0).")
json.dump(m, open(f, "w"), indent=1)
PEOF
  [ -f $E/parity.txt ] || $PYD deploy/parity_check.py $E > $E/parity.txt 2>&1
  log "rtG export c$C: $(tail -n 1 $E/parity.txt | cut -c1-60)"
done
# ---- acceptance (CPU)
OA=$RS/out_rtG; mkdir -p $OA
ALLB="--start-xy 0.05 --start-yaw 5 --dr-x"     # no mocap-specific deviations any more (blind policy)
for M in c$IT c1500; do
  E=$H/deploy_exports/rtGb_s5_$M/L3; mkdir -p $OA/$M
  for L in $RS/v6/in/planned_layouts/*/; do
    L=${L%/}; n=$(basename $L); t=$(echo $n | cut -d_ -f2)
    for c in A R3 RP10 RP10X RP10XS; do
      case $c in A) X="--latency 0-20";; R3) X="--latency 0-20 --start rest --hold 3";;
        RP10) X="--latency 0-20 --start rest --hold 12 --rope 0.8,10 --start-rp 3";;
        RP10X) X="--latency 0-40 --start rest --hold 12 --rope 0.8,10 --start-rp 3 $ALLB";;
        RP10XS) X="--latency 0-40 --start rest --hold 12 --rope 0.8,10 --start-rp 3 $ALLB --stake-jitter 0.03 --obst-shift 0.03";; esac
      [ -f $OA/$M/${n}_$c.json ] || $PYD deploy/sim2sim.py $E --crit task --trials 64 --workers 100 --tasks $t --waist-lock \
        --motion-dirs $L --latent-dir $L $X --out $OA/$M/${n}_$c.json > /dev/null 2>&1
    done
  done
  log "rtG acceptance $M layouts done"
done
# every retrieval candidate of every layout (chain tables), closest-to-real condition, final model
E=$H/deploy_exports/rtGb_s5_c$IT/L3; OC=$OA/cand; mkdir -p $OC
for CD in $H/runs/videos/e2e_robot_D/cand/*/*/; do
  CD=${CD%/}; lay=$(basename $(dirname $CD)); cl=$(basename $CD); t=$(echo $cl | cut -d_ -f1)
  X="--latency 0-40 --start rest --hold 12 --rope 0.8,10 --start-rp 3 $ALLB"
  [ $t != CrouchedTraverse ] && X="$X --stake-jitter 0.03 --obst-shift 0.03"
  [ -f $OC/${lay}_$cl.json ] || $PYD deploy/sim2sim.py $E --crit task --trials 32 --workers 100 --tasks $t --waist-lock \
    --motion-dirs $CD --latent-dir $CD $X --out $OC/${lay}_$cl.json > /dev/null 2>&1
done
log "rtG candidates done"
$PYD - $OA << 'PEOF' > $OA/summary.md
import glob, json, os, sys
import numpy as np
O = sys.argv[1]
sv = lambda f: 100 * np.mean([r["success_v2"] for r in json.load(open(f))["rows"]])
sc = lambda f: np.mean([r.get("swing_scuff_steps", 0) for r in json.load(open(f))["rows"]])
C = ["A", "R3", "RP10", "RP10X", "RP10XS"]
for M in sorted(d for d in os.listdir(O) if d.startswith("c")):
    print(f"## {M} (64 trials)\n\n| layout | " + " | ".join(C) + " | scuff steps (RP10X) |\n|---|" + "---|" * (len(C) + 1))
    for n in sorted({os.path.basename(f).rsplit("_", 1)[0] for f in glob.glob(f"{O}/{M}/*.json")}):
        fs = [f"{O}/{M}/{n}_{c}.json" for c in C]
        print(f"| {n.split('_')[0]} | " + " | ".join(f"{sv(f):.0f}" if os.path.exists(f) else "-" for f in fs)
              + f" | {sc(fs[3]):.1f} |" if os.path.exists(fs[3]) else " | - |")
    print()
print("## candidates (final model, 10 s rope + deviations, 32 trials)\n\n| layout | candidate | success v2 | scuff steps |\n|---|---|---|---|")
for f in sorted(glob.glob(f"{O}/cand/*.json")):
    lay, cl = os.path.basename(f)[:-5].split("_", 1)
    print(f"| {lay} | {cl} | {sv(f):.0f} | {sc(f):.1f} |")
PEOF
log "rtG done"
