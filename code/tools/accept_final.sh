#!/bin/bash
# Final teacher acceptance (criterion v2): W2 + DuMu from tf2_full (accepted), W1 from tf2_w1b.
#  1. v2 evaluation, 4 conditions (summary excludes the split-v2 exclusions)
#  2. per-trial W1 (nominal, push)
#  3. W1 retention: (a) original stride, near-flat stakes vs the original flat expert; (b) DEPLOYED scene (0.10 m
#     stakes, stride x0.93), strict tracking, current vs new teacher
#  4. W1 tempo copies 0.90 / 1.10x
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
cd $H/code
echo "== 1. criterion v2"
bash tools/tf2_eval.sh final full w1b
echo "== 2. per-trial W1"
$PY tools/teacher_w1_trials.py final
$PY tools/teacher_w1_trials.py final _push | tail -22
W1N=$(ls $H/runs/logs/rsl_rl/hnu_track_ThreeStepPoleRun/*_tf2_w1b/model_*.pt | sort -V | tail -1)
W1C=$(sed 's/ /\n/g' $H/runs/night/v6g10t_teachers.txt | grep -A1 -- "--ckpt-ThreeStepPoleRun" | tail -1)
echo "== 3a. retention, original stride near-flat (vs original flat expert)"
export HF_HUB_OFFLINE=1 HNU_BRIDGE_ROOT=$H/data/ref_bridge_video_lift
echo "current: $(CUDA_VISIBLE_DEVICES=0 $PY tools/retention_check.py --task ThreeStepPoleRun --mode stakes_flat --perc-tuned 1 --orig $(cat $H/runs/expert_eval/ThreeStepPoleRun/checkpoint.txt) --tuned $W1C 2>/dev/null | tail -1)" &
echo "new:     $(CUDA_VISIBLE_DEVICES=1 $PY tools/retention_check.py --task ThreeStepPoleRun --mode stakes_flat --perc-tuned 1 --orig $(cat $H/runs/expert_eval/ThreeStepPoleRun/checkpoint.txt) --tuned $W1N 2>/dev/null | tail -1)" &
wait
echo "== 3b. retention, deployed scene (0.10 m stakes, stride x0.93), strict tracking"
D=$H/data/ref_stakes_s93_h10
for who in current new; do
  CK=$W1C; [ $who = new ] && CK=$W1N
  O=$H/runs/retention_task/W1_$who; mkdir -p $O
  HNU_DR=0 HNU_START_FRAC=0 HNU_PERCEPTION=1 HNU_STAKE_H=0.10 HNU_STAKE_TOL=0.17 HNU_ANCHOR_XY=0.40 HNU_RUNUP_FREE_S=0.8 \
    HNU_STAKE_SKIP_FIRST=0 HNU_STAKES_ROOT=$D HNU_CLIPS_ALLOW=$H/data/clip_split_trial/teacher_w1_allow.txt \
    CUDA_VISIBLE_DEVICES=$([ $who = current ] && echo 0 || echo 1) $PY eval_expert.py --task ThreeStepPoleRun --stakes --ckpt $CK \
    --per-clip 16 --motion-dir $D/ThreeStepPoleRun --stake-top 0.10 --out $O > $O.log 2>&1 &
done
wait
$PY - << 'EOF'
import glob, json
for who in ("current", "new"):
    f = glob.glob(f"/root/gpufree-data/hnu_v2/runs/retention_task/W1_{who}/*.json")
    j = json.load(open(f[0])) if f else {}
    print(f"{who:8s} success {j.get('success_rate', float('nan')):.3f}  stake hit {j.get('stake_hit_rate', float('nan')):.3f}  "
          f"body err {j.get('mean_body_pos_err', float('nan')):.3f} m")
EOF
echo "== 4. W1 tempo copies (nominal, v2)"
export HNU_STAKE_H=0.10 HNU_STAKE_TOL=0.17 HNU_DR_LEVEL=1 HNU_DR_LAG_SYNC=1 HNU_DR_LAG_SENSORS=1 PYTHONPATH=$H/code \
  HNU_SPLIT_JSON=$H/data/clip_split_trial/split_v2.json HNU_EVAL_CRIT=task HNU_START_FRAC=0 HNU_DR=0
CK=$(cat $H/runs/tf2_final_teachers.txt)
G=0
for T in 090 110; do
  R=$H/data/ts/ts$T
  HNU_GEOM_W1=$R/ref_stakes_s93_h10 HNU_GEOM_W2=$R/ref_stakes_s83_h10 HNU_BRIDGE_ROOT=$R/ref_bridge_video_lift \
    CUDA_VISIBLE_DEVICES=$G $PY eval_student.py --teacher $CK --scene geom --z-offsets 0,5,10,15,20,25,30,40,50 \
    --latent-dir $R/z_tables_v5t/Z4 --per-clip 8 --out $H/runs/tf2_final_ts/ts$T --tag _teacher > /dev/null 2>&1 &
  G=$((1 - G))
done
wait
$PY - << 'EOF'
import json
import numpy as np
H = "/root/gpufree-data/hnu_v2"
EXC = set(json.load(open(f"{H}/data/clip_split_trial/split_v2.json"))["excluded"])
for s in ("090", "110"):
    rows = [c for c in json.load(open(f"{H}/runs/tf2_final_ts/ts{s}/eval_teacher_teacher.json"))["clips"]
            if c["task"] == "ThreeStepPoleRun" and c["clip"] not in EXC]
    print(f"W1 x{int(s) / 100:.2f}: v2 success {100 * np.mean([c['success_v2'] for c in rows]):.0f} %")
EOF
