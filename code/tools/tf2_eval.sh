#!/bin/bash
# Criterion-v2 evaluation of fine-tuned teachers (run tag) in the distillation env, all trials, 16 rollouts per clip,
# nominal / randomised x clean / push.  The CrouchedTraverse / ThreeStepPoleRun / FiveStepPoleRun checkpoint of the tag replaces the current one
# when it exists.   usage: tf2_eval.sh <name> [run-tag ...]  (later run tags override earlier ones per action;
# default run tag = name), e.g.  tf2_eval.sh final full w1b  -> W2 / DuMu from tf2_full, W1 from tf2_w1b
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
TAG=$1; shift
RUNTAGS=${@:-$TAG}
cd $H/code
export HF_HUB_OFFLINE=1 HNU_BRIDGE_ROOT=$H/data/ref_bridge_video_lift HNU_STAKE_H=0.10 HNU_GEOM_W1=$H/data/ref_stakes_s93_h10 \
  HNU_GEOM_W2=$H/data/ref_stakes_s83_h10 HNU_STAKE_TOL=0.17 HNU_DR_LEVEL=1 HNU_DR_LAG_SYNC=1 HNU_DR_LAG_SENSORS=1 \
  PYTHONPATH=$H/code HNU_SPLIT_JSON=$H/data/clip_split_trial/split.json HNU_EVAL_CRIT=task HNU_START_FRAC=0
CK=$(cat $H/runs/night/v6g10t_teachers.txt)
for RT in $RUNTAGS; do
  for T in ThreeStepPoleRun FiveStepPoleRun CrouchedTraverse; do
    N=$(ls $H/runs/logs/rsl_rl/hnu_track_$T/*_tf2_$RT/model_*.pt 2>/dev/null | sort -V | tail -1)
    if [ -n "$N" ]; then
      OLD=$(echo $CK | sed 's/ /\n/g' | grep -A1 -- "--ckpt-$T" | tail -1)
      CK=${CK/$OLD/$N}
      echo "$T -> $N"
    fi
  done
done
echo "$CK" > $H/runs/tf2_${TAG}_teachers.txt
O=$H/runs/tf2_eval_$TAG
G=0
for DRV in 0 1; do for X in "" "--push"; do
  HNU_DR=$DRV CUDA_VISIBLE_DEVICES=$G $PY eval_student.py --teacher $CK --scene geom --z-offsets 0,5,10,15,20,25,30,40,50 \
    --latent-dir $H/data/z_tables_v5t/Z4 --per-clip 16 $X --out $O/dr$DRV --tag _teacher > $O.dr$DRV$X.log 2>&1 &
  G=$((1 - G))
done; wait; done
$PY - "$TAG" << 'EOF'
import json, os, sys
import numpy as np
H = "/root/gpufree-data/hnu_v2/runs"
tag = sys.argv[1]
T = ["CrouchedTraverse", "ThreeStepPoleRun", "FiveStepPoleRun"]
EXC = set(json.load(open("/root/gpufree-data/hnu_v2/data/clip_split_trial/split_v2.json"))["excluded"])
def agg(f, key, t):  # mean over all trials of the action except the data-quality exclusions (split v2)
    v = [c[key] for c in json.load(open(f))["clips"] if c["task"] == t and c["clip"] not in EXC and c.get(key) is not None]
    return 100 * float(np.mean(v))
print(f"teacher bank '{tag}' vs current (results_v6g10mv2_bank_teacher); all trials except the split-v2 exclusions; "
      "v2 success % DuMu/W1/W2")
for dr in ("0", "1"):
    for x in ("", "_push"):
        new = f"{H}/tf2_eval_{tag}/dr{dr}/eval_teacher_teacher{x}.json"
        old = f"{H}/results_v6g10mv2_bank_teacher_dr{dr}/eval_teacher_teacher{x}.json"
        row = []
        for name, f in (("current", old), (tag, new)):
            if os.path.exists(f):
                row.append(f"{name}: v2 " + "/".join(f"{agg(f, 'success_v2', t):.0f}" for t in T) +
                           "  (fully on all stakes W1/W2 " + "/".join(f"{agg(f, 'stakes_all_full', t):.0f}" for t in T[1:]) +
                           ", body err " + "/".join(f"{agg(f, 'body_err', t):.1f}" for t in T) + " cm)")
        print(f"{'nominal' if dr == '0' else 'random '}{'+push' if x else '     '}  " + "  |  ".join(row))
EOF
