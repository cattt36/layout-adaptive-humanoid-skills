#!/bin/bash
# Teacher acceptance under the CORRECTED collision geometry (16-gon stake prisms, 12-gon post prisms):
#  1. old vs new bank, criterion v2, 4 conditions, 32 rollouts per clip (split-v2 exclusions removed)
#  2. strict-tracking retention in the deployed scene (W1 0.10 m / x0.93, W2 0.10 m / x0.83), old vs new
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
cd $H/code
export HF_HUB_OFFLINE=1 HNU_BRIDGE_ROOT=$H/data/ref_bridge_video_lift HNU_STAKE_H=0.10 HNU_GEOM_W1=$H/data/ref_stakes_s93_h10 \
  HNU_GEOM_W2=$H/data/ref_stakes_s83_h10 HNU_STAKE_TOL=0.17 HNU_DR_LEVEL=1 HNU_DR_LAG_SYNC=1 HNU_DR_LAG_SENSORS=1 \
  PYTHONPATH=$H/code HNU_SPLIT_JSON=$H/data/clip_split_trial/split_v2.json HNU_EVAL_CRIT=task HNU_START_FRAC=0 \
  HNU_STAKE_SHAPE=mesh16 HNU_POST_SHAPE=mesh12
O=$H/runs/accept_prism
OLD=$(cat $H/runs/night/v6g10t_teachers.txt); NEW=$(cat $H/runs/tf2_final_teachers.txt)
for DRV in 0 1; do for X in "" "--push"; do
  HNU_DR=$DRV CUDA_VISIBLE_DEVICES=0 $PY eval_student.py --teacher $OLD --scene geom --z-offsets 0,5,10,15,20,25,30,40,50 \
    --latent-dir $H/data/z_tables_v5t/Z4 --per-clip 32 $X --out $O/dr$DRV --tag _old > /dev/null 2>&1 &
  HNU_DR=$DRV CUDA_VISIBLE_DEVICES=1 $PY eval_student.py --teacher $NEW --scene geom --z-offsets 0,5,10,15,20,25,30,40,50 \
    --latent-dir $H/data/z_tables_v5t/Z4 --per-clip 32 $X --out $O/dr$DRV --tag _new > /dev/null 2>&1 &
  wait
done; done
$PY - << 'EOF'
import json
import numpy as np
O = "/root/gpufree-data/hnu_v2/runs/accept_prism"
EXC = set(json.load(open("/root/gpufree-data/hnu_v2/data/clip_split_trial/split_v2.json"))["excluded"])
T = ["CrouchedTraverse", "ThreeStepPoleRun", "FiveStepPoleRun"]
print("prism geometry, v2 success % DuMu / ThreeStepPoleRun / FiveStepPoleRun  (body err cm)")
for dr in ("0", "1"):
    for x in ("", "_push"):
        cells = []
        for who in ("old", "new"):
            rows = [c for c in json.load(open(f"{O}/dr{dr}/eval_teacher_{who}{x}.json"))["clips"] if c["clip"] not in EXC]
            g = lambda t, k: np.mean([c[k] for c in rows if c["task"] == t])  # noqa: E731
            cells.append(f"{who}: " + "/".join(f"{100 * g(t, 'success_v2'):.0f}" for t in T) + "  (" +
                         "/".join(f"{100 * g(t, 'body_err'):.1f}" for t in T) + ")")
        print(f"{'nominal' if dr == '0' else 'random '}{'+push' if x else '     '}  " + "   |   ".join(cells))
EOF
echo "== deployed-scene retention (strict tracking), prisms"
for T in ThreeStepPoleRun FiveStepPoleRun; do
  D=$H/data/ref_stakes_s93_h10; [ $T = FiveStepPoleRun ] && D=$H/data/ref_stakes_s83_h10
  for who in old new; do
    CK=$(sed 's/ /\n/g' $([ $who = old ] && echo $H/runs/night/v6g10t_teachers.txt || echo $H/runs/tf2_final_teachers.txt) | grep -A1 -- "--ckpt-$T" | tail -1)
    O2=$H/runs/retention_prism/${T}_$who; mkdir -p $O2
    HNU_DR=0 HNU_START_FRAC=0 HNU_PERCEPTION=1 HNU_STAKE_H=0.10 HNU_STAKE_TOL=0.17 HNU_ANCHOR_XY=0.40 HNU_RUNUP_FREE_S=0.8 \
      HNU_STAKE_SKIP_FIRST=0 HNU_STAKES_ROOT=$D HNU_CLIPS_ALLOW=$H/data/clip_split_trial/clips_train.txt,$H/data/clip_split_trial/clips_val.txt,$H/data/clip_split_trial/clips_test.txt \
      CUDA_VISIBLE_DEVICES=$([ $who = old ] && echo 0 || echo 1) $PY eval_expert.py --task $T --stakes --ckpt $CK \
      --per-clip 16 --motion-dir $D/$T --stake-top 0.10 --out $O2 > $O2.log 2>&1 &
  done
  wait
done
$PY - << 'EOF'
import glob, json
for T in ("ThreeStepPoleRun", "FiveStepPoleRun"):
    for who in ("old", "new"):
        f = glob.glob(f"/root/gpufree-data/hnu_v2/runs/retention_prism/{T}_{who}/*.json")
        j = json.load(open(f[0])) if f else {}
        print(f"{T} {who:3s}: success {j.get('success_rate', float('nan')):.3f}  stake hit {j.get('stake_hit_rate', float('nan')):.3f}  "
              f"body err {j.get('mean_body_pos_err', float('nan')):.3f} m")
EOF
