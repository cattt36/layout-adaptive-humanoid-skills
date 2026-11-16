#!/bin/bash
# Task-scene retention: current vs new W2 teacher on the DEPLOYED scene (0.10 m stakes, stride x0.83),
# nominal, from frame 0, strict reference-tracking criterion of eval_expert.py (no stake jitter / pushes / DR).
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
cd $H/code
D=$H/data/ref_stakes_s83_h10
export HF_HUB_OFFLINE=1 HNU_DR=0 HNU_START_FRAC=0 HNU_PERCEPTION=1 HNU_STAKE_H=0.10 HNU_STAKE_TOL=0.17 HNU_ANCHOR_XY=0.40 \
  HNU_RUNUP_FREE_S=0.8 HNU_STAKE_SKIP_FIRST=0 HNU_STAKES_ROOT=$D
cur=$(sed 's/ /\n/g' $H/runs/night/v6g10t_teachers.txt | grep -A1 -- "--ckpt-FiveStepPoleRun" | tail -1)
new=$(ls $H/runs/logs/rsl_rl/hnu_track_FiveStepPoleRun/*_tf2_full/model_*.pt | sort -V | tail -1)
for who in current new; do
  CK=$cur; [ $who = new ] && CK=$new
  O=$H/runs/retention_task/W2_$who
  mkdir -p $O
  CUDA_VISIBLE_DEVICES=$([ $who = current ] && echo 0 || echo 1) $PY eval_expert.py --task FiveStepPoleRun --stakes --ckpt $CK \
    --per-clip 16 --motion-dir $D/FiveStepPoleRun --stake-top 0.10 --out $O > $O.log 2>&1 &
done
wait
$PY - << 'EOF'
import glob, json
for who in ("current", "new"):
    f = glob.glob(f"/root/gpufree-data/hnu_v2/runs/retention_task/W2_{who}/*.json")
    if not f:
        print(who, "no result"); continue
    j = json.load(open(f[0]))
    keys = [k for k in j if isinstance(j[k], (int, float))]
    print(f"{who:8s} " + "  ".join(f"{k}={j[k]:.3f}" for k in keys))
EOF
