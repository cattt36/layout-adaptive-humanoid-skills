#!/bin/bash
# Latency sensitivity of the 0.10 m stake experts: fixed 20 ms, 0-20 ms pieces, fixed 40 ms (strict, clean).
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
cd $H/code
export HNU_PERCEPTION=1 HNU_STAKE_H=0.10 HNU_STAKE_TOL=0.17 HNU_ANCHOR_XY=0.40 HNU_START_FRAC=0 HNU_STAKE_SKIP_FIRST=0
for T in ThreeStepPoleRun FiveStepPoleRun; do
  D=$H/data/ref_stakes_s93_h10; [ $T = FiveStepPoleRun ] && D=$H/data/ref_stakes_s83_h10
  CK=$(cat $H/runs/stakes_eval/g2_h10/$T/checkpoint.txt)
  for C in "fix20 1 1" "var0-20 0 1" "fix40 2 2"; do
    set -- $C; O=/tmp/draglv/$T/$1; mkdir -p $O
    HNU_DR=1 HNU_DR_PARTS=lag HNU_DR_LAG_MIN=$2 HNU_DR_LAG_MAX=$3 HNU_STAKES_ROOT=$D CUDA_VISIBLE_DEVICES=0 $PY eval_expert.py --task $T \
      --stakes --ckpt $CK --per-clip 4 --motion-dir $D/$T --stake-top 0.10 --out $O > $O/log.txt 2>&1
    $PY -c "import json,glob; j=json.load(open(glob.glob('$O/*.json')[0])); print('$T', '$1', 'all-stakes success', round(j['success_all_stakes'],3))"
  done
done
