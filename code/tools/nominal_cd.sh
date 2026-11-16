#!/bin/bash
# Pure nominal evaluation (no dynamics randomisation, zero latency) of the stake teachers h10c / h10d, clean.
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
cd $H/code
export HNU_PERCEPTION=1 HNU_STAKE_H=0.10 HNU_STAKE_TOL=0.17 HNU_ANCHOR_XY=0.40 HNU_START_FRAC=0 HNU_STAKE_SKIP_FIRST=0 HNU_DR=0
for T in ThreeStepPoleRun FiveStepPoleRun; do
  D=$H/data/ref_stakes_s93_h10; [ $T = FiveStepPoleRun ] && D=$H/data/ref_stakes_s83_h10
  for v in h10c h10d; do
    O=/tmp/nominal_cd/$T/$v; mkdir -p $O
    CUDA_VISIBLE_DEVICES=0 HNU_STAKES_ROOT=$D $PY eval_expert.py --task $T --stakes --ckpt $(cat $H/runs/stakes_eval/robust_stakes_$v/$T/checkpoint.txt) \
      --per-clip 8 --motion-dir $D/$T --stake-top 0.10 --out $O > $O/log.txt 2>&1
    $PY -c "import json,glob; j=json.load(open(glob.glob('$O/*.json')[0])); print('$T $v nominal all-stakes', round(100*j['success_all_stakes'],1), 'body err', round(100*j['mean_body_pos_err'],1), 'cm')"
  done
done
