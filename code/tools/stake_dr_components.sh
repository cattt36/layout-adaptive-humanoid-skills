#!/bin/bash
# Which randomisation component breaks the 0.10 m stake experts?  The non-robust experts (g2_h10) are evaluated
# (strict, clean) under: nominal / PD +-20 % / effort limits 0.9-1.05 / mass +-10 % / latency 0-40 ms (0.5 s pieces).
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
cd $H/code
export HNU_PERCEPTION=1 HNU_STAKE_H=0.10 HNU_STAKE_TOL=0.17 HNU_ANCHOR_XY=0.40 HNU_START_FRAC=0 HNU_STAKE_SKIP_FIRST=0
for T in ThreeStepPoleRun FiveStepPoleRun; do
  D=$H/data/ref_stakes_s93_h10; [ $T = FiveStepPoleRun ] && D=$H/data/ref_stakes_s83_h10
  CK=$(cat $H/runs/stakes_eval/g2_h10/$T/checkpoint.txt)
  for C in nominal pd effort mass lag; do
    O=/tmp/drcomp/$T/$C; mkdir -p $O
    if [ $C = nominal ]; then E="HNU_DR=0"; else E="HNU_DR=1 HNU_DR_PARTS=$C HNU_DR_LEVEL=2"; fi
    env $E HNU_STAKES_ROOT=$D CUDA_VISIBLE_DEVICES=0 $PY eval_expert.py --task $T --stakes --ckpt $CK --per-clip 4 \
      --motion-dir $D/$T --stake-top 0.10 --out $O > $O/log.txt 2>&1
    $PY -c "import json,glob; j=json.load(open(glob.glob('$O/*.json')[0])); print('$T', '$C', 'all-stakes success', round(j['success_all_stakes'],3))"
  done
done
