#!/bin/bash
# Verify the claim "W2's retention drop comes from the shortened stride, the motion itself is intact":
#  (1) retention at the ORIGINAL stride (near-flat 0.01 m): flat expert vs robust W2 (already: 88.4 -> 69.6 %)
#  (2) retention at the ROBOT-SCALE stride (0.83, near-flat 0.01 m): flat expert vs robust W2
#      -> if (2) is retained, the drop in (1) is stride-specific, not a loss of the motion pattern
#  (3) same for W1 (stride 0.93) for comparison
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
cd $H/code
mk() { [ -d $3/$1 ] || $PY stage1_reference/make_stakes.py --task $1 --ref $H/data/ref_v1/$1 --csv $H/data/scene/$1.csv --height 0.01 \
  --hscale $2 --out $3 > /dev/null 2>&1; rm -f $3/$1/ThreeStepPoleRun_007_*; }
mk FiveStepPoleRun 0.83 $H/data/ref_stakes_s83_h01
mk ThreeStepPoleRun 0.93 $H/data/ref_stakes_s93sc_h01
for T in FiveStepPoleRun ThreeStepPoleRun; do
  D=$H/data/ref_stakes_s83_h01; [ $T = ThreeStepPoleRun ] && D=$H/data/ref_stakes_s93sc_h01
  echo "$T robot-scale stride: $(CUDA_VISIBLE_DEVICES=0 $PY tools/retention_check.py --task $T --mode stakes_flat --data-dir $D \
    --orig $(cat $H/runs/expert_eval/$T/checkpoint.txt) --tuned $(cat $H/runs/stakes_eval/robust_stakes_h10c/$T/checkpoint.txt) --perc-tuned 1)"
done
