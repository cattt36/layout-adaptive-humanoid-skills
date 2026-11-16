#!/bin/bash
# Build light references for all three tasks (runs in parallel, one process per task).
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
cd $H/code/stage1_reference
for T in ThreeStepPoleRun FiveStepPoleRun CrouchedTraverse; do
  $PY build_reference.py --src /root/gpufree-data/Data-HNU-private/Retargeted/Data/$T \
    --phase $H/data/phase/$T --out $H/data/ref_v1/$T --mode light > $H/data/ref_v1_$T.log 2>&1 &
done
wait
echo done
