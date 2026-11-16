#!/bin/bash
# Is it an observation-pipeline artefact?  Same student: delay buffer configured with lag fixed at 0 (identical
# information to "no delay") vs no delay buffer at all.
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
cd $H/code
export HF_HUB_OFFLINE=1 HNU_BRIDGE_ROOT=$H/data/ref_bridge_video_lift HNU_START_FRAC=0
ST=$H/runs/students_v6r2_d2_s1/Z0/student_2000.pt
ev() {
  local O=/tmp/drabl/$1; shift; mkdir -p $O
  env "$@" CUDA_VISIBLE_DEVICES=0 $PY eval_student.py --student $ST --latent-dir $H/data/z_tables_v5/Z0 --per-clip 4 --out $O > $O/log.txt 2>&1
  echo "$(basename $O): $(grep -a 'CrouchedTraverse/all' $O/log.txt | tr -s ' ' | cut -d' ' -f1-3)  $(grep -a 'ThreeStepPoleRun/all' $O/log.txt | tr -s ' ' | cut -d' ' -f1-3)"
}
ev lag_fixed0 HNU_DR=1 HNU_DR_PARTS=lag HNU_DR_LAG_MIN=0 HNU_DR_LAG_MAX=0
ev lag_fixed2 HNU_DR=1 HNU_DR_PARTS=lag HNU_DR_LAG_MIN=2 HNU_DR_LAG_MAX=2
