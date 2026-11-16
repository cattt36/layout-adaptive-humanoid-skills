#!/bin/bash
# Why do the randomisation-trained students fail in the nominal scene (10 %) but succeed when randomised (61-74 %)?
# Evaluate the v6r2 seed-1 L0 student (clean) under: nominal / dynamics only / latency only (0-2) / fixed latency 1.
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
cd $H/code
export HF_HUB_OFFLINE=1 HNU_BRIDGE_ROOT=$H/data/ref_bridge_video_lift HNU_START_FRAC=0
ST=$H/runs/students_v6r2_d2_s1/Z0/student_2000.pt
ev() {  # tag env...
  local O=/tmp/drabl/$1; shift; mkdir -p $O
  env "$@" CUDA_VISIBLE_DEVICES=0 $PY eval_student.py --student $ST --latent-dir $H/data/z_tables_v5/Z0 --per-clip 4 --out $O > $O/log.txt 2>&1
  echo "$(basename $O): $(grep -a 'CrouchedTraverse/all' $O/log.txt | tr -s ' ' | cut -d' ' -f1-3)  $(grep -a 'ThreeStepPoleRun/all' $O/log.txt | tr -s ' ' | cut -d' ' -f1-3)"
}
ev nominal HNU_DR=0
ev dyn_only HNU_DR=1 HNU_DR_PARTS=dyn
ev lag_only HNU_DR=1 HNU_DR_PARTS=lag
ev lag_fixed1 HNU_DR=1 HNU_DR_PARTS=lag HNU_DR_LAG_MIN=1 HNU_DR_LEVEL=1
ev full HNU_DR=1
