#!/bin/bash
# usage: train_latent.sh <profile Z0..Z4> <out_root> [extra args...]
P=$1; OUT=$2; shift 2
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
export PYTHONPATH=$H/data/data_code
cd $H/code/stage4_latent
$PY motion_latent_z.py --mode train --datasets CrouchedTraverse ThreeStepPoleRun FiveStepPoleRun --allow-mixed-datasets \
  --json-root $H/data/${ANNOT:-annot_v1}/LLM_annotation --npz-root $H/data/latent_npz_v1 \
  --retargeted-json-root $H/data/${ANNOT:-annot_v1}/Retargeted/Dynamics \
  --ablation-profile $P --output-dir $OUT/$P "$@"
