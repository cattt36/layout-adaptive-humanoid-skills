#!/bin/bash
# The fold-1 Z4 z-table export of fix_fold1_z_gpu.sh hung while loading the motions (no output for 35 min, killed);
# redo it alone (same command, limited threads), then rebuild the fold-1 training set.
H=/root/gpufree-data/hnu_v2
N=$H/runs/night
PY=/root/gpufree-data/venvs/mjlab/bin/python
cd $H/code
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
ANNOT=annot_v2 PYTHONPATH=$H/data/data_code CUDA_VISIBLE_DEVICES=1 $PY stage4_latent/export_z_tables.py --profile Z4 \
  --latent-root $H/runs/latent_v5t/seed42 --motion-root $H/data/ref_v1 --out $H/data/z_tables_cv_f1/Z4 > $N/cv_z_f1_Z4.log 2>&1
while pgrep -f "bash tools/fix_fold1_z_gpu.sh" > /dev/null; do sleep 15; done
grep -h normalizers $N/cv_z_f1_Z*.log $N/cv_zts_f1_Z*.log | sort | uniq -c
$PY tools/make_aug_ts.py f1
