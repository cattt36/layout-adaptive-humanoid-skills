#!/bin/bash
# Fold 1 (= split_v2, latents latent_v5t) z tables with the encoder's OWN normalizers (checkpoint), exactly as the CV
# folds 2-4 will be produced: data/z_tables_cv_f1, tempo z data/ts/ts*/z_enc_f1, training set aug_ts_f1 /
# z_tables_cv_f1_aug / clip_split_cv/fold1/clips_train_aug.txt.  Old z tables are kept (not overwritten).
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
N=$H/runs/night
cd $H/code
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
for P in Z0 Z1 Z2 Z3 Z4; do
  ANNOT=annot_v2 PYTHONPATH=$H/data/data_code CUDA_VISIBLE_DEVICES= $PY stage4_latent/export_z_tables.py --profile $P \
    --latent-root $H/runs/latent_v5t/seed42 --motion-root $H/data/ref_v1 --out $H/data/z_tables_cv_f1/$P > $N/cv_z_f1_$P.log 2>&1 &
done
wait
grep -h normalizers $N/cv_z_f1_Z*.log | sort | uniq -c
for P in Z0 Z1 Z2 Z3 Z4; do
  ( ANNOT=annot_v2 PYTHONPATH=$H/data/data_code CUDA_VISIBLE_DEVICES= $PY stage4_latent/export_z_tables_ts.py --profile $P \
      --latent-root $H/runs/latent_v5t/seed42 --motion-root $H/data/ref_v1 --speeds 0.9,0.95,1.05,1.1 \
      --only ThreeStepPoleRun,FiveStepPoleRun --out-root $H/data/ts --zname z_enc_f1 > $N/cv_zts_f1_$P.log 2>&1
    ANNOT=annot_v2 PYTHONPATH=$H/data/data_code CUDA_VISIBLE_DEVICES= $PY stage4_latent/export_z_tables_ts.py --profile $P \
      --latent-root $H/runs/latent_v5t/seed42 --motion-root $H/data/ref_v1 --speeds 0.95,1.05 \
      --only CrouchedTraverse --out-root $H/data/ts --zname z_enc_f1 >> $N/cv_zts_f1_$P.log 2>&1 ) &
done
wait
$PY tools/make_aug_ts.py f1
