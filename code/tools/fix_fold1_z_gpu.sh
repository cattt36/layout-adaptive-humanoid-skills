#!/bin/bash
# Faster path: stop v6g10n (it restarts on the new z anyway) and the CPU exports, redo the fold-1 exports on the GPUs.
H=/root/gpufree-data/hnu_v2
N=$H/runs/night
PY=/root/gpufree-data/venvs/mjlab/bin/python
for p in $(pgrep -f "tools/watchdog.sh v6g10n"); do kill $p; done
for p in $(pgrep -f "run_v6g10n.sh"); do kill $p; done
for p in $(pgrep -f "fix_fold1_z.sh"); do kill $p; done
sleep 2
for p in $(pgrep -f "students_v6g10n|profile v6g10n|results_v6g10n_s|export_z_tables"); do kill $p; done
sleep 10
echo "[$(date +%H:%M:%S)] stopped for the z-table fix (encoder checkpoint normalizers); restart follows" >> $N/v6g10n.log
cd $H/code
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
G=0
for P in Z0 Z1 Z2 Z3 Z4; do
  ( ANNOT=annot_v2 PYTHONPATH=$H/data/data_code CUDA_VISIBLE_DEVICES=$G $PY stage4_latent/export_z_tables.py --profile $P \
      --latent-root $H/runs/latent_v5t/seed42 --motion-root $H/data/ref_v1 --out $H/data/z_tables_cv_f1/$P > $N/cv_z_f1_$P.log 2>&1
    ANNOT=annot_v2 PYTHONPATH=$H/data/data_code CUDA_VISIBLE_DEVICES=$G $PY stage4_latent/export_z_tables_ts.py --profile $P \
      --latent-root $H/runs/latent_v5t/seed42 --motion-root $H/data/ref_v1 --speeds 0.9,0.95,1.05,1.1 \
      --only ThreeStepPoleRun,FiveStepPoleRun --out-root $H/data/ts --zname z_enc_f1 > $N/cv_zts_f1_$P.log 2>&1
    ANNOT=annot_v2 PYTHONPATH=$H/data/data_code CUDA_VISIBLE_DEVICES=$G $PY stage4_latent/export_z_tables_ts.py --profile $P \
      --latent-root $H/runs/latent_v5t/seed42 --motion-root $H/data/ref_v1 --speeds 0.95,1.05 \
      --only CrouchedTraverse --out-root $H/data/ts --zname z_enc_f1 >> $N/cv_zts_f1_$P.log 2>&1 ) &
  G=$((1 - G))
done
wait
grep -h normalizers $N/cv_z_f1_Z*.log | sort | uniq -c
$PY tools/make_aug_ts.py f1
