#!/bin/bash
# Wait for tools/fix_fold1_z_gpu.sh, then summarise the fold-1 z export.
H=/root/gpufree-data/hnu_v2
while pgrep -f "bash tools/fix_fold1_z_gpu.sh" > /dev/null; do sleep 20; done
cd $H
date
grep -h normalizers runs/night/cv_z_f1_Z*.log | sort | uniq -c
grep -h normalizers runs/night/cv_zts_f1_Z*.log | sort | uniq -c
grep -il "error\|traceback" runs/night/cv_z_f1_Z*.log runs/night/cv_zts_f1_Z*.log
for P in Z0 Z1 Z2 Z3 Z4; do
  echo "$P tables $(ls data/z_tables_cv_f1/$P | wc -l)  ts090 $(ls data/ts/ts090/z_enc_f1/$P 2>/dev/null | wc -l)  ts095 $(ls data/ts/ts095/z_enc_f1/$P 2>/dev/null | wc -l)"
done
wc -l data/clip_split_cv/fold1/clips_train_aug.txt
tail -5 runs/fix_fold1_z_gpu.out 2>/dev/null
