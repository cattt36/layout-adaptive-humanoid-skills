#!/bin/bash
# md5 fingerprint of the fold data used by the students (compare 4090 vs L40).
cd /root/gpufree-data/hnu_v2/data
for d in z_tables_cv_f1 z_tables_cv_f2 z_tables_cv_f3 z_tables_cv_f4 z_tables_cv_f1_aug z_tables_cv_f2_aug z_tables_cv_f3_aug z_tables_cv_f4_aug \
         aug_ts_f1 aug_ts_f2 aug_ts_f3 aug_ts_f4 clip_split_cv; do
  echo "$d $(find -L $d -type f \( -name '*.npy' -o -name '*.npz' -o -name '*.json' -o -name '*.txt' \) | sort | xargs md5sum | md5sum | cut -c1-10) $(find -L $d -type f | wc -l)"
done
