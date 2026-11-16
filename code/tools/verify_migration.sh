#!/bin/bash
# Same output on both servers = same files: md5 of the teachers / code, and a combined md5 over the data and the venv.
cd /root/gpufree-data
md5sum $(cat hnu_v2/runs/night/v6g10n_teachers.txt | tr ' ' '\n' | grep "\.pt$") | cut -c1-12
(cd hnu_v2/code && find . -type f -name "*.py" -o -type f -name "*.sh" | sort | xargs md5sum | md5sum | sed 's/^/code: /')
for d in hnu_v2/data/z_tables_cv_f1 hnu_v2/data/z_tables_cv_f1_aug hnu_v2/data/ts/ts090/z_enc_f1 hnu_v2/data/ts/ts105/z_enc_f1 hnu_v2/data/clip_split_cv/fold1 hnu_v2/data/ref_v1 \
         hnu_v2/data/ref_stakes_s93_h10 hnu_v2/data/ref_stakes_s83_h10 hnu_v2/data/ref_bridge_video_lift hnu_v2/data/aug_ts_f1; do
  echo "$d: $(find -L $d -type f 2>/dev/null | sort | xargs md5sum 2>/dev/null | md5sum | cut -c1-12) $(find $d | wc -l)"
done
echo "venv files: $(find venvs/mjlab | wc -l)  venv site-packages md5 of sizes: $(find venvs/mjlab/lib/python3.11/site-packages -type f -printf '%P %s\n' | sort | md5sum | cut -c1-12)"
