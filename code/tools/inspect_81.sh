#!/bin/bash
H=/root/gpufree-data/hnu_v2
cd $H
echo "== paper_root"; ls paper_root
echo "== offline 8.1 logs"
for f in runs/night/cv_offline81_f*_seed*.log runs/night/offline81_v5t_seed*.log; do echo "$f: $(tail -2 $f | tr '\n' ' ' | cut -c1-160)"; done
echo "== summaries"; ls runs/night | grep -i -E "81|summary"
echo "== aggregate_81.py head"; sed -n 1,40p code/tools/aggregate_81.py
echo "== one root"; ls paper_root/latent_cv_f2_seed42 | head; ls paper_root/latent_cv_f2_seed42/* | head -20
