#!/bin/bash
# Restart the overnight pipeline from stage C with only seed 42 on the critical path; seeds 43/44 run in parallel.
H=/root/gpufree-data/hnu_v2
pkill -f 'run_nigh[t].sh'
pkill -f 'train_al[l].sh'
pkill -f 'train_laten[t].sh'
pkill -f 'motion_laten[t]_z.py'
sleep 3
rm -rf $H/runs/latent_v2
sed -i 's/SEEDS="42 43 44" bash stage4_latent\/train_all.sh/SEEDS="42" bash stage4_latent\/train_all.sh/' $H/code/run_night.sh
grep -n 'SEEDS=' $H/code/run_night.sh
cd $H/code
nohup bash run_night.sh > $H/runs/night/launcher2.log 2>&1 < /dev/null &
sleep 2
ANNOT=annot_v2 LATENT_RUN=latent_v2 GPU=0 SEEDS="43 44" nohup bash stage4_latent/train_all.sh > $H/runs/night/C_extra_seeds.log 2>&1 < /dev/null &
sleep 5
pgrep -fa 'motion_laten[t]_z.py' | wc -l
