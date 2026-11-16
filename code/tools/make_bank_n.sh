#!/bin/bash
# Proposed v6g10n teacher bank (prism-geometry acceptance, ): W1 = current teacher (rs10c: 99 % v2 under correct
# collisions, better strict tracking than the fine-tune), W2 + CrouchedTraverse = criterion-v2 fine-tunes (tf2_full).
# Writes runs/night/v6g10n_teachers_proposed.txt (the batch only starts once v6g10n_teachers.txt exists).
H=/root/gpufree-data/hnu_v2
get() { sed 's/ /\n/g' $1 | grep -A1 -- "--ckpt-$2" | tail -1; }
W1=$(get $H/runs/night/v6g10t_teachers.txt ThreeStepPoleRun)
W2=$(get $H/runs/tf2_final_teachers.txt FiveStepPoleRun)
DU=$(get $H/runs/tf2_final_teachers.txt CrouchedTraverse)
echo "--ckpt-ThreeStepPoleRun $W1 --ckpt-FiveStepPoleRun $W2 --ckpt-CrouchedTraverse $DU" > $H/runs/night/v6g10n_teachers_proposed.txt
cat $H/runs/night/v6g10n_teachers_proposed.txt
