#!/bin/bash
# follow-up of tools/base_mid_sweep.sh: with the inner bars cut the model still trips over the bases of
# the END posts (D2 / N3: far end).  Inner posts fixed at 15 cm per side, end posts swept (same brace rule).
# Out: robot_sim_check/out_end_old/<case>/
PYD=/root/gpufree-data/venvs/deploy/bin/python
H=/root/gpufree-data/hnu_v2; RS=/root/gpufree-data/robot_sim_check; O=$RS/out_end_old
cd $H/code
export PYTHONPATH=$H/code
E=$H/deploy_exports/v6g10xm_f1_s5_base_mDh40/L3
mid() { $PYD -c "L=$1; d=max(L-0.05,0.06); print('{\"bar_len\": %.3f, \"brace_d\": %.3f, \"brace_h\": %.3f}' % (2*L, d, d))"; }
MID=$(mid 0.15)
for CASE in none 0.10 0.15 0.20 0.25 0.30; do
  [ $CASE = none ] && END='{"none": true}' || END=$(mid $CASE)
  mkdir -p $O/$CASE; echo "mid $MID end $END" > $O/$CASE/dims.txt
  for L in $RS/v6/in/planned_layouts/[DN]*Crouched*/; do
    L=${L%/}; n=$(basename $L)
    for c in A B C; do
      X=""; case $c in B) X="--dr";; C) X="--dr --perc-noise 0.02 --obst-shift 0.03 --stake-jitter 0.03 --start-yaw 5 --init-off 0.02";; esac
      [ -f $O/$CASE/${n}_$c.json ] || HNU_BASE_MID="$MID" HNU_BASE_END="$END" $PYD deploy/sim2sim.py $E --latency 0-20 --crit task \
        --trials 100 --workers 100 --tasks CrouchedTraverse --motion-dirs $L --latent-dir $L $X --out $O/$CASE/${n}_$c.json > $O/$CASE/${n}_$c.log 2>&1
    done
  done
done
echo done > $O/DONE
