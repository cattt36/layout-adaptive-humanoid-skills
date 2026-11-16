#!/bin/bash
# the real bars are cut (post axis -> bar end: 15 cm inner posts, 20 cm end posts, braces re-welded).
# Same case as out_end_old/0.20 (model) run with the S2k model (trained with the full 1 m bases) to choose the
# deployed model.  Brace rule as in base_mid_sweep.sh (foot = max(L - 5 cm, 6 cm) from the axis, top height = foot).
# Out: robot_sim_check/out_cut_s2k/
PYD=/root/gpufree-data/venvs/deploy/bin/python
H=/root/gpufree-data/hnu_v2; RS=/root/gpufree-data/robot_sim_check; O=$RS/out_cut_s2k
cd $H/code
export PYTHONPATH=$H/code
E=$H/deploy_exports/v6g10xm_base_S2k_s5_mDh40/L3
mid() { $PYD -c "L=$1; d=max(L-0.05,0.06); print('{\"bar_len\": %.3f, \"brace_d\": %.3f, \"brace_h\": %.3f}' % (2*L, d, d))"; }
MID=$(mid 0.15); END=$(mid 0.20)
mkdir -p $O; echo "mid $MID end $END" > $O/dims.txt
for L in $RS/v6/in/planned_layouts/[DN]*Crouched*/; do
  L=${L%/}; n=$(basename $L)
  for c in A B C; do
    X=""; case $c in B) X="--dr";; C) X="--dr --perc-noise 0.02 --obst-shift 0.03 --stake-jitter 0.03 --start-yaw 5 --init-off 0.02";; esac
    [ -f $O/${n}_$c.json ] || HNU_BASE_MID="$MID" HNU_BASE_END="$END" $PYD deploy/sim2sim.py $E --latency 0-20 --crit task \
      --trials 100 --workers 100 --tasks CrouchedTraverse --motion-dirs $L --latent-dir $L $X --out $O/${n}_$c.json > $O/${n}_$c.log 2>&1
  done
done
echo done > $O/DONE
