#!/bin/bash
# Deployment_D model with the CUT post bases of the real bridge (user: inner posts 15 cm, end posts 20 cm
# from the post axis to the bar end; braces re-welded, assumed foot = L - 5 cm from the axis, top height = foot).
# Same policy / scene as v6g10xm_base_S2k_s5_mDh40 (bases resized at run time from meta post_base.mid / .end, see
# deploy/hnu_deploy.py add_post_bases); validation re-run; the 3 bridge layouts A / B / C must equal out_cut_s2k.
PYD=/root/gpufree-data/venvs/deploy/bin/python
H=/root/gpufree-data/hnu_v2; RS=/root/gpufree-data/robot_sim_check
S=$H/deploy_exports/v6g10xm_base_S2k_s5_mDh40/L3; E=$H/deploy_exports/v6g10xm_base_S2k_s5_mDh40_cut/L3
mkdir -p $E
for f in policy.onnx scene.mjb trace.npz parity.txt; do cp -n $S/$f $E/; done
$PYD - $S/policy_meta.json $E/policy_meta.json << 'EOF'
import json, sys
m = json.load(open(sys.argv[1]))
mid = lambda L: dict(bar_len=round(2 * L, 3), brace_d=round(max(L - 0.05, 0.06), 3), brace_h=round(max(L - 0.05, 0.06), 3))
m["bridge"]["post_base"].update(mid=mid(0.15), end=mid(0.20),
                                note="real bars cut: inner posts 15 cm, end posts 20 cm from the post axis to the bar end")
json.dump(m, open(sys.argv[2], "w"), indent=1)
print(m["bridge"]["post_base"])
EOF
bash $H/code/tools/st_base_validate.sh $E
cd $H/code
export PYTHONPATH=$H/code
O=$RS/out_cut_s2k_export; mkdir -p $O
for L in $RS/v6/in/planned_layouts/[DN]*Crouched*/; do
  L=${L%/}; n=$(basename $L)
  for c in A B C; do
    X=""; case $c in B) X="--dr";; C) X="--dr --perc-noise 0.02 --obst-shift 0.03 --stake-jitter 0.03 --start-yaw 5 --init-off 0.02";; esac
    [ -f $O/${n}_$c.json ] || $PYD deploy/sim2sim.py $E --latency 0-20 --crit task --trials 100 --workers 100 --tasks CrouchedTraverse \
      --motion-dirs $L --latent-dir $L $X --out $O/${n}_$c.json > $O/${n}_$c.log 2>&1
  done
done
echo done > $O/DONE
