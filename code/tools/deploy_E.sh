#!/bin/bash
# Deployment_E = the model (v6g10xm d2 seed 5, L3; policy md5 a3f7d9a5) for the modified venue:
#   - bridge posts: bars cut (post axis -> bar end 15 cm inner posts, 20 cm end posts), braces re-welded
#     (assumed foot = L - 5 cm from the axis, top height = foot); meta post_base.mid / .end (hnu_deploy.add_post_bases)
#   - stakes: diameter 40 cm, height 10 cm (scene exported with HNU_STAKE_R=0.20)
# Export + parity + validation + 9 planned layouts (A nominal / C with errors, 100 trials, criterion v2).
# Out: deploy_exports/v6g10xm_d2_s5_E/L3, robot_sim_check/out_E/
PY=/root/gpufree-data/venvs/mjlab/bin/python; PYD=/root/gpufree-data/venvs/deploy/bin/python
H=/root/gpufree-data/hnu_v2; D=$H/data; RS=/root/gpufree-data/robot_sim_check; O=$RS/out_E
E=$H/deploy_exports/v6g10xm_d2_s5_E/L3
cd $H/code
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HNU_BRIDGE_ROOT=$D/ref_bridge_video_lift HNU_STAKE_H=0.10 HNU_STAKE_R=0.20 \
  HNU_GEOM_W1=$D/ref_stakes_s93_h10 HNU_GEOM_W2=$D/ref_stakes_s83_h10 HNU_STAKE_TOL=0.17 HNU_START_FRAC=0.25 HNU_DR_LEVEL=1 \
  HNU_DR_LAG_SYNC=1 HNU_DR_LAG_SENSORS=1 PYTHONPATH=$H/code HNU_STAKE_SHAPE=mesh16 HNU_POST_SHAPE=mesh12
mkdir -p $E $O
[ -f $E/policy.onnx ] || HNU_POST_BASE=1 HNU_BASE_DIMS=1.0,0.10,0.04,0.39,0.37,0.03 CUDA_VISIBLE_DEVICES=0 $PY deploy/export_student.py \
  --student $H/runs/students_v6g10xm_d2_s5/Z3/student_2000.pt --latent-dir $D/z_tables_cv_f1/Z3 --out $E > $E/../export.log 2>&1
$PYD - $E/policy_meta.json << 'EOF'
import json, sys
f = sys.argv[1]; m = json.load(open(f))
cut = lambda L: dict(bar_len=round(2 * L, 3), brace_d=round(max(L - 0.05, 0.06), 3), brace_h=round(max(L - 0.05, 0.06), 3))
m["bridge"]["post_base"].update(mid=cut(0.15), end=cut(0.20))
json.dump(m, open(f, "w"), indent=1)
print(m["bridge"]["post_base"], m["stakes"])
EOF
[ -f $E/parity.txt ] || $PYD deploy/parity_check.py $E > $E/parity.txt 2>&1
md5sum $E/policy.onnx $RS/model/policy.onnx
bash tools/st_base_validate.sh $E
for L in $RS/v6/in/planned_layouts/*/; do
  L=${L%/}; n=$(basename $L); t=$(echo $n | cut -d_ -f2)
  for c in A C; do
    X=""; [ $c = C ] && X="--dr --perc-noise 0.02 --obst-shift 0.03 --stake-jitter 0.03 --start-yaw 5 --init-off 0.02"
    [ -f $O/${n}_$c.json ] || $PYD deploy/sim2sim.py $E --latency 0-20 --crit task --trials 100 --workers 6 --tasks $t \
      --motion-dirs $L --latent-dir $L $X --out $O/${n}_$c.json > $O/${n}_$c.log 2>&1 &
  done
done
wait
for f in $O/*.json; do echo "$(basename $f) $($PYD -c "import json;R=json.load(open('$f'))['rows'];print(sum(r['success_v2'] for r in R), sum(r['cause']=='fall' for r in R))")"; done
echo ALLDONE
