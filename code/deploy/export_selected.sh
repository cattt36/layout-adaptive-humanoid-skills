#!/bin/bash
# After the v6g10s students are finished and the deployment seeds selected (results_v6g10s_selection.json):
# for every level L0-L4 export the selected student (ONNX + meta + scene + trace), check parity of the numpy
# deployment runtime with the training env, and run the full sim2sim matrix (CPU MuJoCo, explicit PD).
H=/root/gpufree-data/hnu_v2
V=${V:-v6g10s}
N=$H/runs/night
PYM=/root/gpufree-data/venvs/mjlab/bin/python
PYD=/root/gpufree-data/venvs/deploy/bin/python
cd $H/code
export HF_HUB_OFFLINE=1 HNU_BRIDGE_ROOT=$H/data/ref_bridge_video_lift HNU_STAKE_H=0.10 HNU_GEOM_W1=$H/data/ref_stakes_s93_h10 \
  HNU_GEOM_W2=$H/data/ref_stakes_s83_h10 HNU_STAKE_TOL=0.17 PYTHONPATH=$H/code
for i in $(seq 1 3000); do grep -q finished $N/$V.log 2>/dev/null && break; sleep 60; done
SEL=$H/runs/results_${V}_selection.json
OUT=$H/deploy_exports/$V; mkdir -p $OUT
for P in Z0 Z1 Z2 Z3 Z4; do
  s=$($PYD -c "import json; print(json.load(open('$SEL'))['$P']['selected_seed'])")
  E=$OUT/L${P#Z}_s$s
  [ -f $E/policy.onnx ] || CUDA_VISIBLE_DEVICES=0 $PYM deploy/export_student.py --student $H/runs/students_${V}_d2_s$s/$P/student_2000.pt \
    --latent-dir $H/data/z_tables_v5/$P --out $E > $E.export.log 2>&1
  $PYD deploy/parity_check.py $E > $E/parity.txt 2>&1
  bash deploy/run_sim2sim.sh $E 48 > $E/sim2sim.log 2>&1
  echo "== L${P#Z} (seed $s): $(tail -n 1 $E/parity.txt)" >> $OUT/summary.txt
  cat $E/sim2sim_summary.txt >> $OUT/summary.txt
done
echo done >> $OUT/summary.txt
