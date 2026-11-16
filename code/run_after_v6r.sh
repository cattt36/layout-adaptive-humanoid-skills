#!/bin/bash
# Items 1-5 (where L3 / L4 should show their value), run after v6r:
#  1 execution quality (landing impact, loading rate, slip, saturation) -- recorded by every v6r evaluation
#  2 contact timing: contact-state mismatch (eval) + z_skill time-to-touchdown probe
#  3 robustness: v6r randomised evaluation (dr1)
#  4 end-to-end chain: language routing (v5 latents) x v6r execution success
#  5 few-shot executability: Q_exec trained on 5 % of the v6r student rollouts, 3 seeds per level
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
N=$H/runs/night
cd $H/code
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HNU_BRIDGE_ROOT=$H/data/ref_bridge_video_lift HNU_CTRL_AUX_DIR=$H/data/ctrl_aux_v5
export V=${V:-v6r2}
# unified geometry scene of the v6g10 / v6g10s students: 0.10 m stakes, robot-scale stride, sensor-only latency
case $V in v6g10*) export HNU_STAKE_H=0.10 HNU_GEOM_W1=$H/data/ref_stakes_s93_h10 HNU_GEOM_W2=$H/data/ref_stakes_s83_h10 \
  HNU_STAKE_TOL=0.17 HNU_DR_LEVEL=1 HNU_DR_LAG_SYNC=1 HNU_DR_LAG_SENSORS=1 ;; esac
O=$N/after_$V.md
for i in $(seq 1 3000); do grep -q "${WAIT_MARK:-finished}" $N/$V.log 2>/dev/null && break; sleep 60; done
R="$H/runs/results_${V}_s1_dr0 $H/runs/results_${V}_s2_dr0 $H/runs/results_${V}_s3_dr0"
RD="$H/runs/results_${V}_s1_dr1 $H/runs/results_${V}_s2_dr1 $H/runs/results_${V}_s3_dr1"
T0=$(ls -d $H/runs/results_${V}_*_teacher_dr0 | tail -1); T1=$(ls -d $H/runs/results_${V}_*_teacher_dr1 | tail -1)
{
  echo "## 1-2 execution quality (nominal, clean)"; $PY tools/exec_quality_table.py $T0 $R
  echo "## 1-2 execution quality (randomised, clean)"; $PY tools/exec_quality_table.py $T1 $RD
  echo "## 2 z_skill contact-timing probe"; $PY tools/z_contact_probe.py $H/data/z_tables_v5 $H/data/ref_v1
  echo "## 3 robustness"; cat $N/${V}_summary_dr0.txt $N/${V}_summary_dr1.txt
} > $O 2>&1
# 4 end-to-end
HNU_LANG_DUMP=1 PYTHONPATH=$H/data/data_code $PY tools/eval_language.py results/router_ood_benchmark_v1.json $H/runs/latent_v5 > /dev/null 2>&1
{ echo "## 4 end-to-end (nominal)"; $PY tools/e2e_chain.py $H/runs/latent_v5 $R
  echo "## 4 end-to-end (randomised)"; $PY tools/e2e_chain.py $H/runs/latent_v5 $RD; } >> $O 2>&1
# 5 few-shot Q_exec on v6r seed-1 students
for P in Z0 Z1 Z2 Z3 Z4; do
  Q=$H/runs/quality_${V}/$P; mkdir -p $Q
  STP=$H/runs/students_${V}_d2_s1/$P/student_2000.pt
  [ -f $H/runs/students_${V}_d3_s1/$P/student_1000.pt ] && STP=$H/runs/students_${V}_d3_s1/$P/student_1000.pt
  [ -f $Q/rollouts.npz ] || CUDA_VISIBLE_DEVICES=$(( ${P#Z} % 2 )) $PY stage5_quality/quality.py collect --policy student \
    --student $STP --latent-dir $H/data/z_tables_v5/$P --num-envs 2048 --steps 1500 \
    --out $Q/rollouts.npz > $Q/collect.log 2>&1
  for FR in 0.05 1.0; do for S in 0 1 2; do
    CUDA_VISIBLE_DEVICES=$(( ${P#Z} % 2 )) $PY stage5_quality/quality.py train --data $Q/rollouts.npz --heuristic-dir $H/data/heuristic \
      --train-frac $FR --seed $S --out $Q/frac${FR}_s$S > $Q/frac${FR}_s$S.log 2>&1
  done; done
done
{ echo "## 5 Q_exec (v6r students), test AUROC / risk@80% (isotonic)"
  $PY - << 'EOF'
import glob, json, os, numpy as np
for P in ["Z0", "Z1", "Z2", "Z3", "Z4"]:
    for fr in ("0.05", "1.0"):
        r = [json.load(open(f)) for f in glob.glob(f"/root/gpufree-data/hnu_v2/runs/quality_{os.environ['V']}/{P}/frac{fr}_s*/quality_eval.json")]
        if r:
            a = [x["isotonic"]["auroc"] for x in r]; k = [100 * x["isotonic"]["risk@cov80"] for x in r]
            print(f"L{P[1]} frac {fr}: AUROC {np.mean(a):.3f} ± {np.std(a):.3f}  risk@80 {np.mean(k):.2f}%  (n={len(r)})")
EOF
} >> $O 2>&1
echo "done" >> $O
