#!/bin/bash
# Complete everything of version A (v6g10s) that is unfinished, then start version B:
#  0  free disk (approved ): partial / obsolete Q_exec rollouts
#  5  Q_exec few-shot (frac 0.05 and 1.0, 3 seeds) for L0-L4 with the CPU-batched trainer, one level at a time;
#     the raw rollouts of a level are deleted once its 6 trainings are done (disk)
#  3  robustness summaries (nominal / randomised, mean ± std over seeds)
#  D  deployment students: perception / placement / drift / set-up sweeps + layout sheets
PY=/root/gpufree-data/venvs/mjlab/bin/python
PYD=/root/gpufree-data/venvs/deploy/bin/python
H=/root/gpufree-data/hnu_v2
N=$H/runs/night
V=v6g10s
cd $H/code
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HNU_BRIDGE_ROOT=$H/data/ref_bridge_video_lift HNU_STAKE_H=0.10 \
  HNU_GEOM_W1=$H/data/ref_stakes_s93_h10 HNU_GEOM_W2=$H/data/ref_stakes_s83_h10 HNU_STAKE_TOL=0.17 HNU_DR_LEVEL=1 \
  HNU_DR_LAG_SYNC=1 HNU_DR_LAG_SENSORS=1 PYTHONPATH=$H/code
log() { echo "[$(date +%H:%M:%S)] $*" >> $N/finish_A.log; }
# 0 free disk
rm -f $H/runs/quality_$V/Z3/rollouts.npz $H/runs/quality_v4d/*/rollouts.npz
log "disk freed: $(df -h $H | tail -1)"
# D (CPU) in the background
(
  for E in $H/deploy_exports/$V/L*_s*; do
    [ -d $E ] || continue
    for SW in perc_sweep placement_sweep drift_sweep setup_sweep; do bash deploy/$SW.sh $E 40 >> $E/sweeps.log 2>&1; done
  done
  $PYD -m pip install -q matplotlib > /dev/null 2>&1
  $PYD deploy/layout_sheet.py $H/deploy_exports/$V $H/deploy_exports/$V/layout 3 > $N/layout_$V.log 2>&1
  log "deployment sweeps + layout sheets done"
) &
# 5 Q_exec, one level per GPU at a time
q5() {  # P gpu
  local P=$1 G=$2 Q=$H/runs/quality_$V/$1
  mkdir -p $Q
  local STP=$H/runs/students_${V}_d2_s1/$P/student_2000.pt
  if [ ! -f $Q/frac1.0_s2/quality_eval.json ]; then
    [ -f $Q/rollouts.npz ] || CUDA_VISIBLE_DEVICES=$G $PY stage5_quality/quality.py collect --policy student --student $STP \
      --latent-dir $H/data/z_tables_v5/$P --num-envs 2048 --steps 1500 --out $Q/rollouts.npz > $Q/collect.log 2>&1
    for FR in 0.05 1.0; do for S in 0 1 2; do
      CUDA_VISIBLE_DEVICES=$G $PY stage5_quality/quality.py train --data $Q/rollouts.npz --heuristic-dir $H/data/heuristic \
        --train-frac $FR --seed $S --out $Q/frac${FR}_s$S > $Q/frac${FR}_s$S.log 2>&1
    done; done
  fi
  n=$(ls $Q/frac*_s*/quality_eval.json 2>/dev/null | wc -l)
  [ "$n" = 6 ] && rm -f $Q/rollouts.npz
  log "Q_exec $P: $n/6 trainings, rollouts $([ -f $Q/rollouts.npz ] && echo kept || echo deleted)"
}
( q5 Z0 0; q5 Z2 0; q5 Z4 0 ) & ( q5 Z1 1; q5 Z3 1 ) & wait
# 3 + 5 report
{
  echo "# v6g10s post-analysis, items 3 and 5 (completed $(date +%m-%d))"
  echo "## 3 robustness"
  $PY tools/robust_summary.py $V 0; echo; $PY tools/robust_summary.py $V 1
  echo "## 5 Q_exec (v6g10s seed-1 students), test AUROC / risk@80% (isotonic)"
  $PY - << 'EOF'
import glob, json, numpy as np
for P in ["Z0", "Z1", "Z2", "Z3", "Z4"]:
    for fr in ("0.05", "1.0"):
        r = [json.load(open(f)) for f in glob.glob(f"/root/gpufree-data/hnu_v2/runs/quality_v6g10s/{P}/frac{fr}_s*/quality_eval.json")]
        if r:
            a = [x["isotonic"]["auroc"] for x in r]; k = [100 * x["isotonic"]["risk@cov80"] for x in r]
            h = r[0]["heuristic"]["auroc"]
            print(f"L{P[1]} frac {fr}: AUROC {np.mean(a):.3f} ± {np.std(a):.3f}  risk@80 {np.mean(k):.2f}%  "
                  f"(n={len(r)}; no-filter risk {100 * r[0]['no_filter_risk']:.1f}%, heuristic AUROC {h:.3f})")
EOF
} > $N/after_${V}_items35.md 2>&1
wait
log "A complete"
# version B (+ A3 control) afterwards
setsid nohup bash run_v6g10sB.sh > /dev/null 2>&1 < /dev/null &
log "version B started"
