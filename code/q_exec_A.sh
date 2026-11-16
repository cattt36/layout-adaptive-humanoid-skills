#!/bin/bash
# Item 5 (Q_exec few-shot) for v6g10s, memory-safe (container limit 100 GB): float16 data, ONE training at a time
# (GPU 0) while the next level's rollouts are collected on GPU 1.  All 30 trainings (L0-L4 x frac 0.05 / 1.0 x 3 seeds)
# run with the same code; raw rollouts of a level are deleted after its 6 trainings.  Then the items 3 + 5 report.
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
N=$H/runs/night
V=v6g10s
cd $H/code
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HNU_BRIDGE_ROOT=$H/data/ref_bridge_video_lift HNU_STAKE_H=0.10 \
  HNU_GEOM_W1=$H/data/ref_stakes_s93_h10 HNU_GEOM_W2=$H/data/ref_stakes_s83_h10 HNU_STAKE_TOL=0.17 HNU_DR_LEVEL=1 \
  HNU_DR_LAG_SYNC=1 HNU_DR_LAG_SENSORS=1 PYTHONPATH=$H/code
log() { echo "[$(date +%H:%M:%S)] $*" >> $N/q_exec_A.log; }
collect() {  # P gpu
  local Q=$H/runs/quality_$V/$1
  mkdir -p $Q
  [ -f $Q/rollouts.npz ] || CUDA_VISIBLE_DEVICES=$2 $PY stage5_quality/quality.py collect --policy student \
    --student $H/runs/students_${V}_d2_s1/$1/student_2000.pt --latent-dir $H/data/z_tables_v5/$1 --num-envs 2048 \
    --steps 1500 --out $Q/rollouts.npz > $Q/collect.log 2>&1
  log "collect $1: $([ -f $Q/rollouts.npz ] && echo ok || echo FAILED)"
}
trainall() {  # P
  local Q=$H/runs/quality_$V/$1
  rm -rf $Q/frac*_s*  # earlier runs (older code / killed): redo all six with the same code
  for FR in 0.05 1.0; do for S in 0 1 2; do
    CUDA_VISIBLE_DEVICES=0 $PY stage5_quality/quality.py train --data $Q/rollouts.npz --heuristic-dir $H/data/heuristic \
      --train-frac $FR --seed $S --out $Q/frac${FR}_s$S > $Q/frac${FR}_s$S.log 2>&1
  done; done
  local n=$(ls $Q/frac*_s*/quality_eval.json 2>/dev/null | wc -l)
  [ "$n" = 6 ] && rm -f $Q/rollouts.npz
  log "train $1: $n/6 done, rollouts $([ -f $Q/rollouts.npz ] && echo kept || echo deleted)"
}
log "start"
collect Z3 1 &
trainall Z0; trainall Z1
wait
collect Z4 1 &
trainall Z2; trainall Z3
wait
trainall Z4
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
log "finished"
