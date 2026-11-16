#!/bin/bash
# v4b chain: latents (human-atom head + sentence encoder + query augmentation) -> z tables -> students -> eval ->
# Q_exec -> offline 8.1 -> language routing -> tables.  Each step leaves a marker in runs/night/v4b_<step>.done.
set -u
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
N=$H/runs/night
cd $H/code
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HNU_TB_ROOT=$H/runs/logs/rsl_rl/students_v4b
log() { echo "[$(date +%H:%M:%S)] $*" >> $N/v4b.log; }
is_done() { [ -f $N/v4b_$1.done ]; }
done_mark() { touch $N/v4b_$1.done; }
ck() { cat $H/runs/expert_eval/$1/checkpoint.txt; }
CK="--ckpt-ThreeStepPoleRun $(ck ThreeStepPoleRun) --ckpt-FiveStepPoleRun $(ck FiveStepPoleRun) --ckpt-CrouchedTraverse $(ck CrouchedTraverse)"
student() {  # P gpu
  local P=$1 g=$2
  [ -f $H/runs/students_v4b/$P/student_2500.pt ] && return
  CUDA_VISIBLE_DEVICES=$g $PY distill_train.py --profile v4b_$P $CK --num-envs 4096 --iters 2500 \
    --latent-dir $H/data/z_tables_v4b/$P --out $H/runs/students_v4b/$P > $N/v4b_E_$P.log 2>&1
}

if ! is_done C42; then
  log "C: latents seed 42 (GPU1)"; SEEDS=42 GPU=1 bash stage4_latent/train_all_v4b.sh > $N/v4b_C42.log 2>&1
  for P in Z0 Z1 Z2 Z3 Z4; do  # retry (one at a time) any profile that failed, e.g. CUDA OOM at start-up
    [ -f $H/runs/latent_v4b/seed42/$P/train_latents.npz ] && continue
    log "C: retry $P"
    CUDA_VISIBLE_DEVICES=1 ANNOT=annot_v2 OMP_NUM_THREADS=4 nice -n 15 bash stage4_latent/train_latent.sh $P $H/runs/latent_v4b/seed42 \
      --epochs 200 --seed 42 --lambda-skill-metric 0.10 --lambda-skill-atoms 0.08 --lambda-human-skill-atoms 0.10 \
      --lambda-human-sem-atoms 0.10 --text-encoder minilm --text-dim 384 --query-augmentation 0.5 --lambda-query-task 0.2 \
      > $H/runs/latent_v4b/train_${P}_seed42_retry.log 2>&1
  done
  done_mark C42
fi
if ! is_done D; then
  log "D: z tables"
  for P in Z0 Z1 Z2 Z3 Z4; do
    ANNOT=annot_v2 PYTHONPATH=$H/data/data_code CUDA_VISIBLE_DEVICES=1 $PY stage4_latent/export_z_tables.py --profile $P \
      --latent-root $H/runs/latent_v4b/seed42 --motion-root $H/data/ref_v1 --out $H/data/z_tables_v4b/$P > $N/v4b_D_$P.log 2>&1
  done
  done_mark D
fi
log "waiting for the extra v3 RL seeds to finish"
while ! grep -q 'seeds done' $N/seeds.log 2>/dev/null; do sleep 60; done
if ! is_done E; then
  log "E: latents seeds 43/44 (GPU0) + students L0-L2 (GPU1)"
  ( SEEDS="43 44" GPU=0 PARALLEL_SEEDS=1 bash stage4_latent/train_all_v4b.sh > $N/v4b_C4344.log 2>&1
    student Z3 0 & student Z4 0 & wait ) &
  student Z0 1 & student Z1 1 & student Z2 1 &
  wait; done_mark E
fi
if ! is_done F; then
  log "F: evaluation"
  mkdir -p $H/runs/results_v4b
  i=0
  for P in Z0 Z1 Z2 Z3 Z4; do
    g=$((i % 2)); i=$((i + 1))
    ( for X in "" "--push"; do CUDA_VISIBLE_DEVICES=$g $PY eval_student.py --student $H/runs/students_v4b/$P/student_2500.pt \
        --latent-dir $H/data/z_tables_v4b/$P --per-clip 8 $X --out $H/runs/results_v4b --tag _$P >> $N/v4b_F_$P.log 2>&1; done ) &
  done
  wait
  cp $H/runs/results/eval_teacher_teacher*.json $H/runs/results/eval_student_ref*.json $H/runs/results/eval_student_nolatent*.json $H/runs/results_v4b/
  done_mark F
fi
if ! is_done G; then
  log "G: Q_exec"
  i=0
  for P in Z0 Z1 Z2 Z3 Z4; do
    g=$((i % 2)); i=$((i + 1)); mkdir -p $H/runs/quality_v4b/$P
    ( CUDA_VISIBLE_DEVICES=$g $PY stage5_quality/quality.py collect --policy student --student $H/runs/students_v4b/$P/student_2500.pt \
        --latent-dir $H/data/z_tables_v4b/$P --num-envs 2048 --steps 1500 --out $H/runs/quality_v4b/$P/rollouts.npz > $N/v4b_G_$P.log 2>&1
      CUDA_VISIBLE_DEVICES=$g $PY stage5_quality/quality.py train --data $H/runs/quality_v4b/$P/rollouts.npz \
        --heuristic-dir $H/data/heuristic --out $H/runs/quality_v4b/$P >> $N/v4b_G_$P.log 2>&1 ) &
    [ $((i % 2)) = 0 ] && wait
  done
  wait; done_mark G
fi
if ! is_done H; then
  log "H: offline 8.1 + language routing + tables"
  export HNU_TEXT_ENCODER=minilm
  for S in 42 43 44; do
    R=$(bash tools/setup_paper_root.sh runs/latent_v4b/seed$S data/annot_v2 2>/dev/null | tail -1)
    bash tools/run_offline_81.sh $R > $N/v4b_H_$S.log 2>&1
  done
  unset HNU_TEXT_ENCODER
  $PY tools/aggregate_81.py $H/paper_root > $N/offline_81_all.md 2>&1
  PYTHONPATH=$H/data/data_code $PY tools/eval_language.py results/router_ood_benchmark_v1.json $H/runs/latent_v2 $H/runs/latent_v3 $H/runs/latent_v4b > $N/language_all.md 2>&1
  $PY tools/make_tables.py $H/runs/results_v4b $H/runs/quality_v4b > $N/tables_v4b.md 2>&1
  done_mark H
fi
log "v4b chain finished"
