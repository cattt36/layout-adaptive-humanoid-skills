#!/bin/bash
# v4c = v4b with human semantics routed ONLY to z_sem (lambda_human_skill_atoms = 0): embodiment-matched supervision
# (robot metrics/atoms) shapes the control latent z_skill, human-derived supervision shapes the semantic latent z_sem.
# Z0-Z3 own no human labels, so they are identical to v4b; only Z4 is retrained (3 seeds) + its student, eval, Q_exec.
set -u
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
N=$H/runs/night
cd $H/code
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export HNU_TB_ROOT=$H/runs/logs/rsl_rl/students_v4c
log() { echo "[$(date +%H:%M:%S)] $*" >> $N/v4c.log; }
while ! grep -q 'v4b chain finished' $N/v4b.log 2>/dev/null; do sleep 60; done
log "C: Z4 latents, 3 seeds"
mkdir -p $H/runs/latent_v4c
for S in 42 43 44; do
  g=$((S % 2))
  [ -f $H/runs/latent_v4c/seed$S/Z4/train_latents.npz ] && continue
  CUDA_VISIBLE_DEVICES=$g ANNOT=annot_v2 nice -n 10 bash stage4_latent/train_latent.sh Z4 $H/runs/latent_v4c/seed$S --epochs 200 \
    --seed $S --lambda-skill-metric 0.10 --lambda-skill-atoms 0.08 --lambda-human-skill-atoms 0.0 \
    --lambda-human-sem-atoms 0.10 --text-encoder minilm --text-dim 384 --query-augmentation 0.5 --lambda-query-task 0.2 \
    > $H/runs/latent_v4c/train_Z4_seed$S.log 2>&1 &
done
mkdir -p $H/runs/latent_v4c; wait
for S in 42 43 44; do for P in Z0 Z1 Z2 Z3; do  # identical by construction -> link the v4b runs
  mkdir -p $H/runs/latent_v4c/seed$S; [ -e $H/runs/latent_v4c/seed$S/$P ] || ln -s $H/runs/latent_v4b/seed$S/$P $H/runs/latent_v4c/seed$S/$P
done; done
log "D: z table Z4"
ANNOT=annot_v2 PYTHONPATH=$H/data/data_code CUDA_VISIBLE_DEVICES=1 $PY stage4_latent/export_z_tables.py --profile Z4 \
  --latent-root $H/runs/latent_v4c/seed42 --motion-root $H/data/ref_v1 --out $H/data/z_tables_v4c/Z4 > $N/v4c_D.log 2>&1
log "E: Z4 student"
ck() { cat $H/runs/expert_eval/$1/checkpoint.txt; }
CK="--ckpt-ThreeStepPoleRun $(ck ThreeStepPoleRun) --ckpt-FiveStepPoleRun $(ck FiveStepPoleRun) --ckpt-CrouchedTraverse $(ck CrouchedTraverse)"
CUDA_VISIBLE_DEVICES=0 $PY distill_train.py --profile v4c_Z4 $CK --num-envs 4096 --iters 2500 \
  --latent-dir $H/data/z_tables_v4c/Z4 --out $H/runs/students_v4c/Z4 > $N/v4c_E.log 2>&1
log "F: eval"
mkdir -p $H/runs/results_v4c
for X in "" "--push"; do
  CUDA_VISIBLE_DEVICES=0 $PY eval_student.py --student $H/runs/students_v4c/Z4/student_2500.pt --latent-dir $H/data/z_tables_v4c/Z4 \
    --per-clip 8 $X --out $H/runs/results_v4c --tag _Z4 >> $N/v4c_F.log 2>&1
done
cp $H/runs/results_v4b/eval_*Z[0-3]*.json $H/runs/results_v4b/eval_teacher*.json $H/runs/results_v4b/eval_student_ref*.json \
   $H/runs/results_v4b/eval_student_nolatent*.json $H/runs/results_v4c/
log "G: Q_exec Z4"
mkdir -p $H/runs/quality_v4c/Z4
CUDA_VISIBLE_DEVICES=0 $PY stage5_quality/quality.py collect --policy student --student $H/runs/students_v4c/Z4/student_2500.pt \
  --latent-dir $H/data/z_tables_v4c/Z4 --num-envs 2048 --steps 1500 --out $H/runs/quality_v4c/Z4/rollouts.npz > $N/v4c_G.log 2>&1
CUDA_VISIBLE_DEVICES=0 $PY stage5_quality/quality.py train --data $H/runs/quality_v4c/Z4/rollouts.npz \
  --heuristic-dir $H/data/heuristic --out $H/runs/quality_v4c/Z4 >> $N/v4c_G.log 2>&1
for P in Z0 Z1 Z2 Z3; do [ -e $H/runs/quality_v4c/$P ] || ln -s $H/runs/quality_v4b/$P $H/runs/quality_v4c/$P; done
log "H: language + tables"
PYTHONPATH=$H/data/data_code $PY tools/eval_language.py results/router_ood_benchmark_v1.json $H/runs/latent_v4b $H/runs/latent_v4c > $N/language_v4bc.md 2>&1
$PY tools/make_tables.py $H/runs/results_v4c $H/runs/quality_v4c > $N/tables_v4c.md 2>&1
log "v4c chain finished"
