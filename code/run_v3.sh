#!/bin/bash
# v3 chain (skill-level transfer terms enabled for every supervision level), started after run_night.sh finished.
set -u
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
N=$H/runs/night
cd $H/code
export HNU_TB_ROOT=$H/runs/logs/rsl_rl/students_v3
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $N/v3.log; }
is_done() { [ -f $N/v3_$1.done ]; }
done_mark() { touch $N/v3_$1.done; }
while ! grep -q 'night pipeline finished' $N/night.log 2>/dev/null; do sleep 60; done
ck() { cat $H/runs/expert_eval/$1/checkpoint.txt; }
CK="--ckpt-ThreeStepPoleRun $(ck ThreeStepPoleRun) --ckpt-FiveStepPoleRun $(ck FiveStepPoleRun) --ckpt-CrouchedTraverse $(ck CrouchedTraverse)"
ITERS=2500

if ! is_done C; then
  log "v3 C: latents (3 seeds, GPU)"
  OUT=$H/runs/latent_v3; mkdir -p $OUT
  for S in 42 43 44; do
    g=1; [ $S != 42 ] && g=0
    for P in Z0 Z1 Z2 Z3 Z4; do
      CUDA_VISIBLE_DEVICES=$g ANNOT=annot_v2 bash stage4_latent/train_latent.sh $P $OUT/seed$S --epochs 200 --seed $S \
        --lambda-skill-metric 0.10 --lambda-skill-atoms 0.08 > $OUT/train_${P}_seed$S.log 2>&1 &
    done
  done
  wait; done_mark C
fi
if ! is_done D; then
  log "v3 D: z tables"
  for P in Z0 Z1 Z2 Z3 Z4; do
    ANNOT=annot_v2 PYTHONPATH=$H/data/data_code CUDA_VISIBLE_DEVICES=1 $PY stage4_latent/export_z_tables.py --profile $P \
      --latent-root $H/runs/latent_v3/seed42 --motion-root $H/data/ref_v1 --out $H/data/z_tables_v3/$P > $N/v3_D_$P.log 2>&1
  done
  done_mark D
fi
if ! is_done E; then
  log "v3 E: students"
  for P in Z0 Z1 Z2 Z3 Z4; do
    g=0; [ $P = Z3 ] || [ $P = Z4 ] && g=1
    [ -f $H/runs/students_v3/$P/student_$ITERS.pt ] || \
      CUDA_VISIBLE_DEVICES=$g $PY distill_train.py --profile v3_$P $CK --num-envs 4096 --iters $ITERS \
        --latent-dir $H/data/z_tables_v3/$P --out $H/runs/students_v3/$P > $N/v3_E_$P.log 2>&1 &
  done
  wait; done_mark E
fi
if ! is_done F; then
  log "v3 F: evaluation"
  mkdir -p $H/runs/results_v3
  i=0
  for P in Z0 Z1 Z2 Z3 Z4; do
    g=$((i % 2)); i=$((i + 1))
    ( for X in "" "--push"; do CUDA_VISIBLE_DEVICES=$g $PY eval_student.py --student $H/runs/students_v3/$P/student_$ITERS.pt \
        --latent-dir $H/data/z_tables_v3/$P --per-clip 8 $X --out $H/runs/results_v3 --tag _$P >> $N/v3_F_$P.log 2>&1; done ) &
  done
  wait
  cp $H/runs/results/eval_teacher_teacher*.json $H/runs/results_v3/ 2>/dev/null
  cp $H/runs/results/eval_student_ref*.json $H/runs/results/eval_student_nolatent*.json $H/runs/results_v3/ 2>/dev/null
  done_mark F
fi
if ! is_done G; then
  log "v3 G: Q_exec"
  i=0
  for P in Z0 Z1 Z2 Z3 Z4; do
    g=$((i % 2)); i=$((i + 1)); mkdir -p $H/runs/quality_v3/$P
    ( CUDA_VISIBLE_DEVICES=$g $PY stage5_quality/quality.py collect --policy student --student $H/runs/students_v3/$P/student_$ITERS.pt \
        --latent-dir $H/data/z_tables_v3/$P --num-envs 2048 --steps 1500 --out $H/runs/quality_v3/$P/rollouts.npz > $N/v3_G_$P.log 2>&1
      CUDA_VISIBLE_DEVICES=$g $PY stage5_quality/quality.py train --data $H/runs/quality_v3/$P/rollouts.npz \
        --heuristic-dir $H/data/heuristic --out $H/runs/quality_v3/$P >> $N/v3_G_$P.log 2>&1 ) &
    [ $((i % 2)) = 0 ] && wait
  done
  wait; done_mark G
fi
if ! is_done H; then
  log "v3 H: offline 8.1 (all seeds, v2 and v3)"
  for V in latent_v2 latent_v3; do for S in 42 43 44; do
    R=$(bash tools/setup_paper_root.sh runs/$V/seed$S data/annot_v2 2>/dev/null | tail -1)
    bash tools/run_offline_81.sh $R > $N/v3_H_${V}_$S.log 2>&1
  done; done
  $PY tools/make_tables.py $H/runs/results $H/runs/quality > $N/tables_v2.md 2>&1
  $PY tools/make_tables.py $H/runs/results_v3 $H/runs/quality_v3 > $N/tables_v3.md 2>&1
  done_mark H
fi
log "v3 chain finished"
