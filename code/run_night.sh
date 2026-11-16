#!/bin/bash
# Overnight pipeline after the three experts are trained (stage 3 -> 5).  Each step writes a marker in
# $H/runs/night/ so the script can be re-run and resumes after the last finished step.
set -u
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
N=$H/runs/night
mkdir -p $N
cd $H/code
export HNU_TB_ROOT=$H/runs/logs/rsl_rl/students
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $N/night.log; }
done_mark() { touch $N/$1.done; }
is_done() { [ -f $N/$1.done ]; }
ck() { cat $H/runs/expert_eval/$1/checkpoint.txt; }

# A. expert evaluation + rollouts
if ! is_done A; then
  log "A: expert eval"
  [ -f $H/runs/expert_eval/ThreeStepPoleRun/eval_push.log ] || bash run_post_expert.sh ThreeStepPoleRun 0 v1 > $N/A_w1.log 2>&1 &
  [ -f $H/runs/expert_eval/FiveStepPoleRun/eval_push.log ] || bash run_post_expert.sh FiveStepPoleRun 1 v1 > $N/A_w2.log 2>&1 &
  [ -f $H/runs/expert_eval/CrouchedTraverse/eval_push.log ] || bash run_post_expert.sh CrouchedTraverse 1 v1 > $N/A_dm.log 2>&1 &
  wait; done_mark A
fi
CK="--ckpt-ThreeStepPoleRun $(ck ThreeStepPoleRun) --ckpt-FiveStepPoleRun $(ck FiveStepPoleRun) --ckpt-CrouchedTraverse $(ck CrouchedTraverse)"

# B. annotations with execution statistics
if ! is_done B; then
  log "B: annotations v2"
  for T in ThreeStepPoleRun FiveStepPoleRun CrouchedTraverse; do
    $PY stage4_annotations/regen_robot_metrics.py --task $T --ref $H/data/ref_v1/$T \
      --old-annotations $H/data/llm_annotation_old/$T --rollouts $H/runs/expert_eval/$T \
      --data-code $H/data/data_code --out $H/data/annot_v2 > $N/B_$T.log 2>&1 &
  done
  wait; done_mark B
fi

# C. latents Z0-Z4 (3 seeds; seed 42 feeds the students)
if ! is_done C; then
  log "C: latents"
  ANNOT=annot_v2 LATENT_RUN=latent_v2 GPU=1 SEEDS="42" bash stage4_latent/train_all.sh > $N/C.log 2>&1
  done_mark C
fi

# D. z tables
if ! is_done D; then
  log "D: z tables"
  for P in Z0 Z1 Z2 Z3 Z4; do
    ANNOT=annot_v2 PYTHONPATH=$H/data/data_code CUDA_VISIBLE_DEVICES=1 $PY stage4_latent/export_z_tables.py --profile $P \
      --latent-root $H/runs/latent_v2/seed42 --motion-root $H/data/ref_v1 --out $H/data/z_tables/$P > $N/D_$P.log 2>&1
  done
  done_mark D
fi

# E. students (8 runs, 4 per GPU)
ITERS=${ITERS:-2500}
student() {  # name gpu extra...
  local name=$1 g=$2; shift 2
  [ -f $H/runs/students/$name/student_$ITERS.pt ] && return
  CUDA_VISIBLE_DEVICES=$g $PY distill_train.py --profile $name $CK --num-envs 4096 --iters $ITERS \
    --out $H/runs/students/$name "$@" > $N/E_$name.log 2>&1
}
if ! is_done E; then
  log "E: students"
  student Z0 0 --latent-dir $H/data/z_tables/Z0 &
  student Z1 0 --latent-dir $H/data/z_tables/Z1 &
  student Z2 0 --latent-dir $H/data/z_tables/Z2 &
  student ref 0 --latent-dir $H/data/z_tables/Z4 --obs-group student_ref &
  student Z3 1 --latent-dir $H/data/z_tables/Z3 &
  student Z4 1 --latent-dir $H/data/z_tables/Z4 &
  student Z4_moe 1 --latent-dir $H/data/z_tables/Z4 --arch moe &
  student nolatent 1 --latent-dir $H/data/z_tables/Z4 --obs-group student_nolatent &
  wait; done_mark E
fi

# F. evaluation (clean + push), teacher baseline
if ! is_done F; then
  log "F: evaluation"
  mkdir -p $H/runs/results
  CUDA_VISIBLE_DEVICES=0 $PY eval_student.py --teacher $CK --latent-dir $H/data/z_tables/Z4 --per-clip 8 --out $H/runs/results --tag _teacher > $N/F_teacher.log 2>&1
  CUDA_VISIBLE_DEVICES=0 $PY eval_student.py --teacher $CK --latent-dir $H/data/z_tables/Z4 --per-clip 8 --push --out $H/runs/results --tag _teacher > $N/F_teacher_push.log 2>&1
  i=0
  for S in Z0 Z1 Z2 Z3 Z4 Z4_moe ref nolatent; do
    Z=$S; [ $S = Z4_moe ] || [ $S = ref ] || [ $S = nolatent ] && Z=Z4
    g=$((i % 2)); i=$((i + 1))
    ( CUDA_VISIBLE_DEVICES=$g $PY eval_student.py --student $H/runs/students/$S/student_$ITERS.pt --latent-dir $H/data/z_tables/$Z \
        --per-clip 8 --out $H/runs/results --tag _$S > $N/F_$S.log 2>&1
      CUDA_VISIBLE_DEVICES=$g $PY eval_student.py --student $H/runs/students/$S/student_$ITERS.pt --latent-dir $H/data/z_tables/$Z \
        --per-clip 8 --push --out $H/runs/results --tag _$S > $N/F_${S}_push.log 2>&1 ) &
    [ $((i % 4)) = 0 ] && wait
  done
  wait; done_mark F
fi

# G. executability Q_exec per latent profile (rollouts of the matching student)
if ! is_done G; then
  log "G: Q_exec"
  i=0
  for P in Z0 Z1 Z2 Z3 Z4; do
    g=$((i % 2)); i=$((i + 1))
    mkdir -p $H/runs/quality/$P
    ( CUDA_VISIBLE_DEVICES=$g $PY stage5_quality/quality.py collect --policy student --student $H/runs/students/$P/student_$ITERS.pt \
        --latent-dir $H/data/z_tables/$P --num-envs 2048 --steps 1500 --out $H/runs/quality/$P/rollouts.npz > $N/G_$P.log 2>&1
      CUDA_VISIBLE_DEVICES=$g $PY stage5_quality/quality.py train --data $H/runs/quality/$P/rollouts.npz \
        --heuristic-dir $H/data/heuristic --out $H/runs/quality/$P >> $N/G_$P.log 2>&1 ) &
    [ $((i % 2)) = 0 ] && wait
  done
  wait; done_mark G
fi
log "night pipeline finished"
