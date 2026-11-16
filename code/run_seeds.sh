#!/bin/bash
# Extra RL seeds (2, 3) for the five v3 latent students: train, evaluate (clean + push), then aggregate over seeds.
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
N=$H/runs/night
cd $H/code
export HNU_TB_ROOT=$H/runs/logs/rsl_rl/students_v3
ck() { cat $H/runs/expert_eval/$1/checkpoint.txt; }
CK="--ckpt-ThreeStepPoleRun $(ck ThreeStepPoleRun) --ckpt-FiveStepPoleRun $(ck FiveStepPoleRun) --ckpt-CrouchedTraverse $(ck CrouchedTraverse)"
log() { echo "[$(date +%H:%M:%S)] $*" >> $N/seeds.log; }
for S in 2 3; do
  log "seed $S train"
  for P in Z0 Z1 Z2 Z3 Z4; do
    g=0; [ $P = Z3 ] || [ $P = Z4 ] && g=1
    O=$H/runs/students_v3_s$S/$P
    [ -f $O/student_2500.pt ] || CUDA_VISIBLE_DEVICES=$g $PY distill_train.py --profile v3_${P}_s$S $CK --num-envs 4096 --iters 2500 \
      --seed $((1000 * S)) --latent-dir $H/data/z_tables_v3/$P --out $O > $N/seeds_E_${P}_s$S.log 2>&1 &
  done
  wait
  log "seed $S eval"
  i=0
  for P in Z0 Z1 Z2 Z3 Z4; do
    g=$((i % 2)); i=$((i + 1))
    ( for X in "" "--push"; do CUDA_VISIBLE_DEVICES=$g $PY eval_student.py --student $H/runs/students_v3_s$S/$P/student_2500.pt \
        --latent-dir $H/data/z_tables_v3/$P --per-clip 8 $X --out $H/runs/results_v3_s$S --tag _$P >> $N/seeds_F_${P}_s$S.log 2>&1; done ) &
  done
  wait
done
$PY tools/aggregate_seeds.py $H/runs/results_v3 $H/runs/results_v3_s2 $H/runs/results_v3_s3 > $N/tables_v3_seeds.md 2>&1
log "seeds done"
