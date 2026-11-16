#!/bin/bash
# Re-run v3 students Z0 (GPU0) and Z2 (GPU1) that hit CUDA OOM, then evaluate, Q_exec, and rebuild the tables.
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
N=$H/runs/night
cd $H/code
export HNU_TB_ROOT=$H/runs/logs/rsl_rl/students_v3
ck() { cat $H/runs/expert_eval/$1/checkpoint.txt; }
CK="--ckpt-ThreeStepPoleRun $(ck ThreeStepPoleRun) --ckpt-FiveStepPoleRun $(ck FiveStepPoleRun) --ckpt-CrouchedTraverse $(ck CrouchedTraverse)"
echo "[$(date +%H:%M:%S)] fix_v3 start" >> $N/v3.log
g=0
for P in Z0 Z2; do
  ( rm -rf $H/runs/students_v3/$P
    CUDA_VISIBLE_DEVICES=$g $PY distill_train.py --profile v3_$P $CK --num-envs 4096 --iters 2500 \
      --latent-dir $H/data/z_tables_v3/$P --out $H/runs/students_v3/$P > $N/v3_E_$P.log 2>&1
    for X in "" "--push"; do
      CUDA_VISIBLE_DEVICES=$g $PY eval_student.py --student $H/runs/students_v3/$P/student_2500.pt --latent-dir $H/data/z_tables_v3/$P \
        --per-clip 8 $X --out $H/runs/results_v3 --tag _$P >> $N/v3_F_$P.log 2>&1
    done
    mkdir -p $H/runs/quality_v3/$P
    CUDA_VISIBLE_DEVICES=$g $PY stage5_quality/quality.py collect --policy student --student $H/runs/students_v3/$P/student_2500.pt \
      --latent-dir $H/data/z_tables_v3/$P --num-envs 2048 --steps 1500 --out $H/runs/quality_v3/$P/rollouts.npz > $N/v3_G_$P.log 2>&1
    CUDA_VISIBLE_DEVICES=$g $PY stage5_quality/quality.py train --data $H/runs/quality_v3/$P/rollouts.npz \
      --heuristic-dir $H/data/heuristic --out $H/runs/quality_v3/$P >> $N/v3_G_$P.log 2>&1 ) &
  g=1
done
wait
$PY tools/make_tables.py $H/runs/results_v3 $H/runs/quality_v3 > $N/tables_v3.md 2>&1
$PY tools/make_tables.py $H/runs/results $H/runs/quality > $N/tables_v2.md 2>&1
echo "[$(date +%H:%M:%S)] fix_v3 done" >> $N/v3.log
