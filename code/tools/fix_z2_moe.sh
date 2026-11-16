#!/bin/bash
# Re-run the two v2 students that hit CUDA OOM (Z2 on GPU0, Z4_moe on GPU1), evaluate them, train Z2's Q_exec.
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
N=$H/runs/night
cd $H/code
export HNU_TB_ROOT=$H/runs/logs/rsl_rl/students
ck() { cat $H/runs/expert_eval/$1/checkpoint.txt; }
CK="--ckpt-ThreeStepPoleRun $(ck ThreeStepPoleRun) --ckpt-FiveStepPoleRun $(ck FiveStepPoleRun) --ckpt-CrouchedTraverse $(ck CrouchedTraverse)"
echo "[$(date +%H:%M:%S)] fix start" >> $N/fix.log
rm -rf $H/runs/students/Z2 $H/runs/students/Z4_moe
CUDA_VISIBLE_DEVICES=0 $PY distill_train.py --profile Z2 $CK --num-envs 4096 --iters 2500 --latent-dir $H/data/z_tables/Z2 \
  --out $H/runs/students/Z2 > $N/fix_Z2.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY distill_train.py --profile Z4_moe $CK --num-envs 4096 --iters 2500 --arch moe --latent-dir $H/data/z_tables/Z4 \
  --out $H/runs/students/Z4_moe > $N/fix_moe.log 2>&1 &
wait
for X in "" "--push"; do
  CUDA_VISIBLE_DEVICES=0 $PY eval_student.py --student $H/runs/students/Z2/student_2500.pt --latent-dir $H/data/z_tables/Z2 \
    --per-clip 8 $X --out $H/runs/results --tag _Z2 >> $N/fix_eval.log 2>&1
  CUDA_VISIBLE_DEVICES=1 $PY eval_student.py --student $H/runs/students/Z4_moe/student_2500.pt --latent-dir $H/data/z_tables/Z4 \
    --per-clip 8 $X --out $H/runs/results --tag _Z4_moe >> $N/fix_eval.log 2>&1
done
mkdir -p $H/runs/quality/Z2
CUDA_VISIBLE_DEVICES=0 $PY stage5_quality/quality.py collect --policy student --student $H/runs/students/Z2/student_2500.pt \
  --latent-dir $H/data/z_tables/Z2 --num-envs 2048 --steps 1500 --out $H/runs/quality/Z2/rollouts.npz > $N/fix_G_Z2.log 2>&1
CUDA_VISIBLE_DEVICES=0 $PY stage5_quality/quality.py train --data $H/runs/quality/Z2/rollouts.npz \
  --heuristic-dir $H/data/heuristic --out $H/runs/quality/Z2 >> $N/fix_G_Z2.log 2>&1
echo "[$(date +%H:%M:%S)] fix done" >> $N/fix.log
