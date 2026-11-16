#!/bin/bash
# Re-run the L4 latent-gated MoE student (OOM in the first pass) once the extra latent seeds have finished,
# then evaluate it (clean + push) and train its Q_exec like the others.
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
cd $H/code
export HNU_TB_ROOT=$H/runs/logs/rsl_rl/students
while pgrep -f 'motion_laten[t]_z.py' > /dev/null; do sleep 60; done
ck() { cat $H/runs/expert_eval/$1/checkpoint.txt; }
CK="--ckpt-ThreeStepPoleRun $(ck ThreeStepPoleRun) --ckpt-FiveStepPoleRun $(ck FiveStepPoleRun) --ckpt-CrouchedTraverse $(ck CrouchedTraverse)"
rm -rf $H/runs/students/Z4_moe
echo "[$(date +%H:%M:%S)] moe start" >> $H/runs/night/moe.log
CUDA_VISIBLE_DEVICES=0 $PY distill_train.py --profile Z4_moe $CK --num-envs 4096 --iters 2500 --arch moe \
  --latent-dir $H/data/z_tables/Z4 --out $H/runs/students/Z4_moe >> $H/runs/night/moe.log 2>&1
for P in "" "--push"; do
  CUDA_VISIBLE_DEVICES=0 $PY eval_student.py --student $H/runs/students/Z4_moe/student_2500.pt --latent-dir $H/data/z_tables/Z4 \
    --per-clip 8 $P --out $H/runs/results --tag _Z4_moe >> $H/runs/night/moe.log 2>&1
done
echo "[$(date +%H:%M:%S)] moe done" >> $H/runs/night/moe.log
