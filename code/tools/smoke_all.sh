#!/bin/bash
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
cd $H/code
L=$H/runs/logs/rsl_rl
C1=$(ls -t $L/hnu_track_ThreeStepPoleRun/*_v1/model_*.pt | head -1)
C2=$(ls -t $L/hnu_track_FiveStepPoleRun/*_v1/model_*.pt | head -1)
C3=$(ls -t $L/hnu_track_CrouchedTraverse/*_v1/model_*.pt | head -1)
CK="--ckpt-ThreeStepPoleRun $C1 --ckpt-FiveStepPoleRun $C2 --ckpt-CrouchedTraverse $C3"
export CUDA_VISIBLE_DEVICES=1
$PY stage5_quality/make_heuristic.py $H/data/ref_v1 $H/data/heuristic 2>&1 | tail -1
$PY stage5_quality/quality.py collect --policy teacher $CK --latent-dir $H/data/z_smoke/Z0 --num-envs 256 --steps 300 --out $H/runs/q_smoke.npz 2>&1 | tail -1
$PY stage5_quality/quality.py train --data $H/runs/q_smoke.npz --heuristic-dir $H/data/heuristic --epochs 5 --out $H/runs/q_smoke 2>&1 | tail -1
for G in student_nolatent; do
  $PY distill_train.py --profile smoke --obs-group $G --latent-dir $H/data/z_smoke/Z0 $CK --num-envs 256 --iters 11 --minibatch 4096 --out $H/runs/distill_smoke_$G 2>&1 | grep '^{' | tail -1
done
$PY distill_train.py --profile smoke --arch moe --latent-dir $H/data/z_smoke/Z0 $CK --num-envs 256 --iters 11 --minibatch 4096 --out $H/runs/distill_smoke_moe 2>&1 | grep '^{' | tail -1
$PY eval_student.py --student $H/runs/distill_smoke_moe/student_11.pt --latent-dir $H/data/z_smoke/Z0 --per-clip 1 --out $H/runs/distill_smoke_moe 2>&1 | grep success | head -3
