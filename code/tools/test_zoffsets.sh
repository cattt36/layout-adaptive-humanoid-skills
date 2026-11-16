#!/bin/bash
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
cd $H/code
ck() { cat $H/runs/expert_eval/$1/checkpoint.txt; }
CK="--ckpt-ThreeStepPoleRun $(ck ThreeStepPoleRun) --ckpt-FiveStepPoleRun $(ck FiveStepPoleRun) --ckpt-CrouchedTraverse $(ck CrouchedTraverse)"
export CUDA_VISIBLE_DEVICES=1
echo "== legacy v2 student eval"
$PY eval_student.py --student $H/runs/students/Z0/student_2500.pt --latent-dir $H/data/z_tables/Z0 --per-clip 1 --out /tmp/zt --tag _legacy 2>&1 | grep -E 'success|Error' | head -4
echo "== new 4-offset student (5 iters) + eval + moe"
$PY distill_train.py --profile t $CK --num-envs 128 --iters 5 --minibatch 2048 --latent-dir $H/data/z_tables/Z0 --out /tmp/zt_new 2>&1 | grep -E '^\{|Error' | tail -1
$PY eval_student.py --student /tmp/zt_new/student_5.pt --latent-dir $H/data/z_tables/Z0 --per-clip 1 --out /tmp/zt --tag _new 2>&1 | grep -E 'success|Error' | head -2
$PY -c "import torch;c=torch.load('/tmp/zt_new/student_5.pt',weights_only=False);print('obs_dim',c['obs_dim'],'z_offsets',c['z_offsets'])"
$PY -c "import torch;c=torch.load('$H/runs/students/Z0/student_2500.pt',weights_only=False);print('legacy obs_dim',c['obs_dim'])"
