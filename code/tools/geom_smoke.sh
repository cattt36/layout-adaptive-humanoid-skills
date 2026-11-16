#!/bin/bash
# Smoke test of the unified geometry scene: teacher bank (stake experts gate_h06 + bridge expert) evaluated in the
# distillation env, nominal; then a 3-iteration DAgger run.
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
cd $H/code
export HF_HUB_OFFLINE=1 HNU_BRIDGE_ROOT=$H/data/ref_bridge_video_lift HNU_STAKE_H=0.06 HNU_STAKE_TOL=0.17
CK="--ckpt-ThreeStepPoleRun $(cat $H/runs/stakes_eval/gate_h06/ThreeStepPoleRun/checkpoint.txt) --ckpt-FiveStepPoleRun $(cat $H/runs/stakes_eval/gate_h06/FiveStepPoleRun/checkpoint.txt) --ckpt-CrouchedTraverse $(cat $H/runs/bridge_eval/bridge_video_lift_perc/checkpoint.txt)"
CUDA_VISIBLE_DEVICES=0 $PY eval_student.py --teacher $CK --scene geom --z-offsets 0,5,10,15,20,25,30,40,50 \
  --latent-dir $H/data/z_tables_v5/Z4 --per-clip 4 --out /tmp/geom_smoke > /tmp/geom_smoke.log 2>&1
grep -a -e '/all' -e Error -e Traceback /tmp/geom_smoke.log | head
CUDA_VISIBLE_DEVICES=0 $PY distill_train.py --profile smoke $CK --num-envs 64 --iters 3 --scene geom --z-offsets 0,5,10,15,20,25,30,40,50 \
  --hidden 1024,512,256 --hist 20 --lin-vel --latent-dir $H/data/z_tables_v5/Z4 --out /tmp/geom_smoke_st 2>&1 | grep -e '"it"' -e Error | tail -n 2
