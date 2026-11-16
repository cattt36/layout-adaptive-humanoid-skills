#!/bin/bash
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
D=$(ls -d $H/runs/logs/rsl_rl/hnu_track_ThreeStepPoleRun/*_v1 | head -1)
RUN=$(basename $D)
cd $H/runs
CUDA_VISIBLE_DEVICES=1 nohup $PY $H/code/train_hnu.py HNU-Track-ThreeStepPoleRun --env.scene.num-envs 4096 \
  --env.commands.motion.motion-file $H/data/ref_v1/ThreeStepPoleRun --agent.max-iterations 4000 --agent.run-name v1 \
  --agent.resume True --agent.load-run $RUN --agent.load-checkpoint model_6000.pt \
  > $H/runs/logs/ThreeStepPoleRun_v1_resume.log 2>&1 < /dev/null &
echo resumed from $RUN
