#!/bin/bash
# Capture-video bridge layout: 4 posts (lab z from the video, refined by the motion data), thin posts (r 0.05 m),
# round log beam ~0.15 m.  Reference clearances, faithful heights and with the beam lifted to the reference.
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
cd $H
export HNU_POSTS_Z=${1:--2.55,-1.42,0.45,1.95} HNU_POST_R=${2:-0.05} HNU_BEAM=${3:-0.15}
rm -rf data/bridge_video data/bridge_video_lift
$PY code/tools/make_bridge_usd.py data/ref_v1/CrouchedTraverse data/scene/CrouchedTraverse.csv data/bridge_video 2>&1 | grep -v -i warn | head -n 8
HNU_BRIDGE_LIFT=1 $PY code/tools/make_bridge_usd.py data/ref_v1/CrouchedTraverse data/scene/CrouchedTraverse.csv data/bridge_video_lift 2>&1 \
  | grep -v -i warn > runs/logs/make_bridge_video_lift.log
head -n 8 runs/logs/make_bridge_video_lift.log
