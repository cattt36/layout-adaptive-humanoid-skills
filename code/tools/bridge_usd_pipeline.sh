#!/bin/bash
# DuMuQiao.usd + boundary markers: two scene variants, each (1) zero-shot eval of the marker-bridge expert (bridge1),
# (2) fine-tune from bridge1, (3) eval clean / push+noise.
#   usd_faithful : USD parts at lab-faithful (scaled) heights incl. the 5 support posts
#   usd_lift     : same, plank (+ posts) raised to the retargeted reference's clearance
# usage: bridge_usd_pipeline.sh <variant> <gpu> [iters]
V=$1; G=$2; ITERS=${3:-6000}
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
T=CrouchedTraverse
cd $H/code
J=$H/data/bridge_$V
if [ ! -d $J ]; then
  LIFT=0; [ $V = usd_lift ] && LIFT=1
  HNU_BRIDGE_LIFT=$LIFT $PY tools/make_bridge_usd.py $H/data/ref_v1/$T $H/data/scene/$T.csv $J > $H/runs/logs/make_bridge_$V.log 2>&1
fi
R=$H/data/ref_bridge_$V
mkdir -p $R/$T
for j in $J/*_bridge.json; do
  c=$(basename $j _bridge.json)
  ok=$($PY -c "import json; print(int(json.load(open('$j'))['fit_err_m'] < 0.1))")
  if [ "$ok" = 1 ]; then ln -sf $H/data/ref_v1/$T/${c}_motion.npz $R/$T/; ln -sf $j $R/$T/; fi
done
export HNU_BRIDGE_ROOT=$R
SRC=$(cat $H/runs/bridge_eval/bridge1/checkpoint.txt)
O=$H/runs/bridge_eval/${V}_zeroshot_bridge1
mkdir -p $O
CUDA_VISIBLE_DEVICES=$G $PY eval_expert.py --task $T --bridge --ckpt $SRC --per-clip 8 --out $O > $O/clean.log 2>&1
RUN=bridge_$V
( cd $H/runs && CUDA_VISIBLE_DEVICES=$G $PY $H/code/train_hnu.py HNU-Track-$T-Bridge --env.scene.num-envs 4096 \
    --agent.experiment-name hnu_track_$T --agent.run-name $RUN --agent.max-iterations $ITERS \
    --agent.resume True --agent.load-run $(basename $(dirname $SRC)) --agent.load-checkpoint $(basename $SRC) \
    > $H/runs/logs/${T}_$RUN.log 2>&1 < /dev/null )
RD=$(ls -d $H/runs/logs/rsl_rl/hnu_track_$T/*_$RUN | tail -1)
CK=$(ls $RD/model_*.pt | sort -V | tail -1)
O=$H/runs/bridge_eval/$RUN
mkdir -p $O
echo $CK > $O/checkpoint.txt
CUDA_VISIBLE_DEVICES=$G $PY eval_expert.py --task $T --bridge --ckpt $CK --per-clip 8 --out $O > $O/clean.log 2>&1
CUDA_VISIBLE_DEVICES=$G $PY eval_expert.py --task $T --bridge --ckpt $CK --per-clip 8 --push --noise --out $O > $O/push.log 2>&1
export MUJOCO_GL=egl PYTHONPATH=$H/code
for C in CrouchedTraverse_007 CrouchedTraverse_021; do
  CUDA_VISIBLE_DEVICES=$G $PY tools/record_expert.py HNU-Track-$T-Bridge $C $CK $H/videos/geometry/${RUN}_$C.mp4 \
    "CrouchedTraverse expert, DuMuQiao.usd + markers ($V)" >> $H/videos/geometry/log.txt 2>&1
done
echo done > $O/DONE
