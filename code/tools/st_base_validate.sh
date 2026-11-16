#!/bin/bash
# model/validation of the Deployment bundle for a new export (as tools/deploy_v2_run.sh: base = 0-20 ms,
# lat20 = fixed 20 ms; fold-1 plan4 geometry, 8 trials per clip), now with the post bases (meta post_base).
# usage: bash tools/st_base_validate.sh <export dir>
PYD=/root/gpufree-data/venvs/deploy/bin/python
H=/root/gpufree-data/hnu_v2; D=$H/data; EX=$1
cd $H/code
export PYTHONPATH=$H/code
GPD="$D/geom_plan4_f1/ref_stakes_s93_h10/ThreeStepPoleRun,$D/geom_plan4_f1/ref_stakes_s83_h10/FiveStepPoleRun,$D/ref_bridge_video_lift/CrouchedTraverse"
mkdir -p $EX/validation
for NAME in base lat20; do
  LAT="0-20"; [ $NAME = lat20 ] && LAT=20
  [ -f $EX/validation/$NAME.json ] || $PYD deploy/sim2sim.py $EX --latency $LAT --crit task --trials 8 --workers 96 \
    --motion-dirs $GPD --latent-dir $D/z_lib_f1/plan4/Z3 --out $EX/validation/$NAME.json >> $EX/validation/validation.log 2>&1
done
