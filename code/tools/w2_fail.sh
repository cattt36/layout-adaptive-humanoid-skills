#!/bin/bash
# Where do L0 / L4 (v6g10j s1, v6g10t s1) fail on the hard FiveStepPoleRun trials?  Nominal, task criterion, detail rows.
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
cd $H/code
export HF_HUB_OFFLINE=1 HNU_BRIDGE_ROOT=$H/data/ref_bridge_video_lift HNU_STAKE_H=0.10 HNU_GEOM_W1=$H/data/ref_stakes_s93_h10 \
  HNU_GEOM_W2=$H/data/ref_stakes_s83_h10 HNU_STAKE_TOL=0.17 HNU_DR_LEVEL=1 HNU_DR_LAG_SYNC=1 HNU_DR_LAG_SENSORS=1 \
  PYTHONPATH=$H/code HNU_SPLIT_JSON=$H/data/clip_split_trial/split.json HNU_EVAL_CRIT=task HNU_EVAL_DETAIL=1 HNU_DR=0 HNU_START_FRAC=0
O=$H/runs/w2_fail
mkdir -p $O
# the geom scene builds the bridge from a bridge clip -> one CrouchedTraverse trial is included (ignored in the summary)
printf "FiveStepPoleRun_002\nFiveStepPoleRun_005\nFiveStepPoleRun_010\nFiveStepPoleRun_014\nFiveStepPoleRun_013\nCrouchedTraverse_008\n" > $O/clips.txt
export HNU_CLIPS_ALLOW=$O/clips.txt
G=1; [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 | tr -d ' ')" -gt "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1 | tr -d ' ')" ] && G=0
CK=$(cat $H/runs/night/v6g10t_teachers.txt)
CUDA_VISIBLE_DEVICES=$G $PY eval_student.py --teacher $CK --scene geom --z-offsets 0,5,10,15,20,25,30,40,50 \
  --latent-dir $H/data/z_tables_v5t/Z4 --per-clip 32 --out $O --tag _teacher > $O/teacher.log 2>&1
for B in j t; do for P in Z0 Z4; do
  CUDA_VISIBLE_DEVICES=$G $PY eval_student.py --student $H/runs/students_v6g10${B}_d2_s1/$P/student_2000.pt \
    --latent-dir $H/data/z_tables_v5t/$P --per-clip 32 --out $O --tag _${B}_$P > $O/${B}_$P.log 2>&1 || tail -5 $O/${B}_$P.log
done; done
$PY tools/w2_fail.py
