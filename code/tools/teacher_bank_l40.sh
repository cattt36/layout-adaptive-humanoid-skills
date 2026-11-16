#!/bin/bash
# Teacher bank on the L40 server -- exactly the evaluation block of run_v6g10n_l40.sh (which then skips it), on GPU ${1:-2}.
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
N=$H/runs/night
V=v6g10n
cd $H/code
export HF_HUB_OFFLINE=1 HNU_BRIDGE_ROOT=$H/data/ref_bridge_video_lift HNU_STAKE_H=0.10 HNU_GEOM_W1=$H/data/ref_stakes_s93_h10 \
  HNU_GEOM_W2=$H/data/ref_stakes_s83_h10 HNU_STAKE_TOL=0.17 HNU_START_FRAC=0.25 HNU_DR_LEVEL=1 HNU_DR_LAG_SYNC=1 \
  HNU_DR_LAG_SENSORS=1 PYTHONPATH=$H/code HNU_STAKE_SHAPE=mesh16 HNU_POST_SHAPE=mesh12
CK=$(cat $N/${V}_teachers.txt)
for CR in task track; do
  R=$H/runs/results_${V}; [ $CR = track ] && R=$H/runs/results_${V}trk
  for DRV in 0 1; do for X in "" "--push"; do
    [ -f ${R}_bank_teacher_dr$DRV/eval_teacher_teacher$([ -n "$X" ] && echo _push).json ] && continue
    HNU_SPLIT_JSON=$H/data/clip_split_trial/split_v2.json HNU_EVAL_CRIT=$CR HNU_DR=$DRV HNU_START_FRAC=0 CUDA_VISIBLE_DEVICES=${1:-2} \
      $PY eval_student.py --teacher $CK --scene geom --z-offsets 0,5,10,15,20,25,30,40,50 --latent-dir $H/data/z_tables_v5t/Z4 \
      --per-clip 32 $X --out ${R}_bank_teacher_dr$DRV --tag _teacher >> $N/${V}_teacher.log 2>&1
  done; done
done
echo done
