#!/bin/bash
# Re-run one student. usage: rerun_student.sh <version: v3|v4b> <profile Z0..Z4> <gpu> [iters]
V=$1; P=$2; G=$3; I=${4:-2500}
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
cd $H/code
ck() { cat $H/runs/expert_eval/$1/checkpoint.txt; }
CK="--ckpt-ThreeStepPoleRun $(ck ThreeStepPoleRun) --ckpt-FiveStepPoleRun $(ck FiveStepPoleRun) --ckpt-CrouchedTraverse $(ck CrouchedTraverse)"
O=$H/runs/students_$V/$P
[ -d $O ] && mv $O ${O}_failed_$(date +%H%M%S)
export HF_HUB_OFFLINE=1 HNU_TB_ROOT=$H/runs/logs/rsl_rl/students_$V
SEEDARG=""; [ "$V" = v4d ] && SEEDARG="--seed 1000"
CUDA_VISIBLE_DEVICES=$G nohup $PY distill_train.py --profile ${V}_$P $CK --num-envs 4096 --iters $I $SEEDARG \
  --latent-dir $H/data/z_tables_$V/$P --out $O > $H/runs/night/${V}_E_${P}_s1.log 2>&1 < /dev/null &
echo "rerun $V $P on gpu $G"
