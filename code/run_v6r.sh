#!/bin/bash
# v6 + robustness (user decision train v6 directly with robustness).
# R1  robust teachers (HNU_DR=1: PD gains, effort limits, link masses, 0-40 ms latency, on top of pushes / CoM /
#     friction / encoder bias): CrouchedTraverse bridge-perception expert, ThreeStepPoleRun / FiveStepPoleRun flat experts, 4000 its each.
#     (The current bridge expert drops from 91% to 2% under this randomisation -> robust teachers are required.)
# R2  v6 students (estimated base velocity, 20-step history, failure-weighted DAgger round 2) distilled in the
#     randomised bridge scene from the robust teachers; L0-L4, RL seeds 1-3; DAgger 4000 + round 2 2000.
# R3  evaluation of teachers and students: nominal and randomised (robustness test), clean and push.
# Starts after the 0.06 m stake experts are evaluated.
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
N=$H/runs/night
V=v6r
cd $H/code
export HF_HUB_OFFLINE=1 HNU_BRIDGE_ROOT=$H/data/ref_bridge_video_lift
log() { echo "[$(date +%H:%M:%S)] $*" >> $N/$V.log; }
for i in $(seq 1 600); do [ -f $H/runs/stakes_eval/h06main_summary.txt ] && break; sleep 60; done
log "start"
last() { local RD=$(ls -d $H/runs/logs/rsl_rl/hnu_track_$1/*_$2 | tail -1); ls $RD/model_*.pt | sort -V | tail -1; }
robust() {  # Task task_id_suffix gpu src run extra_env...
  local T=$1 SUF=$2 G=$3 SRC=$4 RUN=$5; shift 5
  [ -n "$(ls $H/runs/logs/rsl_rl/hnu_track_$T/*_$RUN/model_*.pt 2>/dev/null)" ] && return
  ( cd $H/runs && env HNU_DR=1 "$@" CUDA_VISIBLE_DEVICES=$G $PY $H/code/train_hnu.py HNU-Track-$T$SUF --env.scene.num-envs 4096 \
      --agent.experiment-name hnu_track_$T --agent.run-name $RUN --agent.max-iterations 4000 \
      --agent.resume True --agent.load-run $(basename $(dirname $SRC)) --agent.load-checkpoint $(basename $SRC) \
      > $H/runs/logs/${T}_$RUN.log 2>&1 < /dev/null )
}
if [ ! -f $N/${V}_R1.done ]; then
  log "R1 robust teachers"
  robust CrouchedTraverse -Bridge 0 $(cat $H/runs/bridge_eval/bridge_video_lift_perc/checkpoint.txt) robust_bridge HNU_PERCEPTION=1 HNU_BRIDGE_W=-5.0 &
  robust ThreeStepPoleRun "" 1 $(cat $H/runs/expert_eval/ThreeStepPoleRun/checkpoint.txt) robust_flat &
  robust FiveStepPoleRun "" 1 $(cat $H/runs/expert_eval/FiveStepPoleRun/checkpoint.txt) robust_flat &
  wait
  touch $N/${V}_R1.done
fi
CKD=$(last CrouchedTraverse robust_bridge); CK1=$(last ThreeStepPoleRun robust_flat); CK2=$(last FiveStepPoleRun robust_flat)
CK="--ckpt-ThreeStepPoleRun $CK1 --ckpt-FiveStepPoleRun $CK2 --ckpt-CrouchedTraverse $CKD"
echo "$CK" > $N/${V}_teachers.txt
for DRV in 0 1; do  # teacher bank in the distillation env, nominal / randomised
  for X in "" "--push"; do
    HNU_DR=$DRV CUDA_VISIBLE_DEVICES=0 $PY eval_student.py --teacher $CK --scene bridge --z-offsets 0,5,10,15,20,25,30,40,50 \
      --latent-dir $H/data/z_tables_v5/Z4 --per-clip 8 $X --out $H/runs/results_${V}_teacher_dr$DRV --tag _teacher >> $N/${V}_teacher.log 2>&1
  done
done
log "R2 students"
COMMON="--scene bridge --z-offsets 0,5,10,15,20,25,30,40,50 --hidden 1024,512,256 --hist 20 --lin-vel"
one() {  # P gpu seed
  local P=$1 G=$2 s=$3 S=$H/runs/students_${V}_s$3 O=$H/runs/students_${V}_d2_s$3
  [ -f $S/$P/student_4000.pt ] || HNU_DR=1 HNU_TB_ROOT=$H/runs/logs/rsl_rl/students_${V}_s$s CUDA_VISIBLE_DEVICES=$G \
    $PY distill_train.py --profile ${V}_$P $CK --num-envs 4096 --iters 4000 --seed $((1000 * s)) $COMMON \
    --latent-dir $H/data/z_tables_v5/$P --out $S/$P > $N/${V}_s${s}_E_$P.log 2>&1
  [ -f $S/$P/student_4000.pt ] || return
  [ -f $O/$P/student_2000.pt ] || HNU_DR=1 HNU_TB_ROOT=$H/runs/logs/rsl_rl/students_${V}_d2_s$s CUDA_VISIBLE_DEVICES=$G \
    $PY distill_train.py --profile ${V}d2_$P $CK --num-envs 4096 --iters 2000 --beta-iters 1 --lr 2e-4 --action-noise 0.1 \
    --fail-weight 2 --seed $((2000 * s)) $COMMON --init-from $S/$P/student_4000.pt --latent-dir $H/data/z_tables_v5/$P \
    --out $O/$P > $N/${V}_d2_s${s}_$P.log 2>&1
  [ -f $O/$P/student_2000.pt ] || return
  for DRV in 0 1; do
    for X in "" "--push"; do
      HNU_DR=$DRV CUDA_VISIBLE_DEVICES=$G $PY eval_student.py --student $O/$P/student_2000.pt --latent-dir $H/data/z_tables_v5/$P \
        --per-clip 8 $X --out $H/runs/results_${V}_s${s}_dr$DRV --tag _$P >> $N/${V}_F_s${s}_$P.log 2>&1
    done
  done
}
for s in 1 2 3; do
  log "seed $s"
  ( one Z0 0 $s; one Z3 0 $s ) & ( one Z1 0 $s ) & ( one Z2 1 $s; one Z4 1 $s ) & wait
  for P in Z0 Z1 Z2 Z3 Z4; do one $P 1 $s; done
done
for DRV in 0 1; do
  $PY tools/bridge_err.py $H/runs/results_${V}_teacher_dr$DRV $H/runs/results_${V}_s1_dr$DRV $H/runs/results_${V}_s2_dr$DRV \
    $H/runs/results_${V}_s3_dr$DRV > $N/${V}_summary_dr$DRV.txt 2>&1
done
log "finished"
