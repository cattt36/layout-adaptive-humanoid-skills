#!/bin/bash
# v6 + robustness, second attempt.  v6r showed (a) the full randomisation at once is too hard: the experts restart
# from ~22-step episodes and the bridge expert only recovers to ~120 steps (teacher bank 0 % on the bridge);
# (b) the students died with CUDA OOM because the GPU was shared with the stake training.
# Fixes: DR curriculum (mild 4000 its -> full 6000 its), teacher gate (nominal >= 80 % and randomised >= 60 % on
# CrouchedTraverse in the distillation env, else one more full-DR block), students only after the stake runs are done.
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
N=$H/runs/night
V=v6r2
cd $H/code
export HF_HUB_OFFLINE=1 HNU_BRIDGE_ROOT=$H/data/ref_bridge_video_lift
log() { echo "[$(date +%H:%M:%S)] $*" >> $N/$V.log; }
last() { local RD=$(ls -d $H/runs/logs/rsl_rl/hnu_track_$1/*_$2 | tail -1); ls $RD/model_*.pt | sort -V | tail -1; }
stage() {  # Task suffix gpu src run iters level extra_env...
  local T=$1 SUF=$2 G=$3 SRC=$4 RUN=$5 IT=$6 LV=$7; shift 7
  [ -n "$(ls $H/runs/logs/rsl_rl/hnu_track_$T/*_$RUN/model_*.pt 2>/dev/null)" ] && return
  ( cd $H/runs && env HNU_DR=1 HNU_DR_LEVEL=$LV "$@" CUDA_VISIBLE_DEVICES=$G $PY $H/code/train_hnu.py HNU-Track-$T$SUF \
      --env.scene.num-envs 4096 --agent.experiment-name hnu_track_$T --agent.run-name $RUN --agent.max-iterations $IT \
      --agent.resume True --agent.load-run $(basename $(dirname $SRC)) --agent.load-checkpoint $(basename $SRC) \
      > $H/runs/logs/${T}_$RUN.log 2>&1 < /dev/null )
}
BR="HNU_PERCEPTION=1 HNU_BRIDGE_W=-5.0"
log "R1a mild DR (GPU0)"
stage CrouchedTraverse -Bridge 0 $(cat $H/runs/bridge_eval/bridge_video_lift_perc/checkpoint.txt) r2_mild_bridge 4000 1 $BR &
stage ThreeStepPoleRun "" 0 $(cat $H/runs/expert_eval/ThreeStepPoleRun/checkpoint.txt) r2_mild_flat 4000 1 &
stage FiveStepPoleRun "" 0 $(cat $H/runs/expert_eval/FiveStepPoleRun/checkpoint.txt) r2_mild_flat 4000 1 &
wait
log "R1b full DR"
stage CrouchedTraverse -Bridge 0 $(last CrouchedTraverse r2_mild_bridge) r2_full_bridge 6000 2 $BR &
stage ThreeStepPoleRun "" 0 $(last ThreeStepPoleRun r2_mild_flat) r2_full_flat 6000 2 &
stage FiveStepPoleRun "" 0 $(last FiveStepPoleRun r2_mild_flat) r2_full_flat 6000 2 &
wait
gate() {  # evaluate the teacher bank; prints "dumu_nominal dumu_random"
  local CK="--ckpt-ThreeStepPoleRun $(last ThreeStepPoleRun $1_flat) --ckpt-FiveStepPoleRun $(last FiveStepPoleRun $1_flat) --ckpt-CrouchedTraverse $(last CrouchedTraverse $1_bridge)"
  for DRV in 0 1; do
    for X in "" "--push"; do
      HNU_DR=$DRV CUDA_VISIBLE_DEVICES=0 $PY eval_student.py --teacher $CK --scene bridge --z-offsets 0,5,10,15,20,25,30,40,50 \
        --latent-dir $H/data/z_tables_v5/Z4 --per-clip 8 $X --out $H/runs/results_${V}_$1_teacher_dr$DRV --tag _teacher >> $N/${V}_teacher.log 2>&1
    done
  done
  # retention: each robust expert must still perform its original motion (nominal, from frame 0) within 5 points
  local R1=$(CUDA_VISIBLE_DEVICES=0 $PY tools/retention_check.py --task ThreeStepPoleRun --mode flat --orig $(cat $H/runs/expert_eval/ThreeStepPoleRun/checkpoint.txt) --tuned $(last ThreeStepPoleRun $1_flat))
  local R2=$(CUDA_VISIBLE_DEVICES=0 $PY tools/retention_check.py --task FiveStepPoleRun --mode flat --orig $(cat $H/runs/expert_eval/FiveStepPoleRun/checkpoint.txt) --tuned $(last FiveStepPoleRun $1_flat))
  local R3=$(CUDA_VISIBLE_DEVICES=0 $PY tools/retention_check.py --task CrouchedTraverse --mode bridge --perc-orig 1 --perc-tuned 1 --orig $(cat $H/runs/bridge_eval/bridge_video_lift_perc/checkpoint.txt) --tuned $(last CrouchedTraverse $1_bridge))
  echo "$R1" >> $N/${V}_retention.log; echo "$R2" >> $N/${V}_retention.log; echo "$R3" >> $N/${V}_retention.log
  local RET=1
  for r in "$R1" "$R2" "$R3"; do echo "$r" | grep -q '"retained": true' || RET=0; done
  $PY - << EOF
import json
a = [json.load(open(f"$H/runs/results_${V}_$1_teacher_dr{d}/eval_teacher_teacher.json"))["summary"]["CrouchedTraverse/all"]["success"] for d in (0, 1)]
print(a[0], a[1], $RET)
EOF
}
TAG=r2_full
read NOM RND RET <<< "$(gate r2_full)"
log "gate r2_full: DuMu nominal $NOM randomised $RND retention-ok $RET"
if $PY -c "import sys; sys.exit(0 if ($NOM >= 0.8 and $RND >= 0.6 and $RET == 1) else 1)"; then :; else
  log "gate failed -> extra full-DR block"
  stage CrouchedTraverse -Bridge 0 $(last CrouchedTraverse r2_full_bridge) r2_more_bridge 6000 2 $BR &
  stage ThreeStepPoleRun "" 0 $(last ThreeStepPoleRun r2_full_flat) r2_more_flat 6000 2 &
  stage FiveStepPoleRun "" 0 $(last FiveStepPoleRun r2_full_flat) r2_more_flat 6000 2 &
  wait
  TAG=r2_more
  read NOM RND RET <<< "$(gate r2_more)"
  log "gate r2_more: DuMu nominal $NOM randomised $RND retention-ok $RET"
  [ "$RET" = 1 ] || log "WARNING: retention still not met after the extra block -- check before using these teachers"
fi
CK="--ckpt-ThreeStepPoleRun $(last ThreeStepPoleRun ${TAG}_flat) --ckpt-FiveStepPoleRun $(last FiveStepPoleRun ${TAG}_flat) --ckpt-CrouchedTraverse $(last CrouchedTraverse ${TAG}_bridge)"
echo "$CK" > $N/${V}_teachers.txt
# students: wait until the stake runs are finished (GPU1 free)
for i in $(seq 1 600); do [ -f $H/runs/stakes_eval/runup_h06_summary.txt ] && break; sleep 60; done
log "R2 students (teachers $TAG)"
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
  for DRV in 0 1; do for X in "" "--push"; do
    HNU_DR=$DRV CUDA_VISIBLE_DEVICES=$G $PY eval_student.py --student $O/$P/student_2000.pt --latent-dir $H/data/z_tables_v5/$P \
      --per-clip 8 $X --out $H/runs/results_${V}_s${s}_dr$DRV --tag _$P >> $N/${V}_F_s${s}_$P.log 2>&1
  done; done
}
for s in 1 2 3; do
  log "seed $s"
  ( one Z0 0 $s; one Z3 0 $s ) & ( one Z1 0 $s ) & ( one Z2 1 $s; one Z4 1 $s ) & wait
  for P in Z0 Z1 Z2 Z3 Z4; do one $P 1 $s; done
done
n=$(ls $H/runs/students_${V}_d2_s*/Z*/student_2000.pt 2>/dev/null | wc -l)
for DRV in 0 1; do
  $PY tools/bridge_err.py $H/runs/results_${V}_${TAG}_teacher_dr$DRV $H/runs/results_${V}_s1_dr$DRV $H/runs/results_${V}_s2_dr$DRV \
    $H/runs/results_${V}_s3_dr$DRV > $N/${V}_summary_dr$DRV.txt 2>&1
done
log "students done: $n of 15"
[ "$n" = 15 ] && log "finished"
