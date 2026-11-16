#!/bin/bash
# v6g: final students in the unified geometry scene (ThreeStepPoleRun / FiveStepPoleRun on 0.06 m stakes, CrouchedTraverse under the bridge),
# with robustness.  Teachers: robust stake experts (tools/robust_stakes.sh) + robust bridge expert (run_v6r2.sh).
# Students (all DAgger): v6 inputs (estimated base velocity, 20-step history, obstacle features, 9 z_skill samples),
# DAgger 4000 + round 2 2000 (action noise, failure weighting), randomised dynamics + latency, 25 % frame-0 resets;
# L0-L4 x RL seeds 1-3.  Evaluation: nominal / randomised x clean / push; strict stake criterion (stake_missed).
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
N=$H/runs/night
V=v6g10
cd $H/code
export HF_HUB_OFFLINE=1 HNU_BRIDGE_ROOT=$H/data/ref_bridge_video_lift HNU_STAKE_H=0.10 HNU_GEOM_W1=$H/data/ref_stakes_s93_h10 HNU_GEOM_W2=$H/data/ref_stakes_s83_h10 HNU_STAKE_TOL=0.17 HNU_START_FRAC=0.25 HNU_DR_LEVEL=1  # deployment latency model 0-20 ms
log() { echo "[$(date +%H:%M:%S)] $*" >> $N/$V.log; }
last() { local RD=$(ls -d $H/runs/logs/rsl_rl/hnu_track_$1/*_$2 | tail -1); ls $RD/model_*.pt | sort -V | tail -1; }
log "waiting for the robust 0.10 m stake teachers (retention ok) and for the bridge-scene students to finish"
for i in $(seq 1 3000); do
  grep -q done $H/runs/stakes_eval/robust_stakes_h10c/summary.txt 2>/dev/null && break
  sleep 60
done
# retention: original-stride check failed (W1 -6, W2 -19 points) but the robot-scale-stride check passes (W1 100->99 %,
# W2 87.5->90.6 %): the drop is stride-specific, the motion is intact -> teachers h10c accepted by the user
if [ "${HNU_RETENTION_GATE:-0}" = 1 ] && grep -q '"retained": false' $H/runs/stakes_eval/robust_stakes_h10c/retention.txt; then
  log "STOP: a robust 0.10 m stake teacher failed the retention check -- needs review before distillation"; exit 1
fi
CKD=$(grep -o -e '--ckpt-CrouchedTraverse [^ ]*' $N/v6r2_teachers.txt | cut -d' ' -f2)
CK="--ckpt-ThreeStepPoleRun $(cat $H/runs/stakes_eval/robust_stakes_h10c/ThreeStepPoleRun/checkpoint.txt) --ckpt-FiveStepPoleRun $(cat $H/runs/stakes_eval/robust_stakes_h10c/FiveStepPoleRun/checkpoint.txt) --ckpt-CrouchedTraverse $CKD"
echo "$CK" > $N/${V}_teachers.txt
log "teachers: $CK"
for DRV in 0 1; do for X in "" "--push"; do
  HNU_DR=$DRV HNU_START_FRAC=0 CUDA_VISIBLE_DEVICES=0 $PY eval_student.py --teacher $CK --scene geom --z-offsets 0,5,10,15,20,25,30,40,50 \
    --latent-dir $H/data/z_tables_v5/Z4 --per-clip 8 $X --out $H/runs/results_${V}_bank_teacher_dr$DRV --tag _teacher >> $N/${V}_teacher.log 2>&1
done; done
log "teacher bank evaluated"
COMMON="--scene geom --z-offsets 0,5,10,15,20,25,30,40,50 --hidden 1024,512,256 --hist 20 --lin-vel"
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
    HNU_DR=$DRV HNU_START_FRAC=0 CUDA_VISIBLE_DEVICES=$G $PY eval_student.py --student $O/$P/student_2000.pt \
      --latent-dir $H/data/z_tables_v5/$P --per-clip 8 $X --out $H/runs/results_${V}_s${s}_dr$DRV --tag _$P >> $N/${V}_F_s${s}_$P.log 2>&1
  done; done
}
JOBS=(); for s in 1 2 3; do for P in Z0 Z1 Z2 Z3 Z4; do JOBS+=("$P $s"); done; done
queue() { for i in "${!JOBS[@]}"; do [ $((i % 2)) = $2 ] && one ${JOBS[$i]%% *} $1 ${JOBS[$i]##* }; done; }  # one student per GPU
queue 0 0 & queue 1 1 & wait
for j in "${JOBS[@]}"; do one ${j%% *} 0 ${j##* }; done
for T in CrouchedTraverse ThreeStepPoleRun FiveStepPoleRun; do
  for DRV in 0 1; do
    HNU_TASK=$T $PY tools/exec_quality_table.py $H/runs/results_${V}_bank_teacher_dr$DRV $H/runs/results_${V}_s1_dr$DRV \
      $H/runs/results_${V}_s2_dr$DRV $H/runs/results_${V}_s3_dr$DRV > $N/${V}_table_${T}_dr$DRV.txt 2>&1
  done
done
n=$(ls $H/runs/students_${V}_d2_s*/Z*/student_2000.pt 2>/dev/null | wc -l)
log "students done: $n of 15"
$PY tools/select_seed.py $H/runs/results_${V} > $N/${V}_selection.txt 2>&1
[ "$n" = 15 ] && log "finished"
