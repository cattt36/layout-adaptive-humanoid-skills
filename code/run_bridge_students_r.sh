#!/bin/bash
# CrouchedTraverse students now (user ): v6 inputs, robust teachers from v6r2 (gate passed: DuMu nominal 93.1 %,
# randomised 85.6 %, retention ok), bridge scene (ThreeStepPoleRun / FiveStepPoleRun flat in this scene), randomised training, 25 % frame-0
# resets; L0-L4 x RL seeds 1-3; DAgger 4000 + round 2 2000.  Eval nominal / randomised x clean / push.
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
N=$H/runs/night
V=v6r2
cd $H/code
export HF_HUB_OFFLINE=1 HNU_BRIDGE_ROOT=$H/data/ref_bridge_video_lift HNU_START_FRAC=0.25
log() { echo "[$(date +%H:%M:%S)] $*" >> $N/$V.log; }
CK=$(cat $N/${V}_teachers.txt)
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
    HNU_DR=$DRV HNU_START_FRAC=0 CUDA_VISIBLE_DEVICES=$G $PY eval_student.py --student $O/$P/student_2000.pt \
      --latent-dir $H/data/z_tables_v5/$P --per-clip 8 $X --out $H/runs/results_${V}_s${s}_dr$DRV --tag _$P >> $N/${V}_F_s${s}_$P.log 2>&1
  done; done
}
rm -rf $H/runs/students_${V}_s1 $H/runs/students_${V}_d2_s1 $H/runs/students_${V}_s2 $H/runs/students_${V}_d2_s2  # partial runs
log "R2 students (one per GPU, two queues)"
# a v6 student needs ~16 GB at start-up -> one student per GPU; two queues over the 15 (level, seed) jobs
JOBS=(); for s in 1 2 3; do for P in Z0 Z1 Z2 Z3 Z4; do JOBS+=("$P $s"); done; done
queue() {  # gpu parity
  for i in "${!JOBS[@]}"; do [ $((i % 2)) = $2 ] && one ${JOBS[$i]%% *} $1 ${JOBS[$i]##* }; done
}
queue 0 0 & queue 1 1 & wait
for j in "${JOBS[@]}"; do one ${j%% *} 0 ${j##* }; done  # retry anything that died
for DRV in 0 1; do
  $PY tools/bridge_err.py $H/runs/results_${V}_r2_full_teacher_dr$DRV $H/runs/results_${V}_s1_dr$DRV $H/runs/results_${V}_s2_dr$DRV \
    $H/runs/results_${V}_s3_dr$DRV > $N/${V}_summary_dr$DRV.txt 2>&1
done
n=$(ls $H/runs/students_${V}_d2_s*/Z*/student_2000.pt 2>/dev/null | wc -l)
log "students done: $n of 15"
[ "$n" = 15 ] && log "finished"
