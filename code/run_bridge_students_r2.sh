#!/bin/bash
# Bridge-scene students, corrected latency model.  Finding: students trained with per-step latency jitter
# (0-40 ms) fail with a sustained zero latency (L0 seed 1: 9-13 % vs 85-89 % with 20-40 ms) -> latency is now
# piecewise constant (0.5 s periods).  Seed 1 (already trained): +1000 DAgger its under the new latency model;
# seeds 2-3: full pipeline (4000 + round 2 2000) under the new model.  One student per GPU.
# Evaluation: nominal = no randomisation and zero latency; randomised = full DR; clean / push.
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
N=$H/runs/night
V=v6r2
cd $H/code
export HF_HUB_OFFLINE=1 HNU_BRIDGE_ROOT=$H/data/ref_bridge_video_lift HNU_START_FRAC=0.25
log() { echo "[$(date +%H:%M:%S)] $*" >> $N/$V.log; }
CK=$(cat $N/${V}_teachers.txt)
COMMON="--scene bridge --z-offsets 0,5,10,15,20,25,30,40,50 --hidden 1024,512,256 --hist 20 --lin-vel"
evals() {  # student latent_dir out_tag gpu P
  for DRV in 0 1; do for X in "" "--push"; do
    HNU_DR=$DRV HNU_START_FRAC=0 CUDA_VISIBLE_DEVICES=$4 $PY eval_student.py --student $1 --latent-dir $2 --per-clip 8 $X \
      --out $H/runs/results_${V}_$3_dr$DRV --tag _$5 >> $N/${V}_F_$3_$5.log 2>&1
  done; done
}
fix1() {  # P gpu : seed-1 latency fix-up (+1000 its)
  local P=$1 G=$2 O=$H/runs/students_${V}_d3_s1
  [ -f $O/$P/student_1000.pt ] || HNU_DR=1 HNU_TB_ROOT=$H/runs/logs/rsl_rl/students_${V}_d3_s1 CUDA_VISIBLE_DEVICES=$G \
    $PY distill_train.py --profile ${V}d3_$P $CK --num-envs 4096 --iters 1000 --beta-iters 1 --lr 2e-4 --action-noise 0.1 \
    --fail-weight 2 --seed 3000 $COMMON --init-from $H/runs/students_${V}_d2_s1/$P/student_2000.pt --latent-dir $H/data/z_tables_v5/$P \
    --out $O/$P > $N/${V}_d3_s1_$P.log 2>&1
  [ -f $O/$P/student_1000.pt ] && evals $O/$P/student_1000.pt $H/data/z_tables_v5/$P s1 $G $P
}
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
  [ -f $O/$P/student_2000.pt ] && evals $O/$P/student_2000.pt $H/data/z_tables_v5/$P s$s $G $P
}
# seed-1 Z3 round 2 was still running under the old latency: finish it under the new one first
rm -rf $H/runs/students_${V}_d2_s1/Z3 $H/runs/students_${V}_s2 $H/runs/students_${V}_d2_s2 $H/runs/results_${V}_s1_dr0 $H/runs/results_${V}_s1_dr1
log "students, corrected latency model (piecewise-constant 0-40 ms)"
JOBS=("fix Z0" "fix Z1" "fix Z2" "one3 Z3" "fix Z4"); for s in 2 3; do for P in Z0 Z1 Z2 Z3 Z4; do JOBS+=("$s $P"); done; done
run_job() {  # job gpu
  local k=${1%% *} P=${1##* }
  if [ $k = fix ]; then fix1 $P $2
  elif [ $k = one3 ]; then one $P $2 1; fix1 $P $2
  else one $P $2 $k; fi
}
queue() { for i in "${!JOBS[@]}"; do [ $((i % 2)) = $2 ] && run_job "${JOBS[$i]}" $1; done; }
queue 0 0 & queue 1 1 & wait
for j in "${JOBS[@]}"; do run_job "$j" 0; done
for DRV in 0 1; do
  $PY tools/bridge_err.py $H/runs/results_${V}_r2_full_teacher_dr$DRV $H/runs/results_${V}_s1_dr$DRV $H/runs/results_${V}_s2_dr$DRV \
    $H/runs/results_${V}_s3_dr$DRV > $N/${V}_summary_dr$DRV.txt 2>&1
done
n=$(( $(ls $H/runs/students_${V}_d3_s1/Z*/student_1000.pt 2>/dev/null | wc -l) + $(ls $H/runs/students_${V}_d2_s[23]/Z*/student_2000.pt 2>/dev/null | wc -l) ))
log "students done: $n of 15"
[ "$n" = 15 ] && log "finished"
