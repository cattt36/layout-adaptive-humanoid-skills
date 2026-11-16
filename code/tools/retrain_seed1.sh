#!/bin/bash
# Bridge students, seed 1 retrained from scratch with the corrected latency model (user ), so that all three
# seeds follow the same procedure.  The fixed-up seed-1 models are kept as the latency-model ablation ("s1fix").
# Runs after run_bridge_students_r2.sh has finished seeds 2-3; one student per GPU.
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
N=$H/runs/night
V=v6r2
cd $H/code
export HF_HUB_OFFLINE=1 HNU_BRIDGE_ROOT=$H/data/ref_bridge_video_lift HNU_START_FRAC=0.25
log() { echo "[$(date +%H:%M:%S)] $*" >> $N/$V.log; }
for i in $(seq 1 3000); do grep -q -e '] finished' -e 'students done' $N/$V.log && break; sleep 60; done
log "seed 1 retrain from scratch (corrected latency); fixed-up seed-1 models kept as s1fix"
for DRV in 0 1; do [ -d $H/runs/results_${V}_s1_dr$DRV ] && mv $H/runs/results_${V}_s1_dr$DRV $H/runs/results_${V}_s1fix_dr$DRV; done
[ -d $H/runs/students_${V}_s1 ] && mv $H/runs/students_${V}_s1 $H/runs/students_${V}_s1old
[ -d $H/runs/students_${V}_d2_s1 ] && mv $H/runs/students_${V}_d2_s1 $H/runs/students_${V}_d2_s1old
CK=$(cat $N/${V}_teachers.txt)
COMMON="--scene bridge --z-offsets 0,5,10,15,20,25,30,40,50 --hidden 1024,512,256 --hist 20 --lin-vel"
one() {  # P gpu
  local P=$1 G=$2 S=$H/runs/students_${V}_s1 O=$H/runs/students_${V}_d2_s1
  [ -f $S/$P/student_4000.pt ] || HNU_DR=1 HNU_TB_ROOT=$H/runs/logs/rsl_rl/students_${V}_s1 CUDA_VISIBLE_DEVICES=$G \
    $PY distill_train.py --profile ${V}_$P $CK --num-envs 4096 --iters 4000 --seed 1000 $COMMON \
    --latent-dir $H/data/z_tables_v5/$P --out $S/$P > $N/${V}_s1_E_$P.log 2>&1
  [ -f $S/$P/student_4000.pt ] || return
  [ -f $O/$P/student_2000.pt ] || HNU_DR=1 HNU_TB_ROOT=$H/runs/logs/rsl_rl/students_${V}_d2_s1 CUDA_VISIBLE_DEVICES=$G \
    $PY distill_train.py --profile ${V}d2_$P $CK --num-envs 4096 --iters 2000 --beta-iters 1 --lr 2e-4 --action-noise 0.1 \
    --fail-weight 2 --seed 2000 $COMMON --init-from $S/$P/student_4000.pt --latent-dir $H/data/z_tables_v5/$P \
    --out $O/$P > $N/${V}_d2_s1_$P.log 2>&1
  [ -f $O/$P/student_2000.pt ] || return
  for DRV in 0 1; do for X in "" "--push"; do
    HNU_DR=$DRV HNU_START_FRAC=0 CUDA_VISIBLE_DEVICES=$G $PY eval_student.py --student $O/$P/student_2000.pt \
      --latent-dir $H/data/z_tables_v5/$P --per-clip 8 $X --out $H/runs/results_${V}_s1_dr$DRV --tag _$P >> $N/${V}_F_s1_$P.log 2>&1
  done; done
}
( one Z0 0; one Z2 0; one Z4 0 ) & ( one Z1 1; one Z3 1 ) & wait
for P in Z0 Z1 Z2 Z3 Z4; do one $P 0; done
for DRV in 0 1; do
  $PY tools/bridge_err.py $H/runs/results_${V}_r2_full_teacher_dr$DRV $H/runs/results_${V}_s1_dr$DRV $H/runs/results_${V}_s2_dr$DRV \
    $H/runs/results_${V}_s3_dr$DRV > $N/${V}_summary_dr$DRV.txt 2>&1
done
log "seed1 retrain done"
