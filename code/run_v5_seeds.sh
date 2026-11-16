#!/bin/bash
# v5 + DAgger round 2, extra RL seeds 2 and 3 (bridge scene), same settings as seed 1.  Starts when the stakes
# round-2 runs are finished (GPU idle).  At most 3 jobs on GPU0 and 2 on GPU1 at a time.
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
N=$H/runs/night
cd $H/code
export HF_HUB_OFFLINE=1 HNU_BRIDGE_ROOT=$H/data/ref_bridge_video_lift
ck() { cat $H/runs/expert_eval/$1/checkpoint.txt; }
CKD=$(cat $H/runs/bridge_eval/bridge_video_lift_perc/checkpoint.txt)
CK="--ckpt-ThreeStepPoleRun $(ck ThreeStepPoleRun) --ckpt-FiveStepPoleRun $(ck FiveStepPoleRun) --ckpt-CrouchedTraverse $CKD"
ZOFF=0,5,10,15,20,25,30,40,50
log() { echo "[$(date +%H:%M:%S)] $*" >> $N/v5_seeds.log; }
for i in $(seq 1 600); do [ -f $H/runs/stakes_eval/perc2_summary.txt ] && break; sleep 60; done
log "start"
one() {  # P gpu seed
  local P=$1 G=$2 s=$3 S=$H/runs/students_v5_s$3 O=$H/runs/students_v5_d2_s$3
  [ -f $S/$P/student_4000.pt ] || HNU_TB_ROOT=$H/runs/logs/rsl_rl/students_v5_s$s CUDA_VISIBLE_DEVICES=$G $PY distill_train.py \
    --profile v5_$P $CK --num-envs 4096 --iters 4000 --seed $((1000 * s)) --scene bridge --z-offsets $ZOFF --hidden 1024,512,256 \
    --latent-dir $H/data/z_tables_v5/$P --out $S/$P > $N/v5_s${s}_E_$P.log 2>&1
  [ -f $S/$P/student_4000.pt ] || return
  for X in "" "--push"; do
    CUDA_VISIBLE_DEVICES=$G $PY eval_student.py --student $S/$P/student_4000.pt --latent-dir $H/data/z_tables_v5/$P --per-clip 8 $X \
      --out $H/runs/results_v5_s$s --tag _$P >> $N/v5_s${s}_F_$P.log 2>&1
  done
  [ -f $O/$P/student_2000.pt ] || HNU_TB_ROOT=$H/runs/logs/rsl_rl/students_v5_d2_s$s CUDA_VISIBLE_DEVICES=$G $PY distill_train.py \
    --profile v5d2_$P $CK --num-envs 4096 --iters 2000 --beta-iters 1 --lr 2e-4 --action-noise 0.1 --seed $((2000 * s)) --scene bridge \
    --z-offsets $ZOFF --hidden 1024,512,256 --init-from $S/$P/student_4000.pt --latent-dir $H/data/z_tables_v5/$P --out $O/$P \
    > $N/v5_d2_s${s}_$P.log 2>&1
  for X in "" "--push"; do
    [ -f $O/$P/student_2000.pt ] && CUDA_VISIBLE_DEVICES=$G $PY eval_student.py --student $O/$P/student_2000.pt \
      --latent-dir $H/data/z_tables_v5/$P --per-clip 8 $X --out $H/runs/results_v5_d2_s$s --tag _$P >> $N/v5_d2_s${s}_F_$P.log 2>&1
  done
}
for s in 2 3; do
  log "seed $s"
  ( one Z0 0 $s; one Z3 0 $s ) & ( one Z1 0 $s ) & ( one Z2 1 $s; one Z4 1 $s ) & wait
  for P in Z0 Z1 Z2 Z3 Z4; do one $P 1 $s; done  # retry anything that died
done
$PY tools/bridge_table.py $H/runs/results_v5 $H/runs/results_v5_s2 $H/runs/results_v5_s3 $H/runs/results_v5_d2 \
  $H/runs/results_v5_d2_s2 $H/runs/results_v5_d2_s3 > $N/bridge_v5_seeds_table.md 2>&1
log "finished"
