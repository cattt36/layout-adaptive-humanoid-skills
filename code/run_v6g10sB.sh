#!/bin/bash
# Version B of the deployment students: robust to localisation error of the obstacle features (for experiments
# without motion capture).  Starts after v6g10s (version A) is finished, its deployment seeds are selected and its
# post-analysis is done.  For every level L0-L4, from the selected A student (round 2):
#   B  : DAgger round 3, 2000 its, HNU_PERC_ERR=1 (per-episode initial offset <= 5 cm, odometry scale / heading
#        error <= 2 %, 30 % clean episodes; on the student's obstacle features only, teacher exact)
#   A3 : identical round 3 without the localisation error (control: is a difference due to the error or to the
#        extra training?)
# Then: standard evaluation (nominal / randomised x clean / push) and, for A, A3 and B: ONNX export, parity check,
# sim2sim matrix and the perception-bias / placement / drift sweeps.
PY=/root/gpufree-data/venvs/mjlab/bin/python
PYD=/root/gpufree-data/venvs/deploy/bin/python
H=/root/gpufree-data/hnu_v2
N=$H/runs/night
A=v6g10s
cd $H/code
export HF_HUB_OFFLINE=1 HNU_BRIDGE_ROOT=$H/data/ref_bridge_video_lift HNU_STAKE_H=0.10 HNU_GEOM_W1=$H/data/ref_stakes_s93_h10 \
  HNU_GEOM_W2=$H/data/ref_stakes_s83_h10 HNU_STAKE_TOL=0.17 HNU_START_FRAC=0.25 HNU_DR_LEVEL=1 HNU_DR_LAG_SYNC=1 \
  HNU_DR_LAG_SENSORS=1 PYTHONPATH=$H/code
log() { echo "[$(date +%H:%M:%S)] $*" >> $N/v6g10sB.log; }
log "waiting for $A (finished + selection + post-analysis)"
for i in $(seq 1 3000); do
  grep -q finished $N/$A.log 2>/dev/null && [ -f $H/runs/results_${A}_selection.json ] && \
    grep -q done $N/after_$A.md 2>/dev/null && break
  sleep 60
done
SEL=$H/runs/results_${A}_selection.json
CK=$(cat $N/${A}_teachers.txt)
COMMON="--scene geom --z-offsets 0,5,10,15,20,25,30,40,50 --hidden 1024,512,256 --hist 20 --lin-vel"
seed_of() { $PY -c "import json; print(json.load(open('$SEL'))['$1']['selected_seed'])"; }
one() {  # variant(B|A3) P gpu
  local VAR=$1 P=$2 G=$3 s=$(seed_of $2) PE=0
  [ $VAR = B ] && PE=1
  local O=$H/runs/students_${A}${VAR}/$P
  [ -f $O/student_2000.pt ] || HNU_PERC_ERR=$PE HNU_DR=1 HNU_TB_ROOT=$H/runs/logs/rsl_rl/students_${A}${VAR} CUDA_VISIBLE_DEVICES=$G \
    $PY distill_train.py --profile ${A}${VAR}_$P $CK --num-envs 4096 --iters 2000 --beta-iters 1 --lr 2e-4 --action-noise 0.1 \
    --fail-weight 2 --seed 3000 $COMMON --init-from $H/runs/students_${A}_d2_s$s/$P/student_2000.pt \
    --latent-dir $H/data/z_tables_v5/$P --out $O > $N/${A}${VAR}_$P.log 2>&1
  [ -f $O/student_2000.pt ] || { log "$VAR $P: training failed"; return; }
  for DRV in 0 1; do for X in "" "--push"; do
    HNU_DR=$DRV HNU_START_FRAC=0 CUDA_VISIBLE_DEVICES=$G $PY eval_student.py --student $O/student_2000.pt \
      --latent-dir $H/data/z_tables_v5/$P --per-clip 8 $X --out $H/runs/results_${A}${VAR}_dr$DRV --tag _$P >> $N/${A}${VAR}_eval_$P.log 2>&1
  done; done
  log "$VAR $P (from seed $s) done"
}
JOBS=(); for VAR in B A3; do for P in Z0 Z1 Z2 Z3 Z4; do JOBS+=("$VAR $P"); done; done
queue() { for i in "${!JOBS[@]}"; do [ $((i % 2)) = $1 ] && one ${JOBS[$i]} $1; done; }
queue 0 & queue 1 & wait
log "students done"
# export + parity + sim2sim (matrix and sweeps) for A (selected), A3 and B
for P in Z0 Z1 Z2 Z3 Z4; do
  s=$(seed_of $P)
  for VAR in A A3 B; do
    if [ $VAR = A ]; then ST=$H/runs/students_${A}_d2_s$s/$P/student_2000.pt; E=$H/deploy_exports/$A/L${P#Z}_s$s
    else ST=$H/runs/students_${A}${VAR}/$P/student_2000.pt; E=$H/deploy_exports/${A}${VAR}/L${P#Z}; fi
    [ -f $ST ] || continue
    mkdir -p $E
    [ -f $E/policy.onnx ] || CUDA_VISIBLE_DEVICES=0 $PY deploy/export_student.py --student $ST --latent-dir $H/data/z_tables_v5/$P \
      --out $E > $E/export.log 2>&1
    $PYD deploy/parity_check.py $E > $E/parity.txt 2>&1
    [ -f $E/sim2sim_summary.txt ] || bash deploy/run_sim2sim.sh $E 48 > $E/sim2sim.log 2>&1
    for SW in perception:perc_sweep placement:placement_sweep drift:drift_sweep setup:setup_sweep; do
      [ -f $E/${SW%%:*}/summary.txt ] || bash deploy/${SW##*:}.sh $E 48 >> $E/sweeps.log 2>&1
    done
  done
done
$PY tools/compare_AB.py > $N/v6g10sB_comparison.md 2>&1
log "finished"
