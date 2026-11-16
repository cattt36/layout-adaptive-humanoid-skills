#!/bin/bash
# 4090 server (CPU only), (user request): deployment tolerance sweeps for the exported fold-1 students
# (deploy_exports/<EXP>/L3 and L0), CPU MuJoCo sim2sim with 0-20 ms latency (+ the default friction / CoM / encoder
# randomisation of sim2sim), criterion task / v2, 8 trials per clip, every clip (tables report the TEST clips):
#   perception error : constant obstacle offset 2 / 5 / 8 / 12 cm, per-step noise 2 / 5 / 10 cm
#   placement error  : whole layout shifted 2 / 5 / 8 / 12 cm, every stake moved independently 2 / 5 / 8 cm (ThreeStepPoleRun / FiveStepPoleRun)
#   set-up error     : bridge 3 / 5 cm lower or higher (CrouchedTraverse), start heading 3 / 5 / 10 deg
#   odometry drift   : scale / heading error 1 / 2 / 5 / 10 % (no external tracking; supplementary)
# Same options as deploy/{perc,placement,setup,drift}_sweep.sh, plus --crit task and 8 trials.
# Log: runs/night/deploy_sweeps.log;  table: runs/deploy_sweeps_<EXP>.md (tools/sweep_table.py)
EXP=${EXP:-v6g10n_f1_s1}
W=${W:-96}
PY=/root/gpufree-data/venvs/deploy/bin/python
H=/root/gpufree-data/hnu_v2
N=$H/runs/night
cd $H/code
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $N/deploy_sweeps.log; }
run() {  # export_dir name args...
  local EX=$1 NAME=$2; shift 2
  mkdir -p $EX/sweeps
  [ -f $EX/sweeps/$NAME.json ] && return
  $PY deploy/sim2sim.py $EX --latency 0-20 --crit task --trials 8 --workers $W "$@" --out $EX/sweeps/$NAME.json \
    >> $EX/sweeps/sweeps.log 2>&1
  log "$(basename $EX) $NAME: $([ -f $EX/sweeps/$NAME.json ] && echo ok || echo FAILED)"
}
log "start $EXP"
for L in L3 L0; do
  EX=$H/deploy_exports/$EXP/$L
  run $EX base
  for b in 0.02 0.05 0.08 0.12; do run $EX perc_bias$b --perc-bias $b; done
  for n in 0.02 0.05 0.10; do run $EX perc_noise$n --perc-noise $n; done
  for b in 0.02 0.05 0.08 0.12; do run $EX shift$b --obst-shift $b; done
  for j in 0.02 0.05 0.08; do run $EX stake_jitter$j --tasks ThreeStepPoleRun,FiveStepPoleRun --stake-jitter $j; done
  for dz in -0.05 -0.03 0.03 0.05; do run $EX bridge_dz$dz --tasks CrouchedTraverse --bridge-dz $dz; done
  for y in 3 5 10; do run $EX yaw$y --start-yaw $y; done
  for r in 0.01 0.02 0.05 0.10; do run $EX drift$r --drift $r; done
done
$PY tools/sweep_table.py $EXP > $H/runs/deploy_sweeps_$EXP.md 2>> $N/deploy_sweeps.log
log "done $EXP -> runs/deploy_sweeps_$EXP.md"
