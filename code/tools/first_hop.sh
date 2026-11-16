#!/bin/bash
# DEMO: first hop onto stake 1, vK 3PR 012 (works on the robot) vs 5PR 018 (does not take off), REPn / REALn
# as in tools/pr_compare.sh, 32 traced trials each -> bmJ/${FHO:-fhop}/summary.txt (E5=<export> ONLY5=1: one 5PR model)
PYD=/root/gpufree-data/venvs/deploy/bin/python
H=/root/gpufree-data/hnu_v2; B=/root/gpufree-data/robot_sim_check/bmJ; O=$B/${FHO:-fhop}; mkdir -p $O
cd $H/code
export HF_HUB_OFFLINE=1 PYTHONPATH=$H/code HNU_POST_SHAPE=mesh12 HNU_STAKE_SHAPE=mesh16 HNU_PERCEPTION=1 \
  HNU_STAKE_R=0.20 HNU_STAKE_H=0.10 HNU_STAKE_TOL=0.17 HNU_STAKE_CONTACT=1 HNU_RUNUP_FREE_S=0.8 HNU_S2S_TORQUE=1
ST="--start rest --hold 12 --start-rp 3"
run() {  # tag export layout task cond args...
  local t=$1 E=$2 L=$3 T=$4 k=$5; shift 5
  rm -rf $O/tr_${t}_$k; mkdir -p $O/tr_${t}_$k
  HNU_S2S_TRACE=$O/tr_${t}_$k $PYD deploy/sim2sim.py $E --crit task --trials 32 --workers 16 --tasks $T --motion-dirs $L \
    --latent-dir $L "$@" --out $O/${t}_$k.json > $O/${t}_$k.log 2>&1
}
E3=${E3:-$H/deploy_exports/rtJbx/rtJbx_012}; L3=$B/stk_final/012r
E5=${E5:-$H/deploy_exports/rtJbx/rtJbx_018}; L5=$B/stk_final/018B
P3="3PR $E3 $L3 ThreeStepPoleRun"; [ -n "$ONLY5" ] && P3=""
for p in ${P3:+"$P3"} "5PR $E5 $L5 FiveStepPoleRun"; do
  set -- $p
  run $1 $2 $3 $4 REPn $ST --start-xy 0.01 --start-yaw 1 --latency 0-20 --no-base-rand &
  run $1 $2 $3 $4 REALn $ST --start-xy 0.02 --start-yaw 3 --yaw-comp --yaw-meas-noise 1 --latency 0-40 --dr-x &
done
wait
{ [ -n "$ONLY5" ] || $PYD tools/first_hop.py $E3 $L3 ThreeStepPoleRun $O/tr_3PR_REPn $O/tr_3PR_REALn
  $PYD tools/first_hop.py $E5 $L5 FiveStepPoleRun $O/tr_5PR_REPn $O/tr_5PR_REALn; } > $O/summary.txt 2>&1
cat $O/summary.txt
