#!/bin/bash
# 8.3.2 for all new student pairs (L0 / L3 x folds 1-4 x RL seeds 1-3), (user decision).  4090 server.
# Waits for the first run (tools/deploy_v2_4090.sh = fold 1 seed 1), then runs tools/deploy_v2_run.sh for every
# (fold, seed) whose L0 and L3 students have been relayed from the L40, one at a time (CPU bound), and refreshes the
# pooled table runs/deploy_pool.md (tools/deploy_pool.py) after each.  Log runs/night/deploy_v2.log
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
N=$H/runs/night
cd $H/code
log() { echo "[$(date '+%m-%d %H:%M:%S')] all: $*" >> $N/deploy_v2.log; }
for i in $(seq 1 600); do grep -q "all done -> runs/deploy_sweeps_v6g10xm_f1_s1.md" $N/deploy_v2.log && break; sleep 60; done
log "first pair done; expanding to all folds / seeds"
JOBS="1:1 2:1 3:1 4:1 1:2 2:2 3:2 4:2 1:3 2:3 3:3 4:3"
done_job() { grep -q "v6g10xm_f$1_s$2: all done" $N/deploy_v2.log || { [ "$1:$2" = "1:1" ] && return 0; return 1; }; }
for round in $(seq 1 400); do
  left=0; took=0
  for j in $JOBS; do
    F=${j%:*} S=${j#*:}
    done_job $F $S && continue
    left=1
    FS=""; [ $F != 1 ] && FS=_f$F
    ST=$H/runs/students_v6g10xm${FS}_d2_s$S
    [ -f $ST/Z0/student_2000.pt ] && [ -f $ST/Z3/student_2000.pt ] || continue
    sleep 30  # a relayed file settles
    FOLD=$F SEED=$S bash tools/deploy_v2_run.sh
    $PY tools/deploy_pool.py > $H/runs/deploy_pool.md 2>> $N/deploy_v2.log
    log "pooled table refreshed after f$F s$S"
    took=1
    break
  done
  [ $left = 0 ] && break
  [ $took = 0 ] && sleep 600
done
log "finished all pairs -> runs/deploy_pool.md"
