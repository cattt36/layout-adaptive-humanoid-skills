#!/bin/bash
# automatic follow-up on the L40 (user request): (1) when the old-student completion has ended
# (tools/old_complete_l40_now.sh), rerun 8.4 new vs old with all old students; (2) when the ZR seeds 2-3 are trained and
# evaluated (run_zr_l40_s23.sh), compare ZR with L0 / L3 over seeds 1-3 (tools/zr_compare.py).
# Log runs/night/auto_B.log
H=/root/gpufree-data/hnu_v2
N=$H/runs/night
PY=/root/gpufree-data/venvs/mjlab/bin/python
cd $H/code
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $N/auto_B.log; }
log "start (waiting for the old-student completion)"
until grep -q "old_complete: all old students trained and evaluated under condition B" $N/v6g10n.log; do sleep 300; done
cp $H/runs/new_vs_old.md $H/runs/new_vs_old_partial_B.md
$PY tools/new_vs_old.py > $H/runs/new_vs_old.md.tmp 2>> $N/auto_B.log && mv $H/runs/new_vs_old.md.tmp $H/runs/new_vs_old.md
log "8.4 with all old students -> runs/new_vs_old.md (previous: new_vs_old_partial_B.md)"
until grep -q "ZR seeds 2-3: all folds complete" $N/zr.log; do sleep 300; done
HNU_SEEDS="1 2 3" $PY tools/zr_compare.py > $H/runs/zr_compare.md 2>> $N/auto_B.log
log "ZR vs L0 / L3 over seeds 1-3 -> runs/zr_compare.md"
