#!/bin/bash
# driver: wait for tools/tf_base.sh -> gate (tools/tf_base_gate.sh) -> student stage (tools/st_base.sh)
# only if a CrouchedTraverse teacher passes; otherwise stop and wait for a decision.  Log runs/night/tf_base.log
H=/root/gpufree-data/hnu_v2; N=$H/runs/night
cd $H/code
log() { echo "[$(date '+%m-%d %H:%M:%S')] auto: $*" >> $N/tf_base.log; }
until grep -q "all trained" $N/tf_base.log 2>/dev/null; do sleep 120; done
bash tools/tf_base_gate.sh
if grep -q "CT_CHOICE [AB]" $H/runs/tf_base_gate.md; then
  log "CT teacher passes -> student stage"
  bash tools/st_base.sh
  log "student stage finished"
else
  log "STOP: no CrouchedTraverse teacher passes the gate (runs/tf_base_gate.md); student stage NOT started"
fi
