#!/bin/bash
# 4090, after tools/rwt_sweep.sh (both variants): E2E analysis with the external baseline next to the new
# and old students, frozen router (v2) and router v4.  Outputs runs/expA_pool_rwt.md, runs/expA_pool_v4_rwt.md.
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
N=$H/runs/night
cd $H/code
export PYTHONPATH=$H/data/data_code CUDA_VISIBLE_DEVICES= HNU_WHO_EXTRA=rwtt,rwts
for i in $(seq 1 400); do
  grep -q "variant t all folds done" $N/rwt_sweep.log && grep -q "variant s all folds done" $N/rwt_sweep.log && break
  sleep 30
done
for RT in "" v4; do
  for F in 1 2 3 4; do
    FS=""; [ $F != 1 ] && FS=_f$F
    for S in 1 2 3; do
      [ -f $H/runs/expA_plan${FS}_s${S}_rows.json ] || continue
      HNU_RTAG=$RT nice $PY tools/expA_plan.py $S $F > $H/runs/expA_plan${FS}_s${S}${RT:+_$RT}_rwt.md 2>> $N/rwt_sweep.log &
    done
  done
  wait
  HNU_RTAG=$RT nice $PY tools/expA_pool.py > $H/runs/expA_pool${RT:+_$RT}_rwt.md 2>> $N/rwt_sweep.log
  echo "[$(date '+%m-%d %H:%M:%S')] analysis ${RT:-v2} -> runs/expA_pool${RT:+_$RT}_rwt.md" >> $N/rwt_sweep.log
done
echo "[$(date '+%m-%d %H:%M:%S')] rwt analysis complete" >> $N/rwt_sweep.log
