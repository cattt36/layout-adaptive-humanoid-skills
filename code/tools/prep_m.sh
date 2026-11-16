#!/bin/bash
# v6g10m: stop v6g10k, encoder z of the time-scaled CrouchedTraverse trials (0.95 / 1.05x), training set m, checks, launch.
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
N=$H/runs/night
for p in $(pgrep -f "run_v6g10k.sh"); do kill $p; done
for p in $(pgrep -f "tools/monitor_tb.py v6g10k"); do kill $p; done
sleep 2
for p in $(pgrep -f "students_v6g10k|results_v6g10k|profile v6g10k"); do kill $p; done
sleep 10
echo "left over v6g10k processes:"; pgrep -af "v6g10k" | grep -v prep_m || echo none
cd $H/code
for P in Z0 Z1 Z2 Z3 Z4; do
  ANNOT=annot_v2 PYTHONPATH=$H/data/data_code CUDA_VISIBLE_DEVICES=0 $PY stage4_latent/export_z_tables_ts.py --profile $P \
    --latent-root $H/runs/latent_v5t/seed42 --motion-root $H/data/ref_v1 --speeds 0.95,1.05 --only CrouchedTraverse \
    --out-root $H/data/ts > $N/z_enc_dumu_$P.log 2>&1 || { tail -20 $N/z_enc_dumu_$P.log; exit 1; }
  tail -2 $N/z_enc_dumu_$P.log
done
$PY tools/make_aug_ts.py m || exit 1
$PY tools/check_aug.py _m
setsid nohup bash run_v6g10m.sh > $N/run_v6g10m.out 2>&1 < /dev/null &
sleep 5
bash tools/restart_monitor.sh v6g10m
sleep 240
cat $N/v6g10m.log
for f in v6g10m_s1_E_Z0.log v6g10m_s1_E_Z4.log; do echo "== $f"; grep -ciE 'error|traceback' $N/$f; grep '^{' $N/$f | tail -1; done
