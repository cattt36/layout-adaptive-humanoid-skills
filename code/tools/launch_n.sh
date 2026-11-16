#!/bin/bash
# Launch v6g10n (evening): stop the unneeded round-2 bridge training (du3_B2), write the final teacher bank
# (W1 rs10c, W2 tf2_full, DuMu du3_B: passed the pre-decided bridge gate), start the pipeline, monitor and watchdog.
H=/root/gpufree-data/hnu_v2
N=$H/runs/night
for p in $(pgrep -f "run-name du3_B2"); do kill $p; done
sleep 5
B=$(ls $H/runs/logs/rsl_rl/hnu_track_CrouchedTraverse/*_du3_B/model_*.pt | sort -V | tail -1)
BASE=$(cat $N/v6g10n_teachers_proposed.txt)
OLD=$(echo $BASE | sed 's/ /\n/g' | grep -A1 -- "--ckpt-CrouchedTraverse" | tail -1)
echo "${BASE/$OLD/$B}" > $N/v6g10n_teachers.txt
echo "[$(date '+%m-%d %H:%M')] gate parser bug (push file name) fixed by hand: A and B both PASS; B chosen (beam -5 cm 95 %); round 2 stopped; bank: $(cat $N/v6g10n_teachers.txt)" >> $N/du_ft_auto.log
cat $N/v6g10n_teachers.txt
cd $H/code
setsid nohup bash run_v6g10n.sh > $N/run_v6g10n.out 2>&1 < /dev/null &
sleep 10
bash tools/restart_monitor.sh v6g10n
setsid nohup bash tools/watchdog.sh v6g10n > /dev/null 2>&1 < /dev/null &
sleep 5
echo "--- batch log"; cat $N/v6g10n.log
pgrep -af "run_v6g10n|watchdog.sh v6g10n|monitor_tb.py v6g10n|eval_student" | grep -v launch_n | cut -c1-100
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
