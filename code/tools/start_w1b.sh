#!/bin/bash
# Stop the W1 'full' fine-tune (its settings degrade the long-stride trials) and start W1 round 2 from the
# evaluated checkpoint model_52000 (runs/tf2_full_teachers.txt).  DuMu / W2 are not touched.
H=/root/gpufree-data/hnu_v2
for p in $(pgrep -f "experiment-name hnu_track_ThreeStepPoleRun --agent.run-name tf2_full"); do kill $p; done
sleep 15
pgrep -af "hnu_track_ThreeStepPoleRun" | grep -v start_w1b || echo "W1 full stopped"
grep -o "ckpt-ThreeStepPoleRun [^ ]*" $H/runs/tf2_full_teachers.txt
cd $H/code
SRC_FILE=$H/runs/tf2_full_teachers.txt setsid nohup bash tools/tf_v2.sh w1b 4000 W1B > $H/runs/tf2_w1b.out 2>&1 < /dev/null &
sleep 150
wc -l $H/data/clip_split_trial/teacher_w1_allow.txt
grep -E "Learning iteration|Error|Traceback" $H/runs/logs/ThreeStepPoleRun_tf2_w1b.log | tail -2
