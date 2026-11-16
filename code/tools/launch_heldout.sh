#!/bin/bash
# 1 check the training-clip filter, 2 cancel the queued all-clip jobs (no-latent, look-ahead), 3 start v6g10h + monitor
H=/root/gpufree-data/hnu_v2
cd $H/code
HNU_CLIPS_ALLOW=$H/data/clip_split_v5/clips_train.txt /root/gpufree-data/venvs/mjlab/bin/python - << 'EOF'
from hnu_tasks.multi_motion import resolve_motion_files
H = "/root/gpufree-data/hnu_v2"
spec = ",".join([f"{H}/data/ref_stakes_s93_h10/ThreeStepPoleRun", f"{H}/data/ref_stakes_s83_h10/FiveStepPoleRun",
                 f"{H}/data/ref_bridge_video_lift/CrouchedTraverse"])
fs = [f.rsplit("/", 1)[1].replace("_motion.npz", "") for f in resolve_motion_files(spec)]
test = set(open(f"{H}/data/clip_split_v5/clips_test.txt").read().split())
val = set(open(f"{H}/data/clip_split_v5/clips_val.txt").read().split())
print(f"filtered training clips: {len(fs)}; test clips among them: {len(test & set(fs))}; val clips among them: {len(val & set(fs))}")
EOF
for p in $(pgrep -f "bash run_nolatent.sh"); do kill $p; done
for p in $(pgrep -f "bash run_lookahead.sh"); do kill $p; done
echo "[$(date +%H:%M:%S)] cancelled: replaced by the held-out protocol (run_v6g10h.sh)" >> $H/runs/night/nolatent.log
echo "[$(date +%H:%M:%S)] cancelled: replaced by the held-out protocol (run_v6g10h.sh)" >> $H/runs/night/v6g10s_la0.log
setsid nohup bash run_v6g10h.sh > /dev/null 2>&1 < /dev/null &
sleep 3
CUDA_VISIBLE_DEVICES= setsid nohup /root/gpufree-data/venvs/mjlab/bin/python tools/monitor_tb.py v6g10h > $H/runs/night/monitor_v6g10h.out 2>&1 < /dev/null &
sleep 2
pgrep -af "run_v6g10h|run_nolatent|run_lookahead|monitor_tb.py v6g10h" | cut -c1-100
cat $H/runs/night/v6g10h.log
