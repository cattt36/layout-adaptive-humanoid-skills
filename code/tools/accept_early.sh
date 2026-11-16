#!/bin/bash
# Early acceptance of the W2 and CrouchedTraverse teachers (tf2_full) while W1 round 2 is still training:
#  1. criterion-v2 evaluation, 4 conditions (tf2_eval.sh early full; the W1 numbers in it are an old interim -> ignore)
#  2. retention vs the original experts (W2: stakes_flat vs the flat expert; DuMu: bridge vs bridge_video_lift_perc)
#  3. tempo copies the students will train on: W2 0.90 / 1.10x, DuMu 0.95 / 1.05x (nominal, 8 rollouts per clip)
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
cd $H/code
for i in $(seq 1 60); do
  ls $H/runs/logs/rsl_rl/hnu_track_CrouchedTraverse/*_tf2_full/model_42993.pt > /dev/null 2>&1 && break
  grep -q "CrouchedTraverse tf2_full finished" $H/runs/tf2_full.out && break
  sleep 60
done
echo "== 1. criterion v2 (W1 = old interim, ignore)"
bash tools/tf2_eval.sh early full
W2=$(ls $H/runs/logs/rsl_rl/hnu_track_FiveStepPoleRun/*_tf2_full/model_*.pt | sort -V | tail -1)
DU=$(ls $H/runs/logs/rsl_rl/hnu_track_CrouchedTraverse/*_tf2_full/model_*.pt | sort -V | tail -1)
echo "== 2. retention"
CUDA_VISIBLE_DEVICES=0 $PY tools/retention_check.py --task FiveStepPoleRun --mode stakes_flat --perc-tuned 1 \
  --orig $(cat $H/runs/expert_eval/FiveStepPoleRun/checkpoint.txt) --tuned $W2 &
CUDA_VISIBLE_DEVICES=1 $PY tools/retention_check.py --task CrouchedTraverse --mode bridge --perc-orig 1 --perc-tuned 1 \
  --orig $(cat $H/runs/bridge_eval/bridge_video_lift_perc/checkpoint.txt) --tuned $DU &
wait
echo "== 3. tempo copies (nominal, criterion v2): W2 at 0.90 / 1.10x, DuMu at 0.95 / 1.05x"
export HF_HUB_OFFLINE=1 HNU_STAKE_H=0.10 HNU_STAKE_TOL=0.17 HNU_DR_LEVEL=1 HNU_DR_LAG_SYNC=1 HNU_DR_LAG_SENSORS=1 \
  PYTHONPATH=$H/code HNU_SPLIT_JSON=$H/data/clip_split_trial/split_v2.json HNU_EVAL_CRIT=task HNU_START_FRAC=0 HNU_DR=0
CK=$(cat $H/runs/tf2_early_teachers.txt)
G=0
for T in 090 095 105 110; do
  R=$H/data/ts/ts$T
  HNU_GEOM_W1=$R/ref_stakes_s93_h10 HNU_GEOM_W2=$R/ref_stakes_s83_h10 HNU_BRIDGE_ROOT=$R/ref_bridge_video_lift \
    CUDA_VISIBLE_DEVICES=$G $PY eval_student.py --teacher $CK --scene geom --z-offsets 0,5,10,15,20,25,30,40,50 \
    --latent-dir $R/z_tables_v5t/Z4 --per-clip 8 --out $H/runs/tf2_early_ts/ts$T --tag _teacher > /dev/null 2>&1 &
  G=$((1 - G))
done
wait
$PY - << 'EOF'
import json
H = "/root/gpufree-data/hnu_v2/runs"
for t, sp in (("FiveStepPoleRun", ("090", "110")), ("CrouchedTraverse", ("095", "105"))):
    for s in sp:
        r = json.load(open(f"{H}/tf2_early_ts/ts{s}/eval_teacher_teacher.json"))["summary"][f"{t}/all"]
        print(f"{t} x{int(s) / 100:.2f}: v2 success {100 * r['success_v2']:.0f} %  (old criterion {100 * r['success']:.0f} %)")
EOF
