#!/bin/bash
# L40: one student per GPU (GPU 0) vs two students on one GPU (GPU 1) -- iterations per minute after ~8 min.
# Scratch runs in runs/bench_l40 (not used for anything else).
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
B=$H/runs/bench_l40
mkdir -p $B
cd $H/code
export HF_HUB_OFFLINE=1 HNU_BRIDGE_ROOT=$H/data/ref_bridge_video_lift HNU_STAKE_H=0.10 HNU_GEOM_W1=$H/data/ref_stakes_s93_h10 \
  HNU_GEOM_W2=$H/data/ref_stakes_s83_h10 HNU_STAKE_TOL=0.17 HNU_START_FRAC=0.25 HNU_DR_LEVEL=1 HNU_DR_LAG_SYNC=1 \
  HNU_DR_LAG_SENSORS=1 PYTHONPATH=$H/code HNU_STAKE_SHAPE=mesh16 HNU_POST_SHAPE=mesh12
AUG="HNU_Z_NOISE=0.3 HNU_Z_JITTER=3 HNU_STAKE_JITTER=0 HNU_STAKE_JITTER_ALONG=0.03 HNU_STAKE_JITTER_LAT=0.06 HNU_STAKE_STRAIGHT=0.3 HNU_STAKE_CONTACT=1 HNU_BRIDGE_JITTER_XY=0.05 HNU_BRIDGE_JITTER_ZLO=-0.05 HNU_BRIDGE_JITTER_ZHI=0.06 HNU_BRIDGE_POST_JITTER=0.10 HNU_RESET_SCALE=1.5 HNU_PUSH_SCALE=1.0 HNU_TASK_WEIGHT=CrouchedTraverse:1.5 HNU_TS_SHARE=0.5 HNU_GEOM_W1=$H/data/aug_ts_f1/ref_stakes_s93_h10 HNU_GEOM_W2=$H/data/aug_ts_f1/ref_stakes_s83_h10 HNU_BRIDGE_ROOT=$H/data/aug_ts_f1/ref_bridge_video_lift"
CK=$(cat $H/runs/night/v6g10n_teachers.txt)
run() {  # gpu tag
  env $AUG HNU_CLIPS_ALLOW=$H/data/clip_split_cv/fold1/clips_train_aug.txt HNU_DR=1 HNU_TB_ROOT=$B/tb_$2 CUDA_VISIBLE_DEVICES=$1 \
    $PY distill_train.py --profile bench_$2 $CK --num-envs 4096 --iters 4000 --seed 1000 --scene geom --hidden 1024,512,256 \
    --hist 20 --lin-vel --z-offsets 0,5,10,15,20,25,30,40,50 --latent-dir $H/data/z_tables_cv_f1_aug/Z4 --out $B/$2 > $B/$2.log 2>&1 &
}
run 0 single; run 1 pairA; run 1 pairB
sleep 600
for t in single pairA pairB; do echo "$t: $(tail -1 $B/$t/train_log.jsonl 2>/dev/null | cut -c1-200)"; done
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
pkill -f "profile bench_"
