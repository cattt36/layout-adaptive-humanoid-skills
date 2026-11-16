#!/bin/bash
# Smoke test of the v6g10n DAgger settings (100 iterations, L4, GPU 1) with the interim teacher bank
# (W2 / DuMu accepted tf2_full, W1 interim): errors?  termination rate while the teacher drives (beta ~ 1) vs v6g10m.
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
cd $H/code
export HF_HUB_OFFLINE=1 HNU_BRIDGE_ROOT=$H/data/ref_bridge_video_lift HNU_STAKE_H=0.10 HNU_GEOM_W1=$H/data/ref_stakes_s93_h10 \
  HNU_GEOM_W2=$H/data/ref_stakes_s83_h10 HNU_STAKE_TOL=0.17 HNU_START_FRAC=0.25 HNU_DR_LEVEL=1 HNU_DR_LAG_SYNC=1 \
  HNU_DR_LAG_SENSORS=1 PYTHONPATH=$H/code HNU_SPLIT_JSON=$H/data/clip_split_trial/split_v2.json \
  HNU_STAKE_SHAPE=mesh16 HNU_POST_SHAPE=mesh12
AUG="HNU_Z_NOISE=0.3 HNU_Z_JITTER=3 HNU_STAKE_JITTER=0 HNU_STAKE_JITTER_ALONG=0.03 HNU_STAKE_JITTER_LAT=0.06 HNU_STAKE_STRAIGHT=0.3 HNU_STAKE_CONTACT=1 HNU_BRIDGE_JITTER_XY=0.05 HNU_BRIDGE_JITTER_Z=0.04 HNU_RESET_SCALE=1.5 HNU_PUSH_SCALE=1.0 HNU_TASK_WEIGHT=CrouchedTraverse:1.5 HNU_TS_SHARE=0.5 HNU_GEOM_W1=$H/data/aug_ts_n/ref_stakes_s93_h10 HNU_GEOM_W2=$H/data/aug_ts_n/ref_stakes_s83_h10 HNU_BRIDGE_ROOT=$H/data/aug_ts_n/ref_bridge_video_lift"
CK=$(cat $H/runs/night/v6g10n_teachers_proposed.txt)
env $AUG HNU_CLIPS_ALLOW=$H/data/clip_split_trial/clips_train_aug_n.txt HNU_DR=1 HNU_TB_ROOT=$H/runs/logs/rsl_rl/smoke_v6g10n \
  CUDA_VISIBLE_DEVICES=1 $PY distill_train.py --profile smoke_n_Z4 $CK --num-envs 4096 --iters 100 --seed 1 \
  --scene geom --hidden 1024,512,256 --hist 20 --lin-vel --z-offsets 0,5,10,15,20,25,30,40,50 \
  --latent-dir $H/data/z_tables_v5t_aug_n/Z4 --out $H/runs/smoke_v6g10n_prism/Z4 > $H/runs/smoke_v6g10n.log 2>&1
echo "== v6g10n smoke (first / last)"; grep '^{' $H/runs/smoke_v6g10n.log | head -2; grep '^{' $H/runs/smoke_v6g10n.log | tail -1
grep -ciE 'traceback|error' $H/runs/smoke_v6g10n.log
echo "== v6g10m s1 L4 (same iterations)"; grep '^{' $H/runs/night/v6g10m_s1_E_Z4.log | head -2; grep '^{' $H/runs/night/v6g10m_s1_E_Z4.log | sed -n '10p'
