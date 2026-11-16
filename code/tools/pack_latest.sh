#!/bin/bash
# Pack the latest results / tables / annotations / final checkpoints for the local copy (no intermediate checkpoints).
H=/root/gpufree-data/hnu_v2
OUT=/root/gpufree-data/sync_latest
rm -rf $OUT; mkdir -p $OUT
cd $H
# 1. regenerated annotations (robot parts) and scene data
mkdir -p $OUT/data
cp -r data/annot_v2 $OUT/data/
cp -r data/scene $OUT/data/ 2>/dev/null; rm -f $OUT/data/scene/dumu_capture.mp4
for d in bridge_v1 bridge_usd_lift bridge_video bridge_video_lift; do [ -d data/$d ] && cp -r data/$d $OUT/data/; done
mkdir -p $OUT/data/ctrl_aux_v5 && cp data/ctrl_aux_v5/meta.json $OUT/data/ctrl_aux_v5/
# 2. results, tables, logs summaries
mkdir -p $OUT/runs
for d in results results_v3 results_v3_s2 results_v4b results_v4d results_v4d_s2 results_v4d_s3 results_bridge results_v5 results_v5_d2 \
         results_v5_s2 results_v5_s3 results_v5_d2_s2 results_v5_d2_s3 stakes_eval bridge_eval quality_aux_v4d quality_aux_scarce_v4d; do
  [ -d runs/$d ] && find runs/$d -type f \( -name '*.json' -o -name '*.md' -o -name '*.txt' -o -name '*.log' \) -exec cp --parents {} $OUT/ \;
done
for d in quality_v3 quality_v4b quality_v4d quality; do
  [ -d runs/$d ] && find runs/$d -type f \( -name '*.json' -o -name 'quality_net.pt' \) -exec cp --parents {} $OUT/ \;
done
mkdir -p $OUT/runs/night && cp runs/night/*.md runs/night/*.log $OUT/runs/night/ 2>/dev/null
# 3. final checkpoints only
mkdir -p $OUT/checkpoints/experts
for T in ThreeStepPoleRun FiveStepPoleRun CrouchedTraverse; do cp $(cat runs/expert_eval/$T/checkpoint.txt) $OUT/checkpoints/experts/${T}_flat.pt; done
cp $(cat runs/bridge_eval/bridge_video_lift_perc/checkpoint.txt) $OUT/checkpoints/experts/CrouchedTraverse_bridge_perception.pt
for T in ThreeStepPoleRun FiveStepPoleRun; do
  RD=$(ls -d runs/logs/rsl_rl/hnu_track_$T/*_perc_h10 | tail -1); cp $(ls $RD/model_*.pt | sort -V | tail -1) $OUT/checkpoints/experts/${T}_stakes_perception_h10.pt
done
for v in students_v4d:2500 students_bridge:2500 students_v5:4000 students_v5_d2:2000; do
  d=${v%%:*}; it=${v##*:}
  for P in Z0 Z1 Z2 Z3 Z4; do
    [ -f runs/$d/$P/student_$it.pt ] && mkdir -p $OUT/checkpoints/$d && cp runs/$d/$P/student_$it.pt $OUT/checkpoints/$d/${P}.pt
  done
done
for v in latent_v4d latent_v5; do
  for P in Z0 Z1 Z2 Z3 Z4; do
    f=runs/$v/seed42/$P/best_motion_latent.pt
    [ -f $f ] && mkdir -p $OUT/checkpoints/$v && cp $f $OUT/checkpoints/$v/${P}_seed42.pt
  done
done
# 4. z tables used by the students (small), videos
mkdir -p $OUT/data/z_tables && cp -r data/z_tables_v4d data/z_tables_v5 $OUT/data/z_tables/ 2>/dev/null
cp -r videos $OUT/ 2>/dev/null
cd /root/gpufree-data && tar czf sync_latest.tgz sync_latest && du -sh sync_latest sync_latest.tgz
