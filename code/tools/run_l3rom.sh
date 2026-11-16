#!/bin/bash
# 4090 server, (user decision): "L3-rom" ablation = L3-kin without any human gait-timing signal (only human joint ranges of motion, step length, obstacle relation).  Derived from tools/run_l3rom.sh; everything else identical.
# (original header of run_l3kin.sh:) "L3-kin" ablation = L3 without any human force / moment signal
# (HNU_HUMAN_DYNAMICS=0, stage4_latent/motion_latent_z.py): human kinematics, step info and gait timing kept.
# Identical to the L3 latents in everything else: 4 folds x latent seeds 42 / 43 / 44, the same hyper-parameters
# (LARGS of run_latent_t.sh / run_cv_prep.sh) and the same split files (fold 1 = clip_split_trial/split.json as
# latent_v5t; folds 2-4 = clip_split_cv/fold<k>/split.json).  Offline only (no student).
#   1. train 12 latents (runs/latent_rom_f<k>/seed<s>/Z3), leak check
#   2. complete each seed directory with links to the standard Z0 / Z1 / Z2 / Z4 of the same fold and seed (the
#      offline scripts need all five; their Z0-Z2 rows must reproduce the standard values = self-check)
#   3. offline 8.1 (paper_root/latent_rom_f<k>_seed<s>), HNU_HUMAN_DYNAMICS=0 for the Z3 validators only
#   4. z tables (data/z_tables_rom_f<k>_s<s>/Z3) and linear probes (runs/probe_z_rom.json)
#   5. comparison table L2 / L3-rom / L3-kin / L3 (tools/aggregate_human.py -> runs/l3rom_summary.md)
# Log: runs/night/l3rom.log
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
N=$H/runs/night
cd $H/code
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HNU_CTRL_AUX_DIR=$H/data/ctrl_aux_v5
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $N/l3rom.log; }
LARGS="--epochs 200 --lambda-skill-metric 0.10 --lambda-skill-atoms 0.08 --lambda-human-skill-atoms 0.0 --lambda-human-sem-atoms 0.10 --text-encoder minilm --text-dim 384 --query-augmentation 0.5 --lambda-query-task 0.2 --query-task-stopgrad 1 --latent-dim 32 --lambda-future 0.3 --lambda-human-ctrl 0.3 --split-mode file"
split_of() { [ $1 = 1 ] && echo $H/data/clip_split_trial/split.json || echo $H/data/clip_split_cv/fold$1/split.json; }
std_of() { [ $1 = 1 ] && echo $H/runs/latent_v5t/seed$2 || echo $H/runs/latent_cv_f$1/seed$2; }
latent() {  # fold seed gpu
  local F=$1 S=$2 G=$3 R=$H/runs/latent_rom_f$1
  [ -f $R/seed$S/Z3/train_latents.npz ] && return
  mkdir -p $R
  HNU_HUMAN_DYNAMICS=0 HNU_HUMAN_TIMING=0 HNU_LATENT_SPLIT_JSON=$(split_of $F) CUDA_VISIBLE_DEVICES=$G ANNOT=annot_v2 nice -n 5 \
    bash stage4_latent/train_latent.sh Z3 $R/seed$S --seed $S $LARGS > $R/train_Z3_seed$S.log 2>&1
  log "L3-rom latent fold $F seed $S: $([ -f $R/seed$S/Z3/train_latents.npz ] && echo ok || echo FAILED)"
}
log "start: 12 L3-rom latents"
JOBS=(); for F in 1 2 3 4; do for S in 42 43 44; do JOBS+=("$F $S"); done; done
Q=$N/l3rom_queue; echo 0 > $Q.next; : > $Q.lock
lworker() {
  while true; do
    local i=$(flock $Q.lock bash -c "i=\$(cat $Q.next); echo \$((i + 1)) > $Q.next; echo \$i")
    [ $i -ge ${#JOBS[@]} ] && return
    latent ${JOBS[$i]} $1
  done
}
lworker 0 & lworker 0 & lworker 1 & lworker 1 & wait
n=$(ls $H/runs/latent_rom_f*/seed*/Z3/train_latents.npz 2>/dev/null | wc -l)
log "L3-rom latents done: $n of 12"
[ "$n" = 12 ] || { log "STOP: latent training incomplete"; exit 1; }
grep -l "HNU_HUMAN_TIMING=0" $H/runs/latent_rom_f*/train_Z3_seed*.log | wc -l | xargs -I{} bash -c "echo '[\$(date +%m-%d\ %H:%M:%S)] training logs confirming the switch: {} of 12' >> $N/l3rom.log"
for F in 1 2 3 4; do for S in 42 43 44; do
  D=$H/runs/latent_rom_f$F/seed$S
  log "leak check f$F s$S: $($PY tools/check_cv_latent.py $D/Z3/split_manifest.json $(split_of $F))"
  for P in Z0 Z1 Z2 Z4; do [ -e $D/$P ] || ln -s $(std_of $F $S)/$P $D/$P; done
done; done
# offline 8.1 (CPU), 3 in parallel
off() {  # fold seed
  local R=$(CUDA_VISIBLE_DEVICES= bash tools/setup_paper_root.sh runs/latent_rom_f$1/seed$2 data/annot_v2 | tail -n 1)
  Z3_TIMING=0 HNU_LATENT_SPLIT_JSON=$(split_of $1) CUDA_VISIBLE_DEVICES= bash tools/run_offline_81_kin.sh $R > $N/l3rom_offline81_f$1_seed$2.log 2>&1
  log "offline 8.1 f$1 s$2: $(ls $R/Results/8.1_Offline_Representation/outputs/offline_latent_metrics.csv 2>/dev/null | wc -l) metrics file"
}
i=0; for F in 1 2 3 4; do for S in 42 43 44; do off $F $S & i=$((i + 1)); [ $((i % 3)) = 0 ] && wait; done; done; wait
# z tables (checkpoint normalizers) and probes
for F in 1 2 3 4; do for S in 42 43 44; do
  HNU_HUMAN_DYNAMICS=0 HNU_HUMAN_TIMING=0 ANNOT=annot_v2 PYTHONPATH=$H/data/data_code CUDA_VISIBLE_DEVICES=0 timeout 3600 $PY stage4_latent/export_z_tables.py \
    --profile Z3 --latent-root $H/runs/latent_rom_f$F/seed$S --motion-root $H/data/ref_v1 --out $H/data/z_tables_rom_f${F}_s$S/Z3 \
    > $N/l3rom_z_f${F}_s$S.log 2>&1
done; done
log "z tables: $(grep -l 'normalizers: checkpoint' $N/l3rom_z_f*_s*.log | wc -l) of 12 with checkpoint normalizers"
HNU_PROBE_LEVELS=Z3R HNU_PROBE_OUT=$H/runs/probe_z_rom.json CUDA_VISIBLE_DEVICES= $PY tools/probe_z_cv.py > $N/probe_z_rom.txt 2>&1
log "probes done"
$PY tools/aggregate_human.py > $H/runs/l3rom_summary.md 2>> $N/l3rom.log
log "finished -> runs/l3rom_summary.md"
