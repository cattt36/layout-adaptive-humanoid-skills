#!/bin/bash
# CV data preparation for folds 2-4 (pre-registered , runs/preregistration_cv.md), in parallel with v6g10n:
#  per fold: 15 latents (Z0-Z4 x seeds 42/43/44, same hyper-parameters as latent_v5t, --split-mode file with the fold
#  split) -> leak check -> z tables (seed 42) -> encoder z of the tempo copies -> training set (make_aug_ts f<k>) ->
#  offline 8.1 metrics (3 latent seeds).   Log: runs/night/cv_prep.log
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
N=$H/runs/night
cd $H/code
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HNU_CTRL_AUX_DIR=$H/data/ctrl_aux_v5
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $N/cv_prep.log; }
LARGS="--epochs 200 --lambda-skill-metric 0.10 --lambda-skill-atoms 0.08 --lambda-human-skill-atoms 0.0 --lambda-human-sem-atoms 0.10 --text-encoder minilm --text-dim 384 --query-augmentation 0.5 --lambda-query-task 0.2 --query-task-stopgrad 1 --latent-dim 32 --lambda-future 0.3 --lambda-human-ctrl 0.3 --split-mode file"
latent() {  # fold P seed gpu
  local F=$1 P=$2 S=$3 G=$4 R=$H/runs/latent_cv_f$1
  [ -f $R/seed$S/$P/train_latents.npz ] && return
  mkdir -p $R
  HNU_LATENT_SPLIT_JSON=$H/data/clip_split_cv/fold$F/split.json CUDA_VISIBLE_DEVICES=$G ANNOT=annot_v2 nice -n 5 \
    bash stage4_latent/train_latent.sh $P $R/seed$S --seed $S $LARGS > $R/train_${P}_seed$S.log 2>&1
  log "fold $F latent $P seed $S: $([ -f $R/seed$S/$P/train_latents.npz ] && echo ok || echo FAILED)"
}
log "start: latents for folds 2-4 (45)"
JOBS=(); for F in 2 3 4; do for S in 42 43 44; do for P in Z0 Z1 Z2 Z3 Z4; do JOBS+=("$F $P $S"); done; done; done
Q=$N/cv_prep_queue; echo 0 > $Q.next; : > $Q.lock
lworker() {
  while true; do
    local i=$(flock $Q.lock bash -c "i=\$(cat $Q.next); echo \$((i + 1)) > $Q.next; echo \$i")
    [ $i -ge ${#JOBS[@]} ] && return
    latent ${JOBS[$i]} $1
  done
}
lworker 0 & lworker 0 & lworker 1 & lworker 1 & wait
n=$(ls $H/runs/latent_cv_f*/seed*/Z*/train_latents.npz 2>/dev/null | wc -l)
log "latents done: $n of 45"
[ "$n" = 45 ] || { log "STOP: latent training incomplete"; exit 1; }
for F in 2 3 4; do
  for f in $H/runs/latent_cv_f$F/seed*/Z*/split_manifest.json; do
    log "leak check $f: $($PY tools/check_cv_latent.py $f $H/data/clip_split_cv/fold$F/split.json)"
  done
  for P in Z0 Z1 Z2 Z3 Z4; do
    ANNOT=annot_v2 PYTHONPATH=$H/data/data_code CUDA_VISIBLE_DEVICES=0 $PY stage4_latent/export_z_tables.py --profile $P \
      --latent-root $H/runs/latent_cv_f$F/seed42 --motion-root $H/data/ref_v1 --out $H/data/z_tables_cv_f$F/$P > $N/cv_z_f${F}_$P.log 2>&1
    ANNOT=annot_v2 PYTHONPATH=$H/data/data_code CUDA_VISIBLE_DEVICES=0 $PY stage4_latent/export_z_tables_ts.py --profile $P \
      --latent-root $H/runs/latent_cv_f$F/seed42 --motion-root $H/data/ref_v1 --speeds 0.9,0.95,1.05,1.1 \
      --only ThreeStepPoleRun,FiveStepPoleRun --out-root $H/data/ts --zname z_enc_f$F > $N/cv_zts_f${F}_$P.log 2>&1
    ANNOT=annot_v2 PYTHONPATH=$H/data/data_code CUDA_VISIBLE_DEVICES=0 $PY stage4_latent/export_z_tables_ts.py --profile $P \
      --latent-root $H/runs/latent_cv_f$F/seed42 --motion-root $H/data/ref_v1 --speeds 0.95,1.05 \
      --only CrouchedTraverse --out-root $H/data/ts --zname z_enc_f$F >> $N/cv_zts_f${F}_$P.log 2>&1
  done
  log "fold $F: z tables + tempo z exported ($(ls $H/data/z_tables_cv_f$F | wc -l) levels)"
  log "fold $F training set: $($PY tools/make_aug_ts.py f$F | tr '\n' ' ')"
done
# offline 8.1 per fold and latent seed (CPU)
for F in 2 3 4; do for S in 42 43 44; do
  R=$(CUDA_VISIBLE_DEVICES= bash tools/setup_paper_root.sh runs/latent_cv_f$F/seed$S data/annot_v2 | tail -n 1)
  HNU_LATENT_SPLIT_JSON=$H/data/clip_split_cv/fold$F/split.json CUDA_VISIBLE_DEVICES= bash tools/run_offline_81.sh $R \
    > $N/cv_offline81_f${F}_seed$S.log 2>&1
done; done
log "offline 8.1 done for folds 2-4"
log "finished"
