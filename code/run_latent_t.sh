#!/bin/bash
# Latents for the TRIAL-LEVEL held-out protocol: identical to latent_v5 (same hyper-parameters, levels,
# seeds) except the split -- train trials only, val trials for model selection, test trials never seen
# (--split-mode file, data/clip_split_trial/split.json).  Then: z tables (seed 42, as v5), router v2 fine-tuning +
# query-space abstention on the new latents, Results 8.1 on the new latents, and the student batch v6g10t.
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
N=$H/runs/night
V=v5t
cd $H/code
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HNU_CTRL_AUX_DIR=$H/data/ctrl_aux_v5 \
  HNU_LATENT_SPLIT_JSON=$H/data/clip_split_trial/split.json
log() { echo "[$(date +%H:%M:%S)] $*" >> $N/latent_$V.log; }
LARGS="--epochs 200 --lambda-skill-metric 0.10 --lambda-skill-atoms 0.08 --lambda-human-skill-atoms 0.0 --lambda-human-sem-atoms 0.10 --text-encoder minilm --text-dim 384 --query-augmentation 0.5 --lambda-query-task 0.2 --query-task-stopgrad 1 --latent-dim 32 --lambda-future 0.3 --lambda-human-ctrl 0.3 --split-mode file"
mkdir -p $H/runs/latent_$V
latent() {  # P seed gpu
  [ -f $H/runs/latent_$V/seed$2/$1/train_latents.npz ] && return
  CUDA_VISIBLE_DEVICES=$3 ANNOT=annot_v2 nice -n 5 bash stage4_latent/train_latent.sh $1 $H/runs/latent_$V/seed$2 --seed $2 $LARGS \
    > $H/runs/latent_$V/train_$1_seed$2.log 2>&1
  log "latent $1 seed $2: $([ -f $H/runs/latent_$V/seed$2/$1/train_latents.npz ] && echo ok || echo FAILED)"
}
log "start latents"
JOBS=(); for s in 42 43 44; do for P in Z0 Z1 Z2 Z3 Z4; do JOBS+=("$P $s"); done; done
Q=$N/latent_${V}_queue; echo 0 > $Q.next; : > $Q.lock
lworker() {  # gpu
  while true; do
    local i=$(flock $Q.lock bash -c "i=\$(cat $Q.next); echo \$((i + 1)) > $Q.next; echo \$i")
    [ $i -ge ${#JOBS[@]} ] && return
    latent ${JOBS[$i]} $1
  done
}
lworker 0 & lworker 0 & lworker 0 & lworker 1 & lworker 1 & lworker 1 & wait
n=$(ls $H/runs/latent_$V/seed*/Z*/train_latents.npz 2>/dev/null | wc -l)
log "latents done: $n of 15"
[ "$n" = 15 ] || { log "STOP: latent training incomplete"; exit 1; }
for P in Z0 Z1 Z2 Z3 Z4; do
  ANNOT=annot_v2 PYTHONPATH=$H/data/data_code CUDA_VISIBLE_DEVICES=0 $PY stage4_latent/export_z_tables.py --profile $P \
    --latent-root $H/runs/latent_$V/seed42 --motion-root $H/data/ref_v1 --out $H/data/z_tables_$V/$P > $N/z_tables_${V}_$P.log 2>&1
done
log "z tables exported: $(ls -d $H/data/z_tables_$V/Z* | wc -l) levels"
# router v2 + abstention on the new latents (same procedure as for latent_v5)
CUDA_VISIBLE_DEVICES= PYTHONPATH=$H/data/data_code $PY stage4_latent/router_finetune2.py data_lang/instructions_v1.json \
  data_lang/instructions_val_v2.json results/router_ood_benchmark_v1.json $H/runs/latent_$V > $N/router_ft2_$V.log 2>&1
HNU_ABSTAIN_SPACE=query HNU_INSTRUCTIONS=data_lang/instructions_v1.json CUDA_VISIBLE_DEVICES= PYTHONPATH=$H/data/data_code \
  $PY stage4_latent/router_abstain.py data_lang/instructions_val_v2.json results/router_ood_benchmark_v1.json $H/runs/latent_$V ft2 \
  > $N/router_abstain_$V.log 2>&1
CUDA_VISIBLE_DEVICES= PYTHONPATH=$H/data/data_code $PY stage4_latent/final_router_eval.py data_lang/router_test_v2_sealed.json \
  data_lang/instructions_v1.json data_lang/instructions_val_v2.json $H/runs/latent_$V > $N/router_test_v2_$V.md 2>&1
log "router done"
# Results 8.1 on the new latents (CPU)
for s in 42 43 44; do
  R=$(CUDA_VISIBLE_DEVICES= bash tools/setup_paper_root.sh runs/latent_$V/seed$s data/annot_v2 | tail -n 1)
  CUDA_VISIBLE_DEVICES= bash tools/run_offline_81.sh $R > $N/offline81_${V}_seed$s.log 2>&1
done
$PY tools/aggregate_81.py $H/paper_root > $N/offline81_summary_all.md 2>&1
log "8.1 done"
log "finished"
