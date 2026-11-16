#!/bin/bash
# v5: closed-loop precision.  Same rules for every level; only the level's own supervision fields differ.
#  A  z_skill: 32-d, plus a future-motion prediction head (robot reference 0.25-1 s ahead: joints, feet, pelvis)
#  B  z_skill: human control-relevant targets (force-plate contact timing, time to touchdown / lift-off, GRF)
#     -- only levels that own human numeric data (L3, L4), enforced in motion_latent_z.compute_losses
#  C  students: denser z_skill sampling (9 offsets up to 1 s), larger MLP (1024-512-256), longer DAgger (4000)
# Scene: bridge (CrouchedTraverse under the capture-video bridge with obstacle features; W1/W2 flat for now).
# Baseline for comparison: v4d students in the same scene (run_bridge_students.sh).
set -u
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
N=$H/runs/night
V=v5
cd $H/code
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export HNU_CTRL_AUX_DIR=$H/data/ctrl_aux_v5 HNU_BRIDGE_ROOT=$H/data/ref_bridge_video_lift
export HNU_TB_ROOT=$H/runs/logs/rsl_rl/students_$V
log() { echo "[$(date +%H:%M:%S)] $*" >> $N/$V.log; }
is_done() { [ -f $N/${V}_$1.done ]; }
done_mark() { touch $N/${V}_$1.done; }
ck() { cat $H/runs/expert_eval/$1/checkpoint.txt; }
CKD=$(cat $H/runs/bridge_eval/bridge_video_lift_perc/checkpoint.txt)
CK="--ckpt-ThreeStepPoleRun $(ck ThreeStepPoleRun) --ckpt-FiveStepPoleRun $(ck FiveStepPoleRun) --ckpt-CrouchedTraverse $CKD"
LARGS="--epochs 200 --lambda-skill-metric 0.10 --lambda-skill-atoms 0.08 --lambda-human-skill-atoms 0.0 --lambda-human-sem-atoms 0.10 --text-encoder minilm --text-dim 384 --query-augmentation 0.5 --lambda-query-task 0.2 --query-task-stopgrad 1 --latent-dim 32 --lambda-future 0.3 --lambda-human-ctrl 0.3"
ZOFF=0,5,10,15,20,25,30,40,50
if ! is_done A; then
  log "A: control targets"
  $PY stage4_latent/make_ctrl_aux.py --ref $H/data/ref_v1 --phase $H/data/phase --forceplate $H/data/forceplate \
    --annot $H/data/annot_v2/LLM_annotation --out $HNU_CTRL_AUX_DIR > $N/${V}_A.log 2>&1 && done_mark A
fi
latent() {  # P S gpu
  local P=$1 S=$2 g=$3
  [ -f $H/runs/latent_$V/seed$S/$P/train_latents.npz ] && return
  CUDA_VISIBLE_DEVICES=$g ANNOT=annot_v2 nice -n 10 bash stage4_latent/train_latent.sh $P $H/runs/latent_$V/seed$S --seed $S $LARGS \
    > $H/runs/latent_$V/train_${P}_seed$S.log 2>&1
}
mkdir -p $H/runs/latent_$V
if ! is_done C; then
  log "C: 15 latents"
  for P in Z0 Z1 Z2 Z3 Z4; do latent $P 42 1 & latent $P 43 0 & latent $P 44 0 & done; wait
  for S in 42 43 44; do for P in Z0 Z1 Z2 Z3 Z4; do latent $P $S 0; done; done
  done_mark C
fi
if ! is_done D; then
  log "D: z tables"
  for P in Z0 Z1 Z2 Z3 Z4; do
    ANNOT=annot_v2 PYTHONPATH=$H/data/data_code CUDA_VISIBLE_DEVICES=0 $PY stage4_latent/export_z_tables.py --profile $P \
      --latent-root $H/runs/latent_$V/seed42 --motion-root $H/data/ref_v1 --out $H/data/z_tables_$V/$P > $N/${V}_D_$P.log 2>&1
  done
  done_mark D
fi
S=$H/runs/students_$V
R=$H/runs/results_$V
mkdir -p $S $R
student() {  # P gpu
  [ -f $S/$1/student_4000.pt ] && return
  CUDA_VISIBLE_DEVICES=$2 $PY distill_train.py --profile ${V}_$1 $CK --num-envs 4096 --iters 4000 --seed 1000 --scene bridge \
    --z-offsets $ZOFF --hidden 1024,512,256 --latent-dir $H/data/z_tables_$V/$1 --out $S/$1 > $N/${V}_E_$1.log 2>&1
}
if ! is_done E; then
  log "E: students"
  student Z0 0 & student Z1 0 & student Z2 1 & student Z3 1 & student Z4 0 & wait
  for P in Z0 Z1 Z2 Z3 Z4; do student $P 1; done
  log "F: eval"
  i=0
  for P in Z0 Z1 Z2 Z3 Z4; do
    ( for X in "" "--push"; do
        CUDA_VISIBLE_DEVICES=$((i % 2)) $PY eval_student.py --student $S/$P/student_4000.pt --latent-dir $H/data/z_tables_$V/$P \
          --per-clip 8 $X --out $R --tag _$P >> $N/${V}_F_$P.log 2>&1
      done ) &
    i=$((i + 1))
  done
  wait
  done_mark E
fi
if ! is_done H; then
  log "H: language routing + offline 8.1"
  PYTHONPATH=$H/data/data_code $PY tools/eval_language.py results/router_ood_benchmark_v1.json $H/runs/latent_v4d $H/runs/latent_$V \
    > $N/language_${V}.md 2>&1
  done_mark H
fi
log "$V finished"
