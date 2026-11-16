#!/bin/bash
# v5b: human semantic atoms also supervise z_skill (lambda_human_skill_atoms = 0.05, pre-set small weight, same rule
# for every level).  Only L4 owns human semantic atoms, so Z0-Z3 are unchanged by construction and are not retrained;
# Z4 is retrained (3 latent seeds), its students are trained with RL seeds 1-3 (v5 DAgger 4000 + round 2 2000) and
# compared pairwise with v5 L4 / L3 on the same seeds.  Starts after run_v5_seeds.sh.
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
N=$H/runs/night
V=v5b
cd $H/code
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export HNU_CTRL_AUX_DIR=$H/data/ctrl_aux_v5 HNU_BRIDGE_ROOT=$H/data/ref_bridge_video_lift
log() { echo "[$(date +%H:%M:%S)] $*" >> $N/$V.log; }
for i in $(seq 1 1500); do grep -q finished $N/v5_seeds.log 2>/dev/null && break; sleep 60; done
log "start"
LARGS="--epochs 200 --lambda-skill-metric 0.10 --lambda-skill-atoms 0.08 --lambda-human-skill-atoms 0.05 --lambda-human-sem-atoms 0.10 --text-encoder minilm --text-dim 384 --query-augmentation 0.5 --lambda-query-task 0.2 --query-task-stopgrad 1 --latent-dim 32 --lambda-future 0.3 --lambda-human-ctrl 0.3"
mkdir -p $H/runs/latent_$V
for S in 42 43 44; do
  [ -f $H/runs/latent_$V/seed$S/Z4/train_latents.npz ] || \
  ( CUDA_VISIBLE_DEVICES=$(( S % 2 )) ANNOT=annot_v2 nice -n 10 bash stage4_latent/train_latent.sh Z4 $H/runs/latent_$V/seed$S --seed $S $LARGS \
      > $H/runs/latent_$V/train_Z4_seed$S.log 2>&1 ) &
done
wait
ANNOT=annot_v2 PYTHONPATH=$H/data/data_code CUDA_VISIBLE_DEVICES=0 $PY stage4_latent/export_z_tables.py --profile Z4 \
  --latent-root $H/runs/latent_$V/seed42 --motion-root $H/data/ref_v1 --out $H/data/z_tables_$V/Z4 > $N/${V}_D_Z4.log 2>&1
ck() { cat $H/runs/expert_eval/$1/checkpoint.txt; }
CKD=$(cat $H/runs/bridge_eval/bridge_video_lift_perc/checkpoint.txt)
CK="--ckpt-ThreeStepPoleRun $(ck ThreeStepPoleRun) --ckpt-FiveStepPoleRun $(ck FiveStepPoleRun) --ckpt-CrouchedTraverse $CKD"
ZOFF=0,5,10,15,20,25,30,40,50
one() {  # seed gpu
  local s=$1 G=$2 S=$H/runs/students_${V}_s$1/Z4 O=$H/runs/students_${V}_d2_s$1/Z4
  [ -f $S/student_4000.pt ] || CUDA_VISIBLE_DEVICES=$G $PY distill_train.py --profile ${V}_Z4 $CK --num-envs 4096 --iters 4000 \
    --seed $((1000 * s)) --scene bridge --z-offsets $ZOFF --hidden 1024,512,256 --latent-dir $H/data/z_tables_$V/Z4 --out $S \
    > $N/${V}_s${s}_E.log 2>&1
  [ -f $O/student_2000.pt ] || CUDA_VISIBLE_DEVICES=$G $PY distill_train.py --profile ${V}d2_Z4 $CK --num-envs 4096 --iters 2000 \
    --beta-iters 1 --lr 2e-4 --action-noise 0.1 --seed $((2000 * s)) --scene bridge --z-offsets $ZOFF --hidden 1024,512,256 \
    --init-from $S/student_4000.pt --latent-dir $H/data/z_tables_$V/Z4 --out $O > $N/${V}_d2_s${s}.log 2>&1
  for X in "" "--push"; do
    CUDA_VISIBLE_DEVICES=$G $PY eval_student.py --student $S/student_4000.pt --latent-dir $H/data/z_tables_$V/Z4 --per-clip 8 $X \
      --out $H/runs/results_${V}_s$s --tag _Z4 >> $N/${V}_F_s$s.log 2>&1
    CUDA_VISIBLE_DEVICES=$G $PY eval_student.py --student $O/student_2000.pt --latent-dir $H/data/z_tables_$V/Z4 --per-clip 8 $X \
      --out $H/runs/results_${V}_d2_s$s --tag _Z4 >> $N/${V}_F_s$s.log 2>&1
  done
}
one 1 0 & one 2 1 & one 3 0 & wait
for s in 1 2 3; do one $s 1; done
PYTHONPATH=$H/data/data_code $PY tools/eval_language.py results/router_ood_benchmark_v1.json $H/runs/latent_v5 $H/runs/latent_$V \
  > $N/language_${V}.md 2>&1
log "finished"
