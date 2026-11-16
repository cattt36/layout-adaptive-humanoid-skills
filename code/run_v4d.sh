#!/bin/bash
# v4d (final candidate): v4c + stop-gradient on z_skill in the routing loss, i.e.
#   z_skill <- motion + embodiment-matched supervision (robot metrics / robot atoms), anchor for every alignment
#   z_sem   <- task text, robot/human semantics (+ human-only atom head), routing to z_skill (z_skill frozen)
#   text front-end: frozen multilingual MiniLM, query augmentation with each level's own descriptions.
# All five levels retrained (3 latent seeds), students x 3 RL seeds, eval, Q_exec, offline 8.1, language routing.
set -u
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
N=$H/runs/night
V=v4d
cd $H/code
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export HNU_TB_ROOT=$H/runs/logs/rsl_rl/students_$V
log() { echo "[$(date +%H:%M:%S)] $*" >> $N/$V.log; }
is_done() { [ -f $N/${V}_$1.done ]; }
done_mark() { touch $N/${V}_$1.done; }
ck() { cat $H/runs/expert_eval/$1/checkpoint.txt; }
CK="--ckpt-ThreeStepPoleRun $(ck ThreeStepPoleRun) --ckpt-FiveStepPoleRun $(ck FiveStepPoleRun) --ckpt-CrouchedTraverse $(ck CrouchedTraverse)"
LARGS="--epochs 200 --lambda-skill-metric 0.10 --lambda-skill-atoms 0.08 --lambda-human-skill-atoms 0.0 --lambda-human-sem-atoms 0.10 --text-encoder minilm --text-dim 384 --query-augmentation 0.5 --lambda-query-task 0.2 --query-task-stopgrad 1"
latent() {  # P S gpu
  local P=$1 S=$2 g=$3
  [ -f $H/runs/latent_$V/seed$S/$P/train_latents.npz ] && return
  CUDA_VISIBLE_DEVICES=$g ANNOT=annot_v2 nice -n 10 bash stage4_latent/train_latent.sh $P $H/runs/latent_$V/seed$S --seed $S $LARGS \
    > $H/runs/latent_$V/train_${P}_seed$S.log 2>&1
}
student() {  # P gpu rlseed
  local P=$1 g=$2 s=$3 out=$H/runs/students_$V/$P
  [ $s != 1 ] && out=$H/runs/students_${V}_s$s/$P
  [ -f $out/student_2500.pt ] && return
  CUDA_VISIBLE_DEVICES=$g $PY distill_train.py --profile ${V}_$P $CK --num-envs 4096 --iters 2500 --seed $((1000 * s)) \
    --latent-dir $H/data/z_tables_$V/$P --out $out > $N/${V}_E_${P}_s$s.log 2>&1
}
evaluate() {  # P gpu rlseed
  local P=$1 g=$2 s=$3 out=$H/runs/students_$V/$P res=$H/runs/results_$V
  [ $s != 1 ] && out=$H/runs/students_${V}_s$s/$P && res=$H/runs/results_${V}_s$s
  mkdir -p $res
  for X in "" "--push"; do
    CUDA_VISIBLE_DEVICES=$g $PY eval_student.py --student $out/student_2500.pt --latent-dir $H/data/z_tables_$V/$P \
      --per-clip 8 $X --out $res --tag _$P >> $N/${V}_F_${P}_s$s.log 2>&1
  done
}

mkdir -p $H/runs/latent_$V
if ! is_done C; then
  log "C: 15 latents (GPU0: seeds 43/44, GPU1: seed 42)"
  for P in Z0 Z1 Z2 Z3 Z4; do latent $P 42 1 & latent $P 43 0 & latent $P 44 0 & done; wait
  for S in 42 43 44; do for P in Z0 Z1 Z2 Z3 Z4; do latent $P $S 0; done; done  # retry failures one by one
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
for s in 1 2 3; do
  if ! is_done E$s; then
    log "E$s: students (RL seed $s)"
    student Z0 0 $s & student Z1 0 $s & student Z2 0 $s & student Z3 1 $s & student Z4 1 $s & wait
    for P in Z0 Z1 Z2 Z3 Z4; do student $P 0 $s; done  # retry failures one by one
    i=0; for P in Z0 Z1 Z2 Z3 Z4; do evaluate $P $((i % 2)) $s & i=$((i + 1)); done; wait
    done_mark E$s
  fi
  if [ $s = 1 ] && ! is_done G; then
    log "G: Q_exec"
    cp $H/runs/results/eval_teacher*.json $H/runs/results/eval_student_ref*.json $H/runs/results/eval_student_nolatent*.json $H/runs/results_$V/
    i=0
    for P in Z0 Z1 Z2 Z3 Z4; do
      g=$((i % 2)); i=$((i + 1)); mkdir -p $H/runs/quality_$V/$P
      ( CUDA_VISIBLE_DEVICES=$g $PY stage5_quality/quality.py collect --policy student --student $H/runs/students_$V/$P/student_2500.pt \
          --latent-dir $H/data/z_tables_$V/$P --num-envs 2048 --steps 1500 --out $H/runs/quality_$V/$P/rollouts.npz > $N/${V}_G_$P.log 2>&1
        CUDA_VISIBLE_DEVICES=$g $PY stage5_quality/quality.py train --data $H/runs/quality_$V/$P/rollouts.npz \
          --heuristic-dir $H/data/heuristic --out $H/runs/quality_$V/$P >> $N/${V}_G_$P.log 2>&1 ) &
      [ $((i % 2)) = 0 ] && wait
    done
    wait; done_mark G
    log "H: offline 8.1 + language"
    export HNU_TEXT_ENCODER=minilm
    for S in 42 43 44; do
      R=$(bash tools/setup_paper_root.sh runs/latent_$V/seed$S data/annot_v2 2>/dev/null | tail -1)
      bash tools/run_offline_81.sh $R > $N/${V}_H_$S.log 2>&1
    done
    unset HNU_TEXT_ENCODER
    $PY tools/aggregate_81.py $H/paper_root > $N/offline_81_all.md 2>&1
    PYTHONPATH=$H/data/data_code $PY tools/eval_language.py results/router_ood_benchmark_v1.json $H/runs/latent_v3 $H/runs/latent_v4b $H/runs/latent_$V > $N/language_all.md 2>&1
    $PY tools/make_tables.py $H/runs/results_$V $H/runs/quality_$V > $N/tables_$V.md 2>&1
  fi
done
$PY tools/aggregate_seeds.py $H/runs/results_$V $H/runs/results_${V}_s2 $H/runs/results_${V}_s3 > $N/tables_${V}_seeds.md 2>&1
log "$V chain finished"
