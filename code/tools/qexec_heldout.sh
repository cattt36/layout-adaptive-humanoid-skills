#!/bin/bash
# 4090 server, (user request): Q_exec vs closed-loop execution success on the held-out clips (EXPLORATORY
# item 5), fold-1 students of RL seed $SEED (default 1).  Every level L0-L4 identically.
#  1. Q_exec re-trained on rollouts of the fold-1 TRAINING clips only (HNU_CLIPS_ALLOW = clips_train.txt, 51 clips).
#     The earlier runs/quality_v6g10n_s<seed> (run_sec_4090.sh step c) collected on all 86 clips incl. the test clips;
#     it is kept and evaluated alongside as a reference.  Otherwise identical to step c (collect / train / 2 fracs x 3).
#  2. closed-loop evaluation of the students on every clip with the Q_exec nets attached (eval_student.py,
#     HNU_EVAL_QEXEC; measurement only): nominal (dr0) and randomised + push (dr1 --push), criterion task / v2; plus
#     nominal with the training terminations (HNU_EVAL_CRIT=track), the label Q_exec is trained on
#  3. tables: tools/qexec_closed_loop.py -> runs/qexec_closed_loop_s<seed>.md
# Log: runs/night/qexec_heldout.log
SEED=${SEED:-1}
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
N=$H/runs/night
cd $H/code
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HNU_BRIDGE_ROOT=$H/data/ref_bridge_video_lift HNU_STAKE_H=0.10 \
  HNU_GEOM_W1=$H/data/ref_stakes_s93_h10 HNU_GEOM_W2=$H/data/ref_stakes_s83_h10 HNU_STAKE_TOL=0.17 HNU_START_FRAC=0.25 \
  HNU_DR_LEVEL=1 HNU_DR_LAG_SYNC=1 HNU_DR_LAG_SENSORS=1 PYTHONPATH=$H/code HNU_STAKE_SHAPE=mesh16 HNU_POST_SHAPE=mesh12 \
  HNU_SPLIT_JSON=$H/data/clip_split_trial/split_v2.json
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $N/qexec_heldout.log; }
S=$H/runs/students_v6g10n_d2_s$SEED
Q=$H/runs/quality_v6g10n_tr_s$SEED
Q0=$H/runs/quality_v6g10n_s$SEED
[ $(ls $S/Z{0,1,2,3}/student_2000.pt 2>/dev/null | wc -l) = 4 ] || { log "STOP: students missing"; exit 1; }
log "start seed $SEED (students $(md5sum $S/Z*/student_2000.pt | md5sum | cut -c1-8))"
lane() {  # gpu, levels
  local G=$1; shift
  for P in "$@"; do
    mkdir -p $Q/$P
    [ $(ls $Q/$P/frac*_s*/quality_eval.json 2>/dev/null | wc -l) = 6 ] || [ -f $Q/$P/rollouts.npz ] || \
      HNU_CLIPS_ALLOW=$H/data/clip_split_cv/fold1/clips_train.txt HNU_DR=1 CUDA_VISIBLE_DEVICES=$G \
        $PY stage5_quality/quality.py collect --policy student --student $S/$P/student_2000.pt \
        --latent-dir $H/data/z_tables_cv_f1/$P --num-envs 2048 --steps 1500 --out $Q/$P/rollouts.npz > $Q/$P/collect.log 2>&1
    for FR in 1.0 0.05; do for SS in 0 1 2; do
      [ -f $Q/$P/frac${FR}_s$SS/quality_eval.json ] || CUDA_VISIBLE_DEVICES=$G $PY stage5_quality/quality.py train \
        --data $Q/$P/rollouts.npz --heuristic-dir $H/data/heuristic --train-frac $FR --seed $SS --out $Q/$P/frac${FR}_s$SS \
        > $Q/$P/frac${FR}_s$SS.log 2>&1
    done; done
    n=$(ls $Q/$P/frac*_s*/quality_eval.json 2>/dev/null | wc -l)
    [ "$n" = 6 ] && rm -f $Q/$P/rollouts.npz
    log "Q_exec (training clips only) $P: $n/6 trainings, clips $(grep -c . $H/data/clip_split_cv/fold1/clips_train.txt)"
    [ "$n" = 6 ] || continue
    NETS=$(ls $Q/$P/frac1.0_s*/quality_net.pt $Q/$P/frac0.05_s*/quality_net.pt $Q0/$P/frac1.0_s*/quality_net.pt | paste -sd,)
    for c in "task 0 " "task 1 --push" "track 0 "; do  # track: the training terminations = the label of Q_exec
      set -- $c
      local O=$H/runs/results_v6g10n$([ $1 = track ] && echo trk)_Q_s${SEED}_dr$2
      [ -f $O/qexec_$P$([ -n "$3" ] && echo _push).npz ] && continue
      HNU_EVAL_QEXEC=$NETS HNU_EVAL_CRIT=$1 HNU_DR=$2 HNU_START_FRAC=0 CUDA_VISIBLE_DEVICES=$G $PY eval_student.py \
        --student $S/$P/student_2000.pt --latent-dir $H/data/z_tables_cv_f1/$P --per-clip 32 $3 --out $O --tag _$P \
        >> $N/qexec_eval_$P.log 2>&1
      log "closed loop $P $1 dr$2 $3: $([ -f $O/qexec_$P$([ -n "$3" ] && echo _push).npz ] && echo ok || echo FAILED)"
    done
  done
}
lane 0 Z0 Z1 & lane 1 Z3 Z2 & wait  # L4 dropped
CUDA_VISIBLE_DEVICES= $PY tools/qexec_closed_loop.py $SEED > $H/runs/qexec_closed_loop_s$SEED.md 2> $N/qexec_closed_loop.err
log "finished -> runs/qexec_closed_loop_s$SEED.md"
