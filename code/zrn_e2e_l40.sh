#!/bin/bash
# E2E candidate sweep for the explicit-reference students WITHOUT rhythm decoupling ("zrn", run_zrn_l40.sh), L40,
# .  Same sweep as tools/zr_e2e.sh for the decoupled ER ("zr"): planner candidate tables of level ZR, 16
# rollouts per (test layout, candidate), nominal, criterion v2, per-rollout outcomes (HNU_EVAL_EPISODES) -> preview on
# rollouts 1-4, execution on 9-16.  The E2E data of the 4090 live in hnu_v2_4090, so the sweep reads and writes there,
# and the pooling uses the path-adjusted copies tools_l40/ (validated against runs/e2e_refusal_v4_zr.md).
# A GPU is used only when all 12 training jobs are claimed and no training job of run_zrn_l40.sh is open on it.
# Pools: router v2 and v4 with HNU_WHO_EXTRA=zr,zrn; primary comparison ER-dec - ER-conv (HNU_REF_WHO=zr, 10,000
# replicates, runs/zrn_analysis_plan.md).  Log runs/night/zrn_e2e.log (hnu_v2)
PY=/root/gpufree-data/venvs/mjlab/bin/python
PYA=/root/gpufree-data/venvs/ana/bin/python  # pooling (needs scikit-learn; rows of zr reproduced byte-identically )
H=/root/gpufree-data/hnu_v2
H4=/root/gpufree-data/hnu_v2_4090
D=$H4/data
N=$H/runs/night
cd $H/code
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HNU_BRIDGE_ROOT=$H/data/ref_bridge_video_lift HNU_STAKE_H=0.10 HNU_STAKE_TOL=0.17 \
  HNU_START_FRAC=0.25 HNU_DR_LEVEL=1 HNU_DR_LAG_SYNC=1 HNU_DR_LAG_SENSORS=1 PYTHONPATH=$H/code HNU_STAKE_SHAPE=mesh16 HNU_POST_SHAPE=mesh12
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $N/zrn_e2e.log; }
gpu_free() {  # gpu: all training jobs claimed and none of them open on this gpu
  [ $(ls $N/zrn_claims 2>/dev/null | wc -l) -ge 12 ] || return 1
  local open=$(awk -v g="$1" '/start on gpu/ {split($0, a, "fold "); k=substr(a[2], 1, index(a[2], ":") - 1); if ($NF == g) o[k]=1}
                             /trained and evaluated|FAILED/ {split($0, a, "fold "); k=a[2]; sub(/ (trained|round).*/, "", k); delete o[k]}
                             END {n=0; for (k in o) n++; print n}' $N/zrn.log)
  [ "$open" = 0 ]
}
sweep() {  # fold seed gpu
  local F=$1 S=$2 G=$3 FS=""; [ $F != 1 ] && FS=_f$F
  local ST=$H/runs/students_v6g10n${FS}_d2_s$S/ZR/student_2000.pt
  local SPL=$D/clip_split_cv/fold$F/split.json; [ $F = 1 ] && SPL=$D/clip_split_trial/split_v2.json
  local K=$(python3 -c "import json;print(json.load(open('$D/z_cand_plan_f$F/pairs.json'))['K'])")
  for k in $(seq -f %02g 0 $((K - 1))); do
    local O=$H4/runs/results_candplan_zrn${FS}_k${k}_s${S}_dr0 GK=$D/geom_cand_f$F/k$k
    if [ ! -f $O/eval_student_ZR.json ]; then
      HNU_SPLIT_JSON=$SPL HNU_GEOM_W1=$GK/ref_stakes_s93_h10 HNU_GEOM_W2=$GK/ref_stakes_s83_h10 HNU_EVAL_EPISODES=1 HNU_EVAL_CRIT=task \
        HNU_DR=0 HNU_START_FRAC=0 CUDA_VISIBLE_DEVICES=$G $PY eval_student.py --student $ST --latent-dir $D/z_cand_plan_f$F/k$k/ZR \
        --per-clip 16 --out $O --tag _ZR >> $N/zrn_e2e_eval.log 2>&1
    fi
    if [ -f $O/eval_student_ZR.json ]; then
      for P in Z0 Z1 Z2 Z3; do ln -sf eval_student_ZR.json $O/eval_student_$P.json; [ -f $O/qexec_ZR.npz ] && ln -sf qexec_ZR.npz $O/qexec_$P.npz; done
    else
      log "FAILED f$F s$S k$k"
    fi
  done
}
JOBS=()
for S in 1 2 3; do for F in 1 2 3 4; do JOBS+=("$F $S"); done; done
Q=$N/zrn_e2e_claims
rm -rf $Q; mkdir -p $Q
worker() {  # gpu
  local G=$1
  while true; do
    local left=0 took=0
    for i in "${!JOBS[@]}"; do
      set -- ${JOBS[$i]}
      [ -f $Q/done_$i ] && continue
      left=1
      local FS=""; [ $1 != 1 ] && FS=_f$1
      [ -f $H/runs/students_v6g10n${FS}_d2_s$2/ZR/student_2000.pt.ok ] || continue
      gpu_free $G || break
      mkdir $Q/claim_$i 2> /dev/null || continue
      log "start f$1 s$2 (gpu $G)"
      sweep $1 $2 $G
      touch $Q/done_$i
      log "fold $1 seed $2 swept"
      took=1
      break
    done
    [ $left = 0 ] && return
    [ $took = 0 ] && sleep 300
  done
}
log "start"
for G in 0 1 2 3 4; do worker $G & sleep 10; done
wait
log "all sweeps done, pooling"
cd $H4/code
export PYTHONPATH=$D/data_code CUDA_VISIBLE_DEVICES= HNU_WHO_EXTRA=zr,zrn
for RT in "" v4; do
  for S in 1 2 3; do for F in 1 2 3 4; do HNU_RTAG=$RT nice $PYA tools_l40/expA_plan.py $S $F > /dev/null 2>> $N/zrn_e2e.log & done; done
  wait
  HNU_RTAG=$RT nice $PYA tools_l40/expA_pool.py > $H4/runs/expA_pool${RT:+_$RT}_zr_zrn.md 2>> $N/zrn_e2e.log
done
HNU_RTAG=v4 HNU_REF_WHO=zr nice $PYA tools_l40/e2e_refusal_pool.py > $H4/runs/e2e_refusal_v4_zr_zrn.md 2>> $N/zrn_e2e.log &
HNU_RTAG=v4 HNU_WHO_ONLY=zr,zrn HNU_REF_WHO=zr HNU_REPS=10000 nice $PYA tools_l40/e2e_refusal_pool.py \
  > $H4/runs/e2e_refusal_v4_zrn_primary.md 2>> $N/zrn_e2e.log &
wait
for S in 1 2 3; do $PYA tools_l40/preview_select.py zrn $S > $H4/runs/preview_zrn_s$S.md 2>> $N/zrn_e2e.log; done
log "all steps done -> hnu_v2_4090/runs/e2e_refusal_v4_zr_zrn.md, expA_pool[_v4]_zr_zrn.md, preview_zrn_s*.md"
