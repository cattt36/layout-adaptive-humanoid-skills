#!/bin/bash
# Test of the nominal-env fix: DAgger round 3 (1000 its) of seed-1 students Z0 / Z3 with synchronised latency
# (HNU_DR_LAG_SYNC=1: one 0-20 ms lag per env shared by all observation terms), then nominal / latency evaluation.
# 1024 envs (fits next to the running v6g10 student on the GPU).
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
cd $H/code
export HF_HUB_OFFLINE=1 HNU_BRIDGE_ROOT=$H/data/ref_bridge_video_lift HNU_STAKE_H=0.10 HNU_GEOM_W1=$H/data/ref_stakes_s93_h10 \
  HNU_GEOM_W2=$H/data/ref_stakes_s83_h10 HNU_STAKE_TOL=0.17 HNU_DR_LEVEL=1 HNU_DR_LAG_SYNC=1 HNU_DR_LAG_SENSORS=${SENS:-0}
CK=$(cat $H/runs/night/v6g10_teachers.txt)
COMMON="--scene geom --z-offsets 0,5,10,15,20,25,30,40,50 --hidden 1024,512,256 --hist 20 --lin-vel"
G=${G:-1}
O=$H/runs/test_sync_ft${SENS:+_sens}; mkdir -p $O
for P in ${PS:-Z3 Z0}; do
  [ -f $O/$P/student_1000.pt ] || HNU_DR=1 HNU_START_FRAC=0.25 CUDA_VISIBLE_DEVICES=$G $PY distill_train.py --profile sync_$P $CK \
    --num-envs 1024 --iters 1000 --beta-iters 1 --lr 2e-4 --action-noise 0.1 --fail-weight 2 --seed 3000 $COMMON \
    --init-from $H/runs/students_v6g10_d2_s1/$P/student_2000.pt --latent-dir $H/data/z_tables_v5/$P --out $O/$P > $O/train_$P.log 2>&1
  for C in "nom 0 dyn 0 0" "lag1 1 lag 1 1" "lag01 1 lag 0 1" "full 1 dyn,lag 0 1"; do
    set -- $C
    [ -f $O/eval_$1/eval_student_$P.json ] || HNU_DR=$2 HNU_DR_PARTS=$3 HNU_DR_LAG_MIN=$4 HNU_DR_LAG_MAX=$5 HNU_START_FRAC=0 \
      CUDA_VISIBLE_DEVICES=$G $PY eval_student.py --student $O/$P/student_1000.pt --latent-dir $H/data/z_tables_v5/$P --per-clip 8 \
      --out $O/eval_$1 --tag _$P >> $O/eval.log 2>&1
    # the old round-2 student under the same synchronised latency (fair comparison)
    [ $1 = nom ] || [ -f $O/old_$1/eval_student_$P.json ] || HNU_DR=$2 HNU_DR_PARTS=$3 HNU_DR_LAG_MIN=$4 HNU_DR_LAG_MAX=$5 \
      HNU_START_FRAC=0 CUDA_VISIBLE_DEVICES=$G $PY eval_student.py --student $H/runs/students_v6g10_d2_s1/$P/student_2000.pt \
      --latent-dir $H/data/z_tables_v5/$P --per-clip 8 --out $O/old_$1 --tag _$P >> $O/eval.log 2>&1
  done
done
$PY - <<'EOF'
import json, os
O = "/root/gpufree-data/hnu_v2/runs/test_sync_ft" + ("_sens" if os.environ.get("SENS") else "")
R = "/root/gpufree-data/hnu_v2/runs/results_v6g10_s1_dr0"
for T in ("CrouchedTraverse", "ThreeStepPoleRun", "FiveStepPoleRun"):
    for P in ("Z0", "Z3"):
        def s(d):
            f = f"{d}/eval_student_{P}.json"
            return f"{100 * json.load(open(f))['summary'][T + '/all']['success']:.1f}" if os.path.exists(f) else "-"
        print(f"{T} {P}  old: nom {s(R)} lag1 {s(O + '/old_lag1')} lag01 {s(O + '/old_lag01')} full {s(O + '/old_full')}"
              f"   new: nom {s(O + '/eval_nom')} lag1 {s(O + '/eval_lag1')} lag01 {s(O + '/eval_lag01')} full {s(O + '/eval_full')}")
EOF
