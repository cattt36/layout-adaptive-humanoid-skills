#!/bin/bash
# Pilot: can held-out students execute UNSEEN clips?  Held-out v6g10h students reach 85-93 % on training
# clips but 0 % on val / test clips (memorisation).  Three students, held-out protocol, shortened DAgger (2000 + 1000):
#   Z0A : L0 + z augmentation (uniform noise +-0.3 on every z sample, random time shift +-3 steps)
#   Z4A : L4 + the same augmentation
#   REF : explicit reference command instead of z_skill (no augmentation)
# Evaluation on all clips (nominal / randomised x clean / push); val / test splits reported.
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
N=$H/runs/night
V=v6g10p
cd $H/code
export HF_HUB_OFFLINE=1 HNU_BRIDGE_ROOT=$H/data/ref_bridge_video_lift HNU_STAKE_H=0.10 HNU_GEOM_W1=$H/data/ref_stakes_s93_h10 \
  HNU_GEOM_W2=$H/data/ref_stakes_s83_h10 HNU_STAKE_TOL=0.17 HNU_START_FRAC=0.25 HNU_DR_LEVEL=1 HNU_DR_LAG_SYNC=1 \
  HNU_DR_LAG_SENSORS=1 PYTHONPATH=$H/code
ALLOW=$H/data/clip_split_v5/clips_train.txt
CK=$(cat $N/v6g10h_teachers.txt)
log() { echo "[$(date +%H:%M:%S)] $*" >> $N/$V.log; }
BASE="--scene geom --z-offsets 0,5,10,15,20,25,30,40,50 --hidden 1024,512,256 --hist 20 --lin-vel"
one() {  # name gpu latent extra-env extra-args
  local P=$1 G=$2 LD=$H/data/z_tables_v5/$3 ENVX=$4 EXTRA=$5 s=1
  local S=$H/runs/students_${V}_s$s O=$H/runs/students_${V}_d2_s$s
  [ -f $S/$P/student_2000.pt ] || env $ENVX HNU_CLIPS_ALLOW=$ALLOW HNU_DR=1 HNU_TB_ROOT=$H/runs/logs/rsl_rl/students_${V}_s$s \
    CUDA_VISIBLE_DEVICES=$G $PY distill_train.py --profile ${V}_$P $CK --num-envs 4096 --iters 2000 --seed 1000 $BASE $EXTRA \
    --latent-dir $LD --out $S/$P > $N/${V}_E_$P.log 2>&1
  [ -f $S/$P/student_2000.pt ] || { log "$P round 1 failed"; return; }
  [ -f $O/$P/student_1000.pt ] || env $ENVX HNU_CLIPS_ALLOW=$ALLOW HNU_DR=1 HNU_TB_ROOT=$H/runs/logs/rsl_rl/students_${V}_d2_s$s \
    CUDA_VISIBLE_DEVICES=$G $PY distill_train.py --profile ${V}d2_$P $CK --num-envs 4096 --iters 1000 --beta-iters 1 --lr 2e-4 \
    --action-noise 0.1 --fail-weight 2 --seed 2000 $BASE $EXTRA --init-from $S/$P/student_2000.pt --latent-dir $LD \
    --out $O/$P > $N/${V}_d2_$P.log 2>&1
  [ -f $O/$P/student_1000.pt ] || { log "$P round 2 failed"; return; }
  for DRV in 0 1; do for X in "" "--push"; do
    HNU_DR=$DRV HNU_START_FRAC=0 CUDA_VISIBLE_DEVICES=$G $PY eval_student.py --student $O/$P/student_1000.pt \
      --latent-dir $LD --per-clip 8 $X --out $H/runs/results_${V}_s${s}_dr$DRV --tag _$P >> $N/${V}_F_$P.log 2>&1
  done; done
  log "$P done"
}
log "start"
AUG="HNU_Z_NOISE=0.3 HNU_Z_JITTER=3"
( one Z0A 0 Z0 "$AUG" "" ; one REF 0 Z0 "HNU_Z_NOISE=0" "--obs-group student_ref" ) &
one Z4A 1 Z4 "$AUG" "" &
wait
$PY - << 'EOF' >> /root/gpufree-data/hnu_v2/runs/night/v6g10p.log 2>&1
import json, os
R = "/root/gpufree-data/hnu_v2/runs"
for P, d in (("Z0 (no aug, 4000+2000)", "results_v6g10h_s1"), ("Z0A", "results_v6g10p_s1"), ("Z4A", "results_v6g10p_s1"), ("REF", "results_v6g10p_s1")):
    tag = "Z0" if P.startswith("Z0 (") else P
    for dr in ("0", "1"):
        f = f"{R}/{d}_dr{dr}/eval_student_{tag}.json"
        if not os.path.exists(f):
            continue
        s = json.load(open(f))["summary"]
        cells = " | ".join(f"{t[:6]} train {100*s[f'{t}/train']['success']:.0f} val {100*s[f'{t}/val']['success']:.0f} test {100*s[f'{t}/test']['success']:.0f}"
                           for t in ("CrouchedTraverse", "ThreeStepPoleRun", "FiveStepPoleRun"))
        print(f"{P:24s} dr{dr}: {cells}")
EOF
log "finished"
