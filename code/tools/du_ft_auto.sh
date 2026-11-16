#!/bin/bash
# Bridge teacher fine-tune for LOWER beams (user: fine-tune the bridge teacher if it is sensitive to beam
# height / post spacing, launch the distillation once it meets the standard).  Runs unattended on the server.
#  sensitivity (corrected geometry): beam -5 cm 79 %, -8 cm 44 %; higher beams and post spacing (+-10 cm) insensitive.
#  round 1: two variants in parallel from the tf2_full bridge teacher, 4000 its:
#     A beam height -6..+6 cm   B -8..+6 cm   (both: posts +-10 cm along the beam, shift +-5 cm, prism posts,
#     default pushes, graded contact penalty, slip penalty, 25 % frame-0 resets)
#  gate (decided before training): beam -5 cm >= 90 %, -3 cm >= 92 %, +8 cm >= 92 %, posts +-10 cm >= 92 %,
#     nominal >= 92 %, randomised >= 90 %, retention ok (bridge, vs the original expert, tol 5 pts),
#     nominal body error <= 1.1x the current bridge teacher (tf2_full)
#  round 2 (only if neither passes): the better variant +3000 its, gate again; still failing -> stop, no launch.
#  pass -> runs/night/v6g10n_teachers.txt (W1 rs10c, W2 tf2_full, DuMu chosen) and launch run_v6g10n.sh + monitor +
#     watchdog.
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
N=$H/runs/night
L=$N/du_ft_auto.log
cd $H/code
log() { echo "[$(date '+%m-%d %H:%M')] $*" >> $L; }
export HF_HUB_OFFLINE=1 HNU_BRIDGE_ROOT=$H/data/ref_bridge_video_lift HNU_POST_SHAPE=mesh12 HNU_STAKE_SHAPE=mesh16
BASE=$(cat $N/v6g10n_teachers_proposed.txt)
get() { echo $1 | sed 's/ /\n/g' | grep -A1 -- "--ckpt-$2" | tail -1; }
SRC_DU=$(get "$BASE" CrouchedTraverse)
last() { ls $H/runs/logs/rsl_rl/hnu_track_CrouchedTraverse/*_$1/model_*.pt 2>/dev/null | sort -V | tail -1; }
train() {  # gpu src run iters zlo
  local G=$1 SRC=$2 RUN=$3 IT=$4 ZLO=$5
  [ -n "$(last $RUN)" ] && return
  ( cd $H/runs && env HNU_DR=1 HNU_DR_LEVEL=2 HNU_PERCEPTION=1 HNU_BRIDGE_W=-5.0 HNU_BRIDGE_FORCE=1 HNU_START_FRAC=0.25 \
      HNU_PUSH_SCALE=1.0 HNU_SLIP_W=-0.3 HNU_BRIDGE_JITTER_XY=0.05 HNU_BRIDGE_JITTER_ZLO=$ZLO HNU_BRIDGE_JITTER_ZHI=0.06 \
      HNU_BRIDGE_POST_JITTER=0.10 CUDA_VISIBLE_DEVICES=$G $PY $H/code/train_hnu.py HNU-Track-CrouchedTraverse-Bridge \
      --env.scene.num-envs 4096 --agent.experiment-name hnu_track_CrouchedTraverse --agent.run-name $RUN --agent.max-iterations $IT \
      --agent.resume True --agent.load-run $(basename $(dirname $SRC)) --agent.load-checkpoint $(basename $SRC) \
      > $H/runs/logs/CrouchedTraverse_$RUN.log 2>&1 < /dev/null )
}
EV="HNU_STAKE_H=0.10 HNU_GEOM_W1=$H/data/ref_stakes_s93_h10 HNU_GEOM_W2=$H/data/ref_stakes_s83_h10 HNU_STAKE_TOL=0.17 \
HNU_DR_LEVEL=1 HNU_DR_LAG_SYNC=1 HNU_DR_LAG_SENSORS=1 PYTHONPATH=$H/code HNU_SPLIT_JSON=$H/data/clip_split_trial/split_v2.json \
HNU_EVAL_CRIT=task HNU_START_FRAC=0"
gate() {  # tag ckpt gpu -> prints PASS/FAIL line, writes $H/runs/du_gate_<tag>/
  local T=$1 CK=$2 G=$3 O=$H/runs/du_gate_$1
  local BANK=${BASE/$SRC_DU/$CK}
  mkdir -p $O
  for C in "nom HNU_DR=0" "rand HNU_DR=1" "push HNU_DR=0 PUSH" "dzm5 HNU_DR=0 HNU_BRIDGE_DZ=-0.05" "dzm3 HNU_DR=0 HNU_BRIDGE_DZ=-0.03" \
           "dzm8 HNU_DR=0 HNU_BRIDGE_DZ=-0.08" "dzp8 HNU_DR=0 HNU_BRIDGE_DZ=0.08" "spm10 HNU_DR=0 HNU_BRIDGE_POST_SPREAD=-0.10" \
           "spp10 HNU_DR=0 HNU_BRIDGE_POST_SPREAD=0.10" "pj10 HNU_DR=0 HNU_BRIDGE_POST_JITTER=0.10"; do
    set -- $C; local NAME=$1; shift
    local X=""; local V=""
    for a in "$@"; do [ $a = PUSH ] && X="--push" || V="$V $a"; done
    local PC=16; [ $NAME = nom ] || [ $NAME = rand ] && PC=32
    env $EV $V CUDA_VISIBLE_DEVICES=$G $PY eval_student.py --teacher $BANK --scene geom --z-offsets 0,5,10,15,20,25,30,40,50 \
      --latent-dir $H/data/z_tables_v5t/Z4 --per-clip $PC $X --out $O --tag _$NAME > /dev/null 2>&1
  done
  local R=$(CUDA_VISIBLE_DEVICES=$G $PY tools/retention_check.py --task CrouchedTraverse --mode bridge --perc-orig 1 --perc-tuned 1 \
    --orig $(cat $H/runs/bridge_eval/bridge_video_lift_perc/checkpoint.txt) --tuned $CK 2>/dev/null | tail -1)
  echo "$R" > $O/retention.json
  $PY - "$O" "$T" << 'EOF'
import json, sys
O, T = sys.argv[1], sys.argv[2]
s = lambda n: json.load(open(f"{O}/eval_teacher_{n}.json"))["summary"]["CrouchedTraverse/all"]  # noqa: E731
v = {n: 100 * s(n)["success"] for n in ("nom", "rand", "push", "dzm5", "dzm3", "dzm8", "dzp8", "spm10", "spp10", "pj10")}
err = 100 * s("nom")["body_err"]
ref_err = 100 * json.load(open("/root/gpufree-data/hnu_v2/runs/accept_prism/dr0/eval_teacher_new.json"))["summary"]["CrouchedTraverse/all"]["body_err"]
ret = json.load(open(f"{O}/retention.json"))
ok = (v["dzm5"] >= 90 and v["dzm3"] >= 92 and v["dzp8"] >= 92 and min(v["spm10"], v["spp10"], v["pj10"]) >= 92
      and v["nom"] >= 92 and v["rand"] >= 90 and ret.get("retained") and err <= 1.1 * ref_err)
score = min(v["dzm5"], v["nom"])
print(("PASS" if ok else "FAIL") + f" {T} score {score:.1f} | " + " ".join(f"{k} {x:.0f}" for k, x in v.items())
      + f" | body err {err:.1f} cm (limit {1.1 * ref_err:.1f}) | retention {ret.get('original')}->{ret.get('finetuned')} {ret.get('retained')}")
EOF
}
launch() {  # chosen DuMu ckpt
  local CK=$1
  echo "${BASE/$SRC_DU/$CK}" > $N/v6g10n_teachers.txt
  log "teacher bank written: $(cat $N/v6g10n_teachers.txt)"
  setsid nohup bash run_v6g10n.sh > $N/run_v6g10n.out 2>&1 < /dev/null &
  sleep 5
  bash tools/restart_monitor.sh v6g10n >> $L 2>&1
  setsid nohup bash tools/watchdog.sh v6g10n > /dev/null 2>&1 < /dev/null &
  log "v6g10n launched (pipeline, TensorBoard monitor, watchdog)"
}
log "round 1: variants A (beam -6..+6 cm) and B (-8..+6 cm), 4000 its each"
train 0 $SRC_DU du3_A 4000 -0.06 &
train 1 $SRC_DU du3_B 4000 -0.08 &
wait
A=$(last du3_A); B=$(last du3_B)
log "round 1 trained: A $A  B $B"
gate A $A 0 > $H/runs/du_gate_A.txt 2>&1 &
gate B $B 1 > $H/runs/du_gate_B.txt 2>&1 &
wait
log "gate A: $(tail -1 $H/runs/du_gate_A.txt)"
log "gate B: $(tail -1 $H/runs/du_gate_B.txt)"
pick() {  # prints the ckpt of the best PASS among the given tags (score), or nothing
  local best="" bs=-1
  for t in "$@"; do
    local line=$(tail -1 $H/runs/du_gate_$t.txt)
    local sc=$(echo "$line" | awk '{print $4}')
    if [[ $line == PASS* ]] && $PY -c "import sys; sys.exit(0 if $sc > $bs else 1)"; then best=$t; bs=$sc; fi
  done
  echo $best
}
W=$(pick A B)
if [ -n "$W" ]; then
  CK=$(last du3_$W); log "round 1: variant $W passes -> $CK"; launch $CK; exit 0
fi
# round 2: continue the better variant (higher score)
SA=$(tail -1 $H/runs/du_gate_A.txt | awk '{print $4}'); SB=$(tail -1 $H/runs/du_gate_B.txt | awk '{print $4}')
if $PY -c "import sys; sys.exit(0 if $SA >= $SB else 1)"; then V=A; ZLO=-0.06; else V=B; ZLO=-0.08; fi
log "round 1: no variant passes; round 2 continues variant $V for 3000 its"
train 0 $(last du3_$V) du3_${V}2 3000 $ZLO
gate ${V}2 $(last du3_${V}2) 0 > $H/runs/du_gate_${V}2.txt 2>&1
log "gate ${V}2: $(tail -1 $H/runs/du_gate_${V}2.txt)"
if [[ $(tail -1 $H/runs/du_gate_${V}2.txt) == PASS* ]]; then
  CK=$(last du3_${V}2); log "round 2 passes -> $CK"; launch $CK; exit 0
fi
log "STOP: no bridge teacher meets the standard after 2 rounds; distillation NOT launched (waiting for the user)"
