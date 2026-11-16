#!/bin/bash
# Gate of the teacher fine-tunes (tools/tf_base.sh), thresholds fixed before the results:
#  CrouchedTraverse (A, B): with bases nominal >= 90 % and randomised >= 88 %, beam -5 cm >= 85 %, posts +-10 cm >= 88 %;
#     without bases (paper scene) nominal >= current teacher - 3 pts, body error <= 1.15 x current; retention vs the
#     original expert ok (tol 5 pts).  Pick: best min(nominal, randomised) with bases.
#  FiveStepPoleRun (C, D): success v2 nominal >= current - 2 pts, randomised >= current - 3 pts; floor touches per
#     rollout at most half of the current teacher (user rule: keep the current teacher only if the stake
#     performance clearly drops; partial scuff reduction is fine); retention ok (stakes_flat).  Pick: fewest touches.
# Teacher banks: W1 unchanged + candidate pairs (A, C) and (B, D) + current bank; evaluated on all clips (teachers track
# their own reference; criterion task, success_v2 for the stakes), 32 rollouts per clip (16 for the stress cases).
# usage: bash tools/tf_base_gate.sh        -> runs/tf_base_gate/, summary runs/tf_base_gate.md, log runs/night/tf_base.log
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2; D=$H/data; N=$H/runs/night; O=$H/runs/tf_base_gate
cd $H/code
export HF_HUB_OFFLINE=1 HNU_BRIDGE_ROOT=$D/ref_bridge_video_lift HNU_STAKE_H=0.10 HNU_STAKE_TOL=0.17 HNU_DR_LEVEL=1 \
  HNU_DR_LAG_SYNC=1 HNU_DR_LAG_SENSORS=1 PYTHONPATH=$H/code HNU_STAKE_SHAPE=mesh16 HNU_POST_SHAPE=mesh12 \
  HNU_GEOM_W1=$D/ref_stakes_s93_h10 HNU_GEOM_W2=$D/ref_stakes_s83_h10 HNU_SPLIT_JSON=$D/clip_split_trial/split_v2.json \
  HNU_EVAL_CRIT=task HNU_START_FRAC=0
log() { echo "[$(date '+%m-%d %H:%M:%S')] gate: $*" >> $N/tf_base.log; }
CUR=$(cat $N/v6g10n_teachers.txt)
get() { echo $CUR | sed 's/ /\n/g' | grep -A1 -- "--ckpt-$1" | tail -1; }
last() { ls $H/runs/logs/rsl_rl/hnu_track_$1/*_$2/model_*.pt 2>/dev/null | sort -V | tail -1; }
bank() {  # CT ckpt, 5PR ckpt
  echo "--ckpt-ThreeStepPoleRun $(get ThreeStepPoleRun) --ckpt-FiveStepPoleRun $2 --ckpt-CrouchedTraverse $1"
}
declare -A BANK
BANK[cur]="$CUR"
BANK[AC]="$(bank $(last CrouchedTraverse base_A) $(last FiveStepPoleRun scuff_C))"
BANK[BD]="$(bank $(last CrouchedTraverse base_B) $(last FiveStepPoleRun scuff_D))"
ev() {  # bank-tag case gpu
  local T=$1 C=$2 G=$3 PC=32 V="HNU_DR=0"
  case $C in
    base_nom) V="HNU_POST_BASE=1 HNU_DR=0 HNU_EVAL_FLOOR=1";;
    base_rand) V="HNU_POST_BASE=1 HNU_DR=1 HNU_EVAL_FLOOR=1";;
    base_push) V="HNU_POST_BASE=1 HNU_DR=0 PUSH"; PC=16;;
    base_dzm5) V="HNU_POST_BASE=1 HNU_DR=0 HNU_BRIDGE_DZ=-0.05"; PC=16;;
    base_pj10) V="HNU_POST_BASE=1 HNU_DR=0 HNU_BRIDGE_POST_JITTER=0.10"; PC=16;;
    nobase_nom) V="HNU_DR=0 HNU_EVAL_FLOOR=1";;
    nobase_rand) V="HNU_DR=1";;
  esac
  local X="" E=""
  for a in $V; do [ $a = PUSH ] && X="--push" || E="$E $a"; done
  [ -f $O/$T/eval_teacher_$C$([ -n "$X" ] && echo _push).json ] && return
  env $E CUDA_VISIBLE_DEVICES=$G $PY eval_student.py --teacher ${BANK[$T]} --scene geom --z-offsets 0,5,10,15,20,25,30,40,50 \
    --latent-dir $D/z_tables_v5t/Z4 --per-clip $PC $X --out $O/$T --tag _$C >> $O.log 2>&1
}
mkdir -p $O
log "start: AC = ${BANK[AC]} | BD = ${BANK[BD]}"
g=0
for T in cur AC BD; do
  ( for C in base_nom base_rand base_push base_dzm5 base_pj10 nobase_nom nobase_rand; do ev $T $C $g; done ) &
  g=$((g + 1))
done
# retention (original motion, nominal, no bases) vs the original experts
H4=/root/gpufree-data/hnu_v2_4090  # original experts: runs of the 4090 server (copied to L40, paths re-rooted)
OD=$(sed "s#/hnu_v2/#/hnu_v2_4090/#" $H4/runs/bridge_eval/bridge_video_lift_perc/checkpoint.txt)
OW2=$(sed "s#/hnu_v2/#/hnu_v2_4090/#" $H4/runs/expert_eval/FiveStepPoleRun/checkpoint.txt)
(
  for X in A B; do
    [ -s $O/ret_$X.json ] || CUDA_VISIBLE_DEVICES=3 $PY tools/retention_check.py --task CrouchedTraverse --mode bridge \
      --perc-orig 1 --perc-tuned 1 --orig $OD --tuned $(last CrouchedTraverse base_$X) 2>/dev/null | tail -1 > $O/ret_$X.json
  done
  for X in C D; do
    [ -s $O/ret_$X.json ] || CUDA_VISIBLE_DEVICES=4 $PY tools/retention_check.py --task FiveStepPoleRun --mode stakes_flat \
      --perc-tuned 1 --orig $OW2 --tuned $(last FiveStepPoleRun scuff_$X) 2>/dev/null | tail -1 > $O/ret_$X.json
  done
) &
wait
$PY - "$O" << 'EOF' > $H/runs/tf_base_gate.md
import json, os, sys
O = sys.argv[1]
def s(t, c, task, push=False):
    f = f"{O}/{t}/eval_teacher_{c}{'_push' if push else ''}.json"
    return json.load(open(f))["summary"][f"{task}/all"] if os.path.exists(f) else {}
def ret(x):
    try:
        return json.load(open(f"{O}/ret_{x}.json"))
    except Exception:
        return {}
pc = lambda d, k="success": 100 * d.get(k, float("nan"))  # noqa: E731
print("# Teacher fine-tune gate (post bases / floor scuff)\n")
print("## CrouchedTraverse (success %, teacher on its own reference, all 27 clips)\n")
print("| bank | bases nominal | bases randomised | bases push | bases beam -5 cm | bases posts +-10 cm | no bases nominal | no bases randomised | body err cm (no bases) | retention |")
print("|---|---|---|---|---|---|---|---|---|---|")
cur = {c: s("cur", c, "CrouchedTraverse") for c in ("nobase_nom",)}
res = {}
for t, x in (("cur", None), ("AC", "A"), ("BD", "B")):
    v = {c: pc(s(t, c, "CrouchedTraverse")) for c in ("base_nom", "base_rand", "base_dzm5", "base_pj10", "nobase_nom", "nobase_rand")}
    v["base_push"] = pc(s(t, "base_push", "CrouchedTraverse", True))
    err = 100 * s(t, "nobase_nom", "CrouchedTraverse").get("body_err", float("nan"))
    r = ret(x) if x else {}
    res[t] = (v, err, r)
    print(f"| {t} | {v['base_nom']:.1f} | {v['base_rand']:.1f} | {v['base_push']:.1f} | {v['base_dzm5']:.1f} | {v['base_pj10']:.1f} | "
          f"{v['nobase_nom']:.1f} | {v['nobase_rand']:.1f} | {err:.1f} | {r.get('original', '')}->{r.get('finetuned', '')} {r.get('retained', '')} |")
cv, cerr, _ = res["cur"]
best, bs = None, -1
for t, x in (("AC", "A"), ("BD", "B")):
    v, err, r = res[t]
    ok = (v["base_nom"] >= 90 and v["base_rand"] >= 88 and v["base_dzm5"] >= 85 and v["base_pj10"] >= 88
          and v["nobase_nom"] >= cv["nobase_nom"] - 3 and err <= 1.15 * cerr and bool(r.get("retained")))
    sc = min(v["base_nom"], v["base_rand"])
    print(f"\n{x}: {'PASS' if ok else 'FAIL'} (score {sc:.1f})")
    if ok and sc > bs:
        best, bs = x, sc
print(f"\nCT_CHOICE {best}\n")
print("## FiveStepPoleRun (success v2 %, floor touches per rollout between the stakes)\n")
print("| bank | nominal v2 | randomised v2 | touches nominal | touches randomised | touch s / rollout | no bases nominal v2 | retention |")
print("|---|---|---|---|---|---|---|---|")
res5 = {}
for t, x in (("cur", None), ("AC", "C"), ("BD", "D")):
    a, b, c = s(t, "base_nom", "FiveStepPoleRun"), s(t, "base_rand", "FiveStepPoleRun"), s(t, "nobase_nom", "FiveStepPoleRun")
    r = ret(x) if x else {}
    res5[t] = (pc(a, "success_v2"), pc(b, "success_v2"), a.get("floor_touch_events", float("nan")), r)
    print(f"| {t} | {pc(a, 'success_v2'):.1f} | {pc(b, 'success_v2'):.1f} | {a.get('floor_touch_events', float('nan')):.2f} | "
          f"{b.get('floor_touch_events', float('nan')):.2f} | {a.get('floor_touch_s', float('nan')):.3f} | {pc(c, 'success_v2'):.1f} | "
          f"{r.get('original', '')}->{r.get('finetuned', '')} {r.get('retained', '')} |")
cn, cr, _, _ = res5["cur"]
best5, bt = None, 1e9
for t, x in (("AC", "C"), ("BD", "D")):
    n_, r_, touch, r = res5[t]
    ok = n_ >= cn - 2 and r_ >= cr - 3 and touch <= 0.5 * res5["cur"][2] and bool(r.get("retained"))
    print(f"\n{x}: {'PASS' if ok else 'FAIL'} (touches {touch:.2f})")
    if ok and touch < bt:
        best5, bt = x, touch
print(f"\nW2_CHOICE {best5}")
EOF
log "done: CT $(grep CT_CHOICE $H/runs/tf_base_gate.md) / 5PR $(grep W2_CHOICE $H/runs/tf_base_gate.md)"
