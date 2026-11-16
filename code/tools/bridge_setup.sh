#!/bin/bash
# Build the bridge motion dir (clips with a good lab->robot registration, fit error < 0.1 m) and evaluate the FLAT
# CrouchedTraverse expert under the bridge (zero-shot baseline).  usage: bridge_setup.sh <dir_name> [gpu]
NAME=${1:-ref_bridge_test}; G=${2:-0}
PY=/root/gpufree-data/venvs/mjlab/bin/python
H=/root/gpufree-data/hnu_v2
D=$H/data/$NAME/CrouchedTraverse
mkdir -p $D
for j in $H/data/bridge_v1/*_bridge.json; do
  c=$(basename $j _bridge.json)
  ok=$($PY -c "import json,sys; print(int(json.load(open('$j'))['fit_err_m'] < 0.1))")
  if [ "$ok" = 1 ] && [ -f $H/data/ref_v1/CrouchedTraverse/${c}_motion.npz ]; then
    ln -sf $H/data/ref_v1/CrouchedTraverse/${c}_motion.npz $D/
    ln -sf $j $D/
  fi
done
echo "clips: $(ls $D/*_motion.npz | wc -l)"
cd $H/code
export HNU_BRIDGE_ROOT=$H/data/$NAME
O=$H/runs/bridge_eval/flat_expert
mkdir -p $O
CUDA_VISIBLE_DEVICES=$G $PY eval_expert.py --task CrouchedTraverse --bridge --ckpt $(cat $H/runs/expert_eval/CrouchedTraverse/checkpoint.txt) \
  --per-clip 8 --out $O > $O/clean.log 2>&1
tail -n 40 $O/clean.log | grep -v Warn
