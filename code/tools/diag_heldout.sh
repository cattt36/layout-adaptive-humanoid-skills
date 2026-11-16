#!/bin/bash
# Why do held-out students fail on test-subject clips?  Export v6g10h L0 seed 1, run sim2sim (nominal, all clips),
# then summarise success / failure cause / failure time per split.
H=/root/gpufree-data/hnu_v2
cd $H/code
export HF_HUB_OFFLINE=1 HNU_BRIDGE_ROOT=$H/data/ref_bridge_video_lift HNU_STAKE_H=0.10 HNU_GEOM_W1=$H/data/ref_stakes_s93_h10 \
  HNU_GEOM_W2=$H/data/ref_stakes_s83_h10 HNU_STAKE_TOL=0.17 PYTHONPATH=$H/code
E=$H/deploy_exports/diag_v6g10h_s1_Z0
mkdir -p $E
G=0; [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1 | tr -d ' ')" -gt 9000 ] && G=1
[ -f $E/policy.onnx ] || CUDA_VISIBLE_DEVICES=$G /root/gpufree-data/venvs/mjlab/bin/python deploy/export_student.py \
  --student $H/runs/students_v6g10h_d2_s1/Z0/student_2000.pt --latent-dir $H/data/z_tables_v5/Z0 --out $E > $E/export.log 2>&1
/root/gpufree-data/venvs/deploy/bin/python deploy/sim2sim.py $E --trials 2 --workers 40 --out $E/nominal.json > /dev/null 2>&1
/root/gpufree-data/venvs/deploy/bin/python - << 'EOF'
import json, collections
H = "/root/gpufree-data/hnu_v2"
sp = {}
for k in ("train", "val", "test"):
    for c in open(f"{H}/data/clip_split_v5/clips_{k}.txt").read().split():
        sp[c] = k
rows = json.load(open(f"{H}/deploy_exports/diag_v6g10h_s1_Z0/nominal.json"))["rows"]
by = collections.defaultdict(list)
for r in rows:
    by[(r["task"], sp.get(r["clip"], "?"))].append(r)
for (t, s), rs in sorted(by.items()):
    causes = collections.Counter(r["cause"] for r in rs if r["cause"])
    frac = [r["steps"] / r["T"] for r in rs if r["cause"]]
    print(f"{t:12s} {s:5s} n={len(rs):3d} success {100 * sum(r['success'] for r in rs) / len(rs):5.1f}%  "
          f"causes {dict(causes)}  failure at {100 * sum(frac) / max(len(frac), 1):.0f}% of the clip (mean)  "
          f"body err {100 * sum(r['body_err'] for r in rs) / len(rs):.1f} cm")
EOF
