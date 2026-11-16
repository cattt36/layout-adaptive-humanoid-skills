#!/bin/bash
# Failure analysis of the trial-level students (v6g10t, seed 1, L0 and L4) on TRAIN and VAL trials only (test trials
# are not looked at, so that any method change stays independent of the test set).  sim2sim, nominal, task criterion.
H=/root/gpufree-data/hnu_v2
cd $H/code
export HF_HUB_OFFLINE=1 HNU_BRIDGE_ROOT=$H/data/ref_bridge_video_lift HNU_STAKE_H=0.10 HNU_GEOM_W1=$H/data/ref_stakes_s93_h10 \
  HNU_GEOM_W2=$H/data/ref_stakes_s83_h10 HNU_STAKE_TOL=0.17 PYTHONPATH=$H/code
for P in Z0 Z4; do
  E=$H/deploy_exports/diag_t_$P; mkdir -p $E
  [ -f $E/policy.onnx ] || CUDA_VISIBLE_DEVICES=0 /root/gpufree-data/venvs/mjlab/bin/python deploy/export_student.py \
    --student $H/runs/students_v6g10t_d2_s1/$P/student_2000.pt --latent-dir $H/data/z_tables_v5t/$P --out $E > $E/export.log 2>&1
  /root/gpufree-data/venvs/deploy/bin/python deploy/sim2sim.py $E --trials 4 --workers 40 --crit task --out $E/task.json > /dev/null 2>&1
done
/root/gpufree-data/venvs/deploy/bin/python - << 'EOF'
import json, collections
import numpy as np
H = "/root/gpufree-data/hnu_v2"
sp = json.load(open(f"{H}/data/clip_split_trial/split.json"))
where = {c: k for k in ("train", "val", "test") for c in sp[k]}
for P in ("Z0", "Z4"):
    rows = [r for r in json.load(open(f"{H}/deploy_exports/diag_t_{P}/task.json"))["rows"] if where.get(r["clip"]) in ("train", "val")]
    print(f"\n######## L{P[1]} (train + val trials only, {len(rows)} episodes)")
    for t in ("ThreeStepPoleRun", "FiveStepPoleRun", "CrouchedTraverse"):
        rs = [r for r in rows if r["task"] == t]
        print(f"== {t}: success {100 * np.mean([r['success'] for r in rs]):.0f}%  causes {dict(collections.Counter(r['cause'] for r in rs if r['cause']))}")
        if t.startswith(("ThreeStepPoleRun", "FiveStepPoleRun")):
            n = max(len(r["stakes"]) for r in rs)
            for i in range(n):
                st = [r["stakes"][i] for r in rs if i < len(r["stakes"]) and r["stakes"][i]["reached"]]
                if st:
                    print(f"   stake {i + 1}: reached {len(st)}/{len(rs)}  on-top {100 * np.mean([s['on'] >= 0.5 for s in st]):.0f}%  "
                          f"dist {100 * np.mean([s['dist'] for s in st]):.1f} cm  along {100 * np.mean([s['along'] for s in st]):+.1f} cm "
                          f"(|.| {100 * np.mean([abs(s['along']) for s in st]):.1f})  lateral {100 * np.mean([s['lat'] for s in st]):+.1f} cm "
                          f"(|.| {100 * np.mean([abs(s['lat']) for s in st]):.1f})")
            fails = [r for r in rs if not r["success"]]
            if fails:
                last = [sum(s["reached"] for s in r["stakes"]) for r in fails]
                print(f"   failures: stakes reached before the failure {dict(collections.Counter(last))}, at {np.mean([r['steps'] / r['T'] for r in fails]) * 100:.0f}% of the clip")
        else:
            fails = [r for r in rs if not r["success"]]
            cont = collections.Counter()
            for r in rs:
                for k, v in r["bridge_contacts"].items():
                    cont[k] += v > 0
            print(f"   episodes touching: {dict(cont)}")
            for r in fails:
                print(f"   fail {r['clip']} t{r['trial']}: {r['cause']} at {100 * r['steps'] / r['T']:.0f}% of the clip, along-beam {r['beam_rel']:+.2f} m, "
                      f"pelvis {r['pelvis_h']:.2f} m (ref {r['ref_pelvis_h']:.2f}), contacts {r['bridge_contacts']}")
EOF
