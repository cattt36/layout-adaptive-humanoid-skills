"""Per-trial criterion-v2 success of the current teacher bank vs the pilot on ThreeStepPoleRun (nominal), with the
stake gaps of each trial -> are the misses spread out or concentrated in a few (odd) trials?"""
import json
import sys

import numpy as np

H = "/root/gpufree-data/hnu_v2"
D = f"{H}/data/ref_stakes_s93_h10/ThreeStepPoleRun"
TAG = sys.argv[1] if len(sys.argv) > 1 else "pilot"
X = sys.argv[2] if len(sys.argv) > 2 else ""  # "" nominal, "_push"
DR = sys.argv[3] if len(sys.argv) > 3 else "0"
cur = {c["clip"]: c for c in json.load(open(f"{H}/runs/results_v6g10mv2_bank_teacher_dr{DR}/eval_teacher_teacher{X}.json"))["clips"]}
pil = {c["clip"]: c for c in json.load(open(f"{H}/runs/tf2_eval_{TAG}/dr{DR}/eval_teacher_teacher{X}.json"))["clips"]}
print(f"dr{DR}{X}   trial  split  gaps_m       | current: old v2 stepped full | {TAG}: v2 stepped full")
for k in sorted(c for c in cur if c.startswith("ThreeStepPoleRun")):
    j = json.load(open(f"{D}/{k}_stakes.json"))
    xy = np.array([[s["x"], s["y"]] for s in j["stakes"]])
    gaps = " ".join(f"{g:.2f}" for g in np.linalg.norm(np.diff(xy, axis=0), axis=1))
    a, b = cur[k], pil.get(k, {})
    print(f"{k:16s} {a['split']:5s}  {gaps:12s} | {100 * a['success']:4.0f} {100 * a['success_v2']:4.0f} "
          f"{100 * a['stakes_stepped_frac']:4.0f} {100 * a['stakes_all_full']:4.0f} | "
          f"{100 * b.get('success_v2', np.nan):4.0f} {100 * b.get('stakes_stepped_frac', np.nan):4.0f} {100 * b.get('stakes_all_full', np.nan):4.0f}")
