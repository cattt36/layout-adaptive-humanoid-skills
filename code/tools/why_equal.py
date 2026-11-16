"""Why are L0-L4 students equal in closed loop?  Evidence:
 1  training clips of the students vs the latent subject split (are the 'test' clips seen in DAgger?)
 2  per-clip success: correlation between levels and with the teacher; share of student failures on clips where the
    teacher also fails
 3  student success / teacher success (retention) per split
  python tools/why_equal.py
"""
import glob
import json
import os

import numpy as np

H = "/root/gpufree-data/hnu_v2"
R = f"{H}/runs"
V = "v6g10s"
TASKS = ["CrouchedTraverse", "ThreeStepPoleRun", "FiveStepPoleRun"]
# 1 clips used for DAgger (unified geometry scene) vs latent split
dirs = {"ThreeStepPoleRun": f"{H}/data/ref_stakes_s93_h10/ThreeStepPoleRun", "FiveStepPoleRun": f"{H}/data/ref_stakes_s83_h10/FiveStepPoleRun",
        "CrouchedTraverse": f"{H}/data/ref_bridge_video_lift/CrouchedTraverse"}
train_clips = {os.path.basename(f).replace("_motion.npz", "") for d in dirs.values() for f in glob.glob(f"{d}/*_motion.npz")}
split = json.load(open(f"{H}/data/z_tables_v5/Z0/split.json"))
print("1) clips in the DAgger training scene:", len(train_clips))
for k in ("train", "val", "test"):
    s = set(split[k])
    print(f"   latent split '{k}': {len(s)} clips, of which used in student training: {len(s & train_clips)}")


def clips(f):
    return {r["clip"]: r for r in json.load(open(f))["clips"]} if os.path.exists(f) else None


for dr, name in (("0", "nominal"), ("1", "randomised")):
    print(f"\n2) per-clip success, {name} environment (clean)")
    teach = clips(f"{R}/results_{V}_bank_teacher_dr{dr}/eval_teacher_teacher.json")
    lv = {}
    for P in ["Z0", "Z1", "Z2", "Z3", "Z4"]:
        cs = [clips(f"{R}/results_{V}_s{s}_dr{dr}/eval_student_{P}.json") for s in (1, 2, 3)]
        cs = [c for c in cs if c]
        lv[P] = {c: np.mean([x[c]["success"] for x in cs]) for c in cs[0]}
    names = sorted(teach)
    M = np.array([[lv[P][c] for c in names] for P in lv])
    t = np.array([teach[c]["success"] for c in names])
    C = np.corrcoef(np.vstack([M, t]))
    print("   correlation of per-clip success (rows / cols: L0..L4, teacher):")
    for i, lab in enumerate(["L0", "L1", "L2", "L3", "L4", "T "]):
        print("    ", lab, " ".join(f"{x:5.2f}" for x in C[i]))
    hard = t < 0.999
    for P in lv:
        f = 1 - np.array([lv[P][c] for c in names])
        print(f"   L{P[1]}: failures on clips where the teacher also fails: {100 * f[hard].sum() / max(f.sum(), 1e-9):.0f}% "
              f"({hard.sum()} of {len(names)} clips)")
    print("   retention (student / teacher success), mean over levels and seeds, per split:")
    for sp in ("train", "val", "test"):
        ids = [c for c in names if teach[c]["split"] == sp]
        ts = np.mean([teach[c]["success"] for c in ids])
        ss = np.mean([np.mean([lv[P][c] for c in ids]) for P in lv])
        print(f"     {sp:5s}: teacher {100 * ts:.1f}  students {100 * ss:.1f}  retention {100 * ss / ts:.1f}%  ({len(ids)} clips)")
