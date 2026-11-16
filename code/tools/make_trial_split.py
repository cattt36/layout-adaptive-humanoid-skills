"""Trial-level held-out split (decided before any result of this protocol): for every action and every
subject, the trials are shuffled (seed 2026) and split ~63 / 12 / 25 % into train / val / test (at least one test
trial per subject when the subject has >= 3 trials).  Students see every subject but never a test trial.
Writes data/clip_split_trial/{clips_train,clips_val,clips_test}.txt and split.json (eval_student split override)."""
import collections
import glob
import json
import os
import random

H = "/root/gpufree-data/hnu_v2"
DIRS = {"ThreeStepPoleRun": f"{H}/data/ref_stakes_s93_h10/ThreeStepPoleRun", "FiveStepPoleRun": f"{H}/data/ref_stakes_s83_h10/FiveStepPoleRun",
        "CrouchedTraverse": f"{H}/data/ref_bridge_video_lift/CrouchedTraverse"}
man = json.load(open(f"{H}/runs/latent_v5/seed42/Z0/split_manifest.json"))
c2s = {f"{a}_{int(t):03d}": s for a, g in man["subject_trial_groups"].items() for s, ts in g.items() for t in ts}
rnd = random.Random(2026)
out = {"train": [], "val": [], "test": []}
for act, d in DIRS.items():
    clips = sorted(os.path.basename(f).replace("_motion.npz", "") for f in glob.glob(f"{d}/*_motion.npz"))
    by = collections.defaultdict(list)
    for c in clips:
        by[c2s.get(c, "unknown")].append(c)
    for subj in sorted(by):
        cs = sorted(by[subj])
        rnd.shuffle(cs)
        n = len(cs)
        n_te = max(1, round(0.25 * n)) if n >= 3 else 0
        n_va = max(1, round(0.12 * n)) if n >= 5 else 0
        out["test"] += cs[:n_te]
        out["val"] += cs[n_te:n_te + n_va]
        out["train"] += cs[n_te + n_va:]
        print(f"{act:12s} {subj:10s} trials {n:2d} -> train {n - n_te - n_va:2d} val {n_va} test {n_te}")
os.makedirs(f"{H}/data/clip_split_trial", exist_ok=True)
for k, v in out.items():
    open(f"{H}/data/clip_split_trial/clips_{k}.txt", "w").write("\n".join(sorted(v)) + "\n")
json.dump({k: sorted(v) for k, v in out.items()} | {"protocol": "trial-level held-out, seed 2026"},
          open(f"{H}/data/clip_split_trial/split.json", "w"), indent=1)
print({k: len(v) for k, v in out.items()})
