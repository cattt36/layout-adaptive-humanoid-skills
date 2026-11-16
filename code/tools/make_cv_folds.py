"""4-fold trial-level cross-validation (pre-registered , runs/preregistration_cv.md):
fold 1 = split_v2 (test 16 / val 9 / train 51); folds 2-4: the remaining valid trials, grouped by action x subject,
shuffled (seed 3030) and dealt round-robin into the fold-2/3/4 test sets; per fold, val = ~12 % of the non-test
trials of every action x subject group with >= 5 trials (seed 3030 + fold), train = the rest.
Writes data/clip_split_cv/fold<k>/{split.json, clips_train.txt, clips_val.txt, clips_test.txt} and folds.json."""
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
v2 = json.load(open(f"{H}/data/clip_split_trial/split_v2.json"))
EXC = set(v2["excluded"])
valid = sorted(os.path.basename(f).replace("_motion.npz", "") for d in DIRS.values() for f in glob.glob(f"{d}/*_motion.npz"))
valid = [c for c in valid if c not in EXC]
group = lambda c: (c.split("_")[0], c2s.get(c, "unknown"))  # noqa: E731
folds = {1: dict(train=sorted(v2["train"]), val=sorted(v2["val"]), test=sorted(v2["test"]))}
rest = [c for c in valid if c not in set(v2["test"])]
by = collections.defaultdict(list)
for c in rest:
    by[group(c)].append(c)
rnd = random.Random(3030)
tests = {2: [], 3: [], 4: []}
k = 0
for g in sorted(by):
    cs = sorted(by[g])
    rnd.shuffle(cs)
    for c in cs:  # round-robin across groups keeps the folds balanced
        tests[2 + k % 3].append(c)
        k += 1
for f in (2, 3, 4):
    te = set(tests[f])
    r = random.Random(3030 + f)
    gb = collections.defaultdict(list)
    for c in valid:
        if c not in te:
            gb[group(c)].append(c)
    val = []
    for g in sorted(gb):
        cs = sorted(gb[g])
        r.shuffle(cs)
        n = len(cs)
        val += cs[:max(1, round(0.12 * n))] if n >= 5 else []
    folds[f] = dict(test=sorted(te), val=sorted(val), train=sorted(c for c in valid if c not in te and c not in set(val)))
# checks: every valid trial in exactly one test set; within a fold train / val / test disjoint and complete
all_test = [c for f in folds for c in folds[f]["test"]]
assert sorted(all_test) == sorted(valid), "every valid trial must be tested exactly once"
for f, s in folds.items():
    assert not (set(s["train"]) & set(s["test"])) and not (set(s["val"]) & set(s["test"])) and not (set(s["train"]) & set(s["val"]))
    if f > 1:
        assert sorted(s["train"] + s["val"] + s["test"]) == sorted(valid)
out = f"{H}/data/clip_split_cv"
for f, s in folds.items():
    d = f"{out}/fold{f}"
    os.makedirs(d, exist_ok=True)
    for kk in ("train", "val", "test"):
        open(f"{d}/clips_{kk}.txt", "w").write("\n".join(s[kk]) + "\n")
    json.dump(dict(s, excluded=sorted(EXC), protocol=f"4-fold trial-level CV, fold {f} (seed 3030; fold 1 = split_v2)"),
              open(f"{d}/split.json", "w"), indent=1)
    per = collections.Counter(c.split("_")[0] for c in s["test"])
    print(f"fold {f}: train {len(s['train'])} val {len(s['val'])} test {len(s['test'])}  test per action {dict(per)}")
json.dump({f"fold{f}": s["test"] for f, s in folds.items()}, open(f"{out}/folds.json", "w"), indent=1)
print("valid trials", len(valid), "| tested once each:", len(all_test) == len(set(all_test)) == len(valid))
