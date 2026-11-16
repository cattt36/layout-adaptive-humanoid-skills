"""Subjects of the TRAIN clips (latent split manifest) and a subject-level low-data subset.
  python tools/subjects.py <split_manifest.json> <clips_train.txt> [<out_lowdata.txt> <fraction>]
"""
import collections
import json
import random
import sys

m = json.load(open(sys.argv[1]))
print("subject_split_policy:", json.dumps(m.get("subject_split_policy"), ensure_ascii=False)[:300])
sa = m["subject_assignment"]
print("subject_assignment sample:", json.dumps(dict(list(sa.items())[:3]) if isinstance(sa, dict) else sa[:3], ensure_ascii=False)[:400])
stg = m["subject_trial_groups"]
print("subject_trial_groups sample:", json.dumps(dict(list(stg.items())[:2]) if isinstance(stg, dict) else stg[:2], ensure_ascii=False)[:400])
train = [ln.strip() for ln in open(sys.argv[2]) if ln.strip()]
# clip -> subject from subject_trial_groups {subject: [trial ids]}
c2s = {}  # stg = {action: {subject: [trial numbers]}}; clip name = <action>_<NNN>
for act, groups in stg.items():
    for subj, trials in groups.items():
        for t in trials:
            c2s[f"{act}_{int(t):03d}"] = subj
per = collections.defaultdict(list)
for c in train:
    per[(c.rsplit("_", 1)[0], c2s.get(c, "?"))].append(c)
print("train clips per (action, subject):")
for k in sorted(per):
    print("  ", k, len(per[k]))
if len(sys.argv) > 4 and sys.argv[4].startswith("subject_"):  # one named training subject per action
    keep = [c for c in train if c2s.get(c) == sys.argv[4]]
    open(sys.argv[3], "w").write("\n".join(sorted(keep)) + "\n")
    print(f"single-subject subset ({sys.argv[4]}): {len(keep)} of {len(train)} train clips ->", sys.argv[3],
          dict(collections.Counter(c.rsplit("_", 1)[0] for c in keep)))
elif len(sys.argv) > 4:
    frac = float(sys.argv[4])
    rnd = random.Random(2026)  # fixed before any result
    keep = []
    for act in sorted({a for a, _ in per}):
        subs = sorted(s for a, s in per if a == act)
        rnd.shuffle(subs)
        n_act = sum(len(per[(act, s)]) for s in subs)
        got = 0
        for s in subs:  # whole subjects until the fraction of this action's clips is reached
            if got >= frac * n_act:
                break
            keep += per[(act, s)]
            got += len(per[(act, s)])
    open(sys.argv[3], "w").write("\n".join(sorted(keep)) + "\n")
    print(f"low-data subset: {len(keep)} of {len(train)} train clips ->", sys.argv[3],
          dict(collections.Counter(c.rsplit('_', 1)[0] for c in keep)))
