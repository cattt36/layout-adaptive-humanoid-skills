"""Per fold: test trials by action x subject (subjects from the latent split manifest)."""
import collections
import json

H = "/root/gpufree-data/hnu_v2"
man = json.load(open(f"{H}/runs/latent_v5/seed42/Z0/split_manifest.json"))
c2s = {f"{a}_{int(t):03d}": s for a, g in man["subject_trial_groups"].items() for s, ts in g.items() for t in ts}
for f in (1, 2, 3, 4):
    sp = json.load(open(f"{H}/data/clip_split_cv/fold{f}/split.json"))
    for k in ("test", "train"):
        cnt = collections.Counter((c.split("_")[0], c2s.get(c, "?")) for c in sp[k])
        print(f"fold {f} {k:5s} n={len(sp[k]):2d}", "  ".join(f"{a[:4]}/{s}:{n}" for (a, s), n in sorted(cnt.items())))
