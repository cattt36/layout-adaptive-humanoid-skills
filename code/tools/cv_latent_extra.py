"""Which trials does a fold's latent training set contain beyond the fold's student training trials?"""
import json

H = "/root/gpufree-data/hnu_v2"


def trials(m, k):
    for key in (f"{k}_trials", k):
        if key in m:
            return m[key]
    s = m.get("splits", {}).get(k, [])
    if isinstance(s, dict):
        s = s["trials"]
    if s and not isinstance(s[0], str):
        print("  (trial entries look like:", str(s[0])[:100], ")")
    return [x if isinstance(x, str) else x.get("trial", x.get("name", str(x))) for x in s] if isinstance(s, list) else list(s)


for name, man, fold in (("fold1 (latent_v5t)", "runs/latent_v5t/seed42/Z0/split_manifest.json", 1),
                        ("fold2", "runs/latent_cv_f2/seed42/Z0/split_manifest.json", 2)):
    m = json.load(open(f"{H}/{man}"))
    f = json.load(open(f"{H}/data/clip_split_cv/fold{fold}/split.json"))
    print(name, "manifest keys:", [k for k in m if not isinstance(m[k], (int, float))][:12])
    for k in ("train", "val", "test"):
        t = trials(m, k)
        print(f"  {k}: {len(t)}  extra vs fold: {sorted(set(t) - set(f[k]))}  missing: {sorted(set(f[k]) - set(t))}")
