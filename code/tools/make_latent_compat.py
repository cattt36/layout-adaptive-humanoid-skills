"""Expose joint (3-action) latents in the per-action layout expected by Results/8.1 scripts:
<out>/<Action>/<Z>/{train,val,test}_latents.npz (rows of that action only), best_motion_latent.pt, schema.json, ..."""
import glob, os, shutil, sys
import numpy as np

src, out = sys.argv[1], sys.argv[2]
ACTIONS = ("CrouchedTraverse", "ThreeStepPoleRun", "FiveStepPoleRun")
for zdir in sorted(glob.glob(os.path.join(src, "Z*"))):
    Z = os.path.basename(zdir)
    for a in ACTIONS:
        d = os.path.join(out, a, Z)
        os.makedirs(d, exist_ok=True)
        for f in os.listdir(zdir):
            p = os.path.join(zdir, f)
            if f.endswith("_latents.npz"):
                x = dict(np.load(p, allow_pickle=True))
                keep = np.array([str(t).startswith(a + "_") for t in x["trial_id"]])
                np.savez_compressed(os.path.join(d, f), **{k: v[keep] if len(v) == len(keep) else v for k, v in x.items()})
            elif f == "split_manifest.json":
                import json

                def filt(o):
                    if isinstance(o, dict):
                        return {k: filt(v) for k, v in o.items()
                                if not (k in ACTIONS and k != a) and not (isinstance(k, str) and ":" in k and k.split(":")[0] in ACTIONS and k.split(":")[0] != a)}
                    if isinstance(o, list):
                        return [filt(v) for v in o if not (isinstance(v, str) and any(v.startswith(x) for x in ACTIONS) and not v.startswith(a))]
                    return o

                json.dump(filt(json.load(open(p))), open(os.path.join(d, f), "w"), indent=1)
            elif os.path.isfile(p):
                dst = os.path.join(d, f)
                if not os.path.exists(dst):
                    os.symlink(p, dst)
print("compat layout written to", out)
