"""Does a latent run use the trial-level split?  (split_manifest.json vs data/clip_split_trial/split.json)"""
import json
import sys

m = json.load(open(sys.argv[1]))
ref = json.load(open("/root/gpufree-data/hnu_v2/data/clip_split_trial/split.json"))
s = m["splits"]
print("structure:", {k: (list(v) if isinstance(v, dict) else type(v).__name__) for k, v in s.items()})
s = {k: (v.get("trial_ids") or v.get("trials") or next(x for x in v.values() if isinstance(x, list))) if isinstance(v, dict) else v for k, v in s.items()}
print("split_mode:", m["split_mode"], {k: len(v) for k, v in s.items()})
print("test == trial-split test:", set(s["test"]) == set(ref["test"]), "| val == trial-split val:", set(s["val"]) == set(ref["val"]),
      "| test trials in latent train:", len(set(s["train"]) & set(ref["test"])))
