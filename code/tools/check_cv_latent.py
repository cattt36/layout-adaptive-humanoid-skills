"""Leak check of a CV-fold latent run: its split_manifest.json vs data/clip_split_cv/fold<k>/split.json.
usage: check_cv_latent.py <split_manifest.json> <fold split.json>   -> prints OK / LEAK"""
import json
import sys

m = json.load(open(sys.argv[1]))
ref = json.load(open(sys.argv[2]))
s = m["splits"]
s = {k: (v.get("trial_ids") or v.get("trials") or next(x for x in v.values() if isinstance(x, list))) if isinstance(v, dict) else v
     for k, v in s.items()}
leak = set(s["train"]) & set(ref["test"])
ok = m["split_mode"] == "file" and not leak and set(s["test"]) == set(ref["test"])
print(("OK" if ok else "LEAK") + f" split_mode {m['split_mode']} sizes {({k: len(v) for k, v in s.items()})} "
      f"test==fold test {set(s['test']) == set(ref['test'])} test trials in latent train {len(leak)}")
