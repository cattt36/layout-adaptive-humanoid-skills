"""Fairness audit of the latent encoders: training arguments of every encoder (fold 1 = latent_v5t, folds 2-4 =
latent_cv_f<k>) compared key by key.  Expected differences only: profile (= supervision level), seed, output dir, split
file.  Also: split per fold identical across levels / seeds, val trials = student val trials."""
import glob
import json
import os
import sys

import torch

H = "/root/gpufree-data/hnu_v2"
roots = {"f1": f"{H}/runs/latent_v5t"} | {f"f{k}": f"{H}/runs/latent_cv_f{k}" for k in (2, 3, 4)}
EXPECTED = {"profile", "ablation_profile", "seed", "out", "out_dir", "output_dir", "save_dir", "run_dir", "split_json", "split_file", "latent_root"}
ref, rows, splits = None, [], {}
for fold, r in roots.items():
    for ck in sorted(glob.glob(f"{r}/seed*/Z*/best_motion_latent.pt")):
        d = torch.load(ck, map_location="cpu", weights_only=False)
        a = d.get("args") or d.get("config") or {}
        a = vars(a) if not isinstance(a, dict) else a
        name = ck.replace(H + "/runs/", "").replace("/best_motion_latent.pt", "")
        if ref is None:
            ref = (name, a)
        diff = sorted(k for k in set(a) | set(ref[1]) if a.get(k) != ref[1].get(k) and k not in EXPECTED)
        rows.append((name, len(a), diff, {k: a.get(k) for k in diff}))
        m = json.load(open(os.path.dirname(ck) + "/split_manifest.json"))["splits"]
        splits.setdefault(fold, set()).add(json.dumps({k: sorted(v["trials"]) for k, v in m.items()}, sort_keys=True))
print(f"reference: {ref[0]} ({len(ref[1])} args)")
bad = 0
for name, n, diff, vals in rows:
    if diff:
        bad += 1
        print(f"  {name}: UNEXPECTED differences {vals}")
print(f"encoders checked: {len(rows)}, with unexpected argument differences: {bad}")
for fold, s in splits.items():
    sp = json.loads(next(iter(s)))
    fs = json.load(open(f"{H}/data/clip_split_cv/fold{fold[1]}/split.json"))
    print(f"{fold}: distinct encoder splits across levels/seeds = {len(s)};  val == student val: {sorted(sp['val']) == sorted(fs['val'])};"
          f"  test == student test (+exclusions): {set(fs['test']) <= set(sp['test'])};  test in encoder train: {len(set(fs['test']) & set(sp['train']))}")
sys.exit(0)
