"""Are the train / val / test clip sets identical across levels (z_tables_v5/Z*/split.json)?"""
import glob
import json
import os

H = "/root/gpufree-data/hnu_v2"
sp = {os.path.basename(os.path.dirname(f)): json.load(open(f)) for f in sorted(glob.glob(f"{H}/data/z_tables_v5/Z*/split.json"))}
ref = sp["Z0"]
print("keys:", list(ref))
for z, s in sp.items():
    same = {k: set(s[k]) == set(ref[k]) for k in ("train", "val", "test")}
    other = {k: (s[k] if not isinstance(s[k], list) else len(s[k])) for k in s if k not in ("train", "val", "test")}
    print(z, same, other)
