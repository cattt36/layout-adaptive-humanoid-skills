"""Subject split of the latent z tables: identical across levels? clips per split and action; write clip lists."""
import collections
import glob
import hashlib
import json
import os
import sys

H = "/root/gpufree-data/hnu_v2"
fs = sorted(glob.glob(f"{H}/data/z_tables_v5/Z*/split.json"))
print({os.path.basename(os.path.dirname(f)): hashlib.md5(open(f, "rb").read()).hexdigest()[:8] for f in fs})
s = json.load(open(fs[0]))
for k, v in s.items():
    if isinstance(v, list):
        print(k, len(v), dict(collections.Counter(c.rsplit("_", 1)[0] for c in v)))
if len(sys.argv) > 1:  # write <out_dir>/clips_<split>.txt
    os.makedirs(sys.argv[1], exist_ok=True)
    for k in ("train", "val", "test"):
        open(os.path.join(sys.argv[1], f"clips_{k}.txt"), "w").write("\n".join(sorted(s[k])) + "\n")
    print("written", sys.argv[1])
