"""setup_layout.py <dir> [--prefix-copy <code copy>]: recreates the directory layout of the training server inside <dir>
after scripts/download_assets.sh <dir>.

Every model file of docs/MANIFEST_models.json is linked (relative symlink, nothing is copied) from its place in the
release (models/...) to its original path below <dir>:
  models/experts/<group>/<task>/<run>/...  -> runs/logs/rsl_rl/hnu_track_<task>/<run>/...
  models/students/students_*/...           -> runs/students_*/...
  models/skill_encoders/latent_*/...       -> runs/latent_*/...
  models/deploy_exports/...                -> deploy_exports/...   (scenes from models/deploy_exports/_scenes/<md5>.mjb)
data/ and runs/results_* are already in place after download_assets.sh.

With --prefix-copy <code copy>, the code/ folder of this repository is also copied to <code copy> with the absolute
paths of the training server (/root/gpufree-data/hnu_v2, /root/gpufree-data/venvs/mjlab/bin/python is left as is)
replaced by <dir>.  Without it, replace the prefix yourself (see the README).

  python scripts/setup_layout.py /data/hnu_v2 --prefix-copy /data/hnu_v2/code
"""
import argparse
import json
import os
import shutil

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVER = "/root/gpufree-data/hnu_v2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--prefix-copy", default=None)
    a = ap.parse_args()
    root = os.path.abspath(a.dir)
    man = json.load(open(os.path.join(REPO, "docs", "MANIFEST_models.json")))
    made = present = missing = 0
    for e in man:
        src = os.path.join(root, e.get("same_as") or e["file"])
        dst = os.path.join(root, e["source"].split("/", 1)[1])  # drop the server root (hnu_v2/ or hnu_v2_4090/)
        if not os.path.exists(src):
            missing += 1
            print("missing in the release:", os.path.relpath(src, root))
            continue
        if os.path.lexists(dst):
            present += 1
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        os.symlink(os.path.relpath(src, os.path.dirname(dst)), dst)
        made += 1
    print(f"{made} links created, {present} already present, {missing} files missing")
    if a.prefix_copy:
        out = os.path.abspath(a.prefix_copy)
        shutil.copytree(os.path.join(REPO, "code"), out, symlinks=True, dirs_exist_ok=True)
        n = 0
        for d, _, fs in os.walk(out):
            for f in fs:
                p = os.path.join(d, f)
                try:
                    s = open(p, encoding="utf-8").read()
                except (UnicodeDecodeError, OSError):
                    continue
                if SERVER in s:
                    open(p, "w", encoding="utf-8").write(s.replace(SERVER, root))
                    n += 1
        print(f"code copied to {out}, prefix replaced in {n} files")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
