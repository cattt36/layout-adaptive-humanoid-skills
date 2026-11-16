"""Fairness audit of the students (L40 server): (1) the running distill_train processes -- HNU_* environment and command
line compared across levels; (2) every args.json / env.json of v6g10n students.  Expected differences only: level
(profile, latent dir / z tables, obs group of the baselines), fold (data dirs, split), seed, output / TensorBoard dirs."""
import glob
import json
import os
import re

H = "/root/gpufree-data/hnu_v2/runs"
LEVEL_KEYS = {"profile", "latent_dir", "out", "init_from", "obs_group", "seed"}
ENV_LEVEL = {"HNU_TB_ROOT"}


def norm(s):  # remove the level / fold / seed from a value so that the rest can be compared
    s = re.sub(r"/Z[0-4]\b", "/Z?", str(s))
    s = re.sub(r"_f[1-4]\b|fold[1-4]|cv_f[1-4]", "<fold>", s)
    return re.sub(r"_s[0-9]\b|_d2_s[0-9]\b", "_s?", s)


print("== running processes ==")
procs = []
for p in os.listdir("/proc"):
    if not p.isdigit():
        continue
    try:
        cmd = open(f"/proc/{p}/cmdline").read().split("\0")
    except OSError:
        continue
    if len(cmd) > 1 and cmd[1].endswith("distill_train.py"):
        env = dict(x.split("=", 1) for x in open(f"/proc/{p}/environ").read().split("\0") if "=" in x)
        prof = cmd[cmd.index("--profile") + 1]
        procs.append((prof, {k: v for k, v in env.items() if k.startswith("HNU_")}, ["<P>" if x == prof else x for x in cmd[2:]]))
if procs:
    ref_env = {k: norm(v) for k, v in procs[0][1].items() if k not in ENV_LEVEL}
    ref_cmd = [norm(x) for x in procs[0][2]]
    for prof, env, cmd in procs:
        e = {k: norm(v) for k, v in env.items() if k not in ENV_LEVEL}
        de = sorted(k for k in set(e) | set(ref_env) if e.get(k) != ref_env.get(k))
        dc = [(a, b) for a, b in zip([norm(x) for x in cmd], ref_cmd) if a != b]
        print(f"  {prof:16s} HNU vars {len(env):2d}  env diffs vs {procs[0][0]}: {de or 'none'}  cmd diffs: {dc or 'none'}"
              f"{'  (length differs)' if len(cmd) != len(ref_cmd) else ''}")
print("== args.json of v6g10n students ==")
files = sorted(glob.glob(f"{H}/students_v6g10n*_s*/*/args.json"))
files = [f for f in files if "4090paused" not in f and "oldz" not in f]
ref = None
for f in files:
    a = json.load(open(f))
    a = {k: norm(v) for k, v in a.items() if k not in LEVEL_KEYS}
    stage = "d2" if "_d2_" in f else "r1"
    if ref is None or stage not in ref:
        ref = (ref or {}) | {stage: (f, a)}
        continue
    d = sorted(k for k in set(a) | set(ref[stage][1]) if a.get(k) != ref[stage][1].get(k))
    if d:
        print(f"  {f.replace(H + '/', '')}: differs in {d}")
print(f"  checked {len(files)} args.json; reference per stage: {', '.join(v[0].replace(H + '/', '') for v in (ref or {}).values())}")
