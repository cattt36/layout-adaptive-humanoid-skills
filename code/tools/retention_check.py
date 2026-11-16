"""Retention check: does a fine-tuned expert still perform the ORIGINAL motion as well as the original expert?

Both experts are evaluated from frame 0, nominal (no randomisation), in the original setting of the action:
  flat        : HNU-Track-<task> (ThreeStepPoleRun / FiveStepPoleRun robust experts)
  bridge      : CrouchedTraverse bridge scene (bridge experts)
  stakes_flat : stakes task with 0.01 m stakes on the unscaled reference, no stake / drift terminations
                (stake experts vs the flat expert: is the original run-up and jumping still there?)
Verdict: retained if finetuned >= original - tolerance.

  python retention_check.py --task ThreeStepPoleRun --mode stakes_flat --orig <pt> --tuned <pt> --perc-tuned 1 [--tol 0.05]
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

H = "/root/gpufree-data/hnu_v2"
ap = argparse.ArgumentParser()
ap.add_argument("--task", required=True)
ap.add_argument("--mode", required=True, choices=["flat", "bridge", "stakes_flat"])
ap.add_argument("--orig", required=True)
ap.add_argument("--tuned", required=True)
ap.add_argument("--perc-orig", type=int, default=0)
ap.add_argument("--perc-tuned", type=int, default=0)
ap.add_argument("--tol", type=float, default=0.05)
ap.add_argument("--data-dir", default=None, help="stakes_flat: override the near-flat reference directory")
a = ap.parse_args()
code = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
res = {}
for tag, ck, perc in (("original", a.orig, a.perc_orig), ("finetuned", a.tuned, a.perc_tuned)):
    out = tempfile.mkdtemp(prefix=f"ret_{a.task}_{tag}_")
    env = dict(os.environ, HNU_DR="0", HNU_START_FRAC="0", HNU_PERCEPTION=str(perc))
    cmd = [sys.executable, "eval_expert.py", "--task", a.task, "--ckpt", ck, "--per-clip", "8", "--out", out]
    if a.mode == "bridge":
        cmd.append("--bridge")
    if a.mode == "stakes_flat":
        d = f"{H}/data/ref_stakes_s93_h01"  # 0.01 m stakes, stride 1.0 (built by tools/flat_in_stakes.sh)
        if a.task == "FiveStepPoleRun":
            d = f"{H}/data/ref_stakes_h01_w2"
            if not os.path.isdir(f"{d}/FiveStepPoleRun"):
                subprocess.run([sys.executable, "stage1_reference/make_stakes.py", "--task", "FiveStepPoleRun", "--ref",
                                f"{H}/data/ref_v1/FiveStepPoleRun", "--csv", f"{H}/data/scene/FiveStepPoleRun.csv", "--height", "0.01",
                                "--out", d], cwd=code, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        d = a.data_dir or d
        env.update(HNU_STAKE_H="0.01", HNU_ANCHOR_XY="100", HNU_STAKE_MISS_TERM="0", HNU_STAKES_ROOT=d, HNU_RUNUP_FREE_S="-1")
        cmd += ["--stakes", "--motion-dir", f"{d}/{a.task}", "--stake-top", "0.01"]
    subprocess.run(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=code)
    f = [x for x in os.listdir(out) if x.endswith(".json")]
    res[tag] = json.load(open(os.path.join(out, f[0])))["success_rate"] if f else float("nan")
ok = res["finetuned"] >= res["original"] - a.tol
print(json.dumps(dict(task=a.task, mode=a.mode, original=round(res["original"], 3), finetuned=round(res["finetuned"], 3),
                      tolerance=a.tol, retained=bool(ok))))
