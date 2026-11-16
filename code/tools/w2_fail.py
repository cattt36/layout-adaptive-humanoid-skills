"""Summarise runs/w2_fail: per trial and model, success, where the failures happen relative to the five stake
stance windows, and the per-stake hit rate (foot on top >= 50 % of the window)."""
import json
import os

import numpy as np

O = "/root/gpufree-data/hnu_v2/runs/w2_fail"
MODELS = [("teacher", "eval_teacher_teacher.json"), ("old L0", "eval_student_t_Z0.json"), ("old L4", "eval_student_t_Z4.json"),
          ("new L0", "eval_student_j_Z0.json"), ("new L4", "eval_student_j_Z4.json")]
data = {n: {c["clip"]: c for c in json.load(open(f"{O}/{f}"))["clips"]} for n, f in MODELS if os.path.exists(f"{O}/{f}")}
clips = sorted(c for c in next(iter(data.values())) if c.startswith("FiveStepPoleRun"))
for clip in clips:
    print(f"\n=== {clip}")
    for n in data:
        c = data[n][clip]
        d = c["detail"]
        win = d["stake_win"]
        ff = np.array(d["fail_frame"])
        ok = np.array(d["success"])
        on = np.array(d["stake_on"])
        # failure phase: before stake 1 / during or after stake k (last stake window started before the failure)
        ph = []
        for f, s in zip(ff, ok):
            if s or f < 0:
                continue
            k = sum(f >= w[0] for w in win)
            ph.append("pre" if k == 0 else f"S{k}")
        cnt = {p: ph.count(p) for p in ["pre", "S1", "S2", "S3", "S4", "S5"] if ph.count(p)}
        hit = (on >= 0.5).mean(0) * 100
        pre = [(f - win[0][0]) / 50 for f, s in zip(ff, ok) if not s and 0 <= f < win[0][0]]
        tpre = f" pre-S1 fails at {np.median(pre):+.2f} s from S1" if pre else ""
        print(f"  {n:8s} succ {100 * ok.mean():3.0f}%  fail at {cnt}{tpre}  hit per stake " +
              " ".join(f"{h:3.0f}" for h in hit))
    print(f"  stake windows (frames, 50 Hz): {win}")
