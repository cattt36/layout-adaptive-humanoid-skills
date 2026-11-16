"""Pilot generalisation table: success % per split (train / val / test) and action, clean evaluation."""
import json
import os

R = "/root/gpufree-data/hnu_v2/runs"
RUNS = [("L0 no aug (4000+2000)", "results_v6g10h_s1", "Z0"), ("L1 no aug (4000+2000)", "results_v6g10h_s1", "Z1"),
        ("L0 + aug (2000+1000)", "results_v6g10p_s1", "Z0A"), ("L4 + aug (2000+1000)", "results_v6g10p_s1", "Z4A"),
        ("explicit ref (2000+1000)", "results_v6g10p_s1", "REF")]
for dr, name in (("0", "nominal"), ("1", "randomised")):
    print(f"\n== {name} (clean): success % train / val / test")
    for lab, d, tag in RUNS:
        f = f"{R}/{d}_dr{dr}/eval_student_{tag}.json"
        if not os.path.exists(f):
            continue
        s = json.load(open(f))["summary"]
        cells = []
        for t in ("CrouchedTraverse", "ThreeStepPoleRun", "FiveStepPoleRun"):
            cells.append(f"{t[:6]} " + "/".join(f"{100 * s[f'{t}/{k}']['success']:.0f}" for k in ("train", "val", "test")))
        be = sum(s[f"{t}/test"]["body_err"] for t in ("CrouchedTraverse", "ThreeStepPoleRun", "FiveStepPoleRun")) / 3
        print(f"  {lab:26s} " + " | ".join(cells) + f" | test body err {100 * be:.1f} cm")
