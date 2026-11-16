"""Failure types of the W1 teachers (nominal and randomised, excl. split-v2 exclusions): share of rollouts that
(a) succeed under v2, (b) fail by falling / not finishing (old criterion fails), (c) finish without falling but miss
>= 1 stake (old success, v2 failure)."""
import json

import numpy as np

H = "/root/gpufree-data/hnu_v2"
EXC = set(json.load(open(f"{H}/data/clip_split_trial/split_v2.json"))["excluded"])
SRC = {"current": f"{H}/runs/results_v6g10mv2_bank_teacher_dr{{dr}}/eval_teacher_teacher.json",
       "new (w1b)": f"{H}/runs/tf2_eval_final/dr{{dr}}/eval_teacher_teacher.json"}
for dr in ("0", "1"):
    print(f"== {'nominal' if dr == '0' else 'randomised'}")
    for name, f in SRC.items():
        rows = [c for c in json.load(open(f.format(dr=dr)))["clips"] if c["task"] == "ThreeStepPoleRun" and c["clip"] not in EXC]
        old = np.mean([c["success"] for c in rows])
        v2 = np.mean([c["success_v2"] for c in rows])
        print(f"   {name:10s} v2 success {100 * v2:4.1f} %  | fall / not finished {100 * (1 - old):4.1f} %  | "
              f"finished but a stake missed {100 * (old - v2):4.1f} %")
