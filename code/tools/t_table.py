"""Trial-level batch v6g10t: per student (level, seed) and teacher, test-trial results under the task criterion
(primary) and the tracking criterion (secondary), plus train / val for reference.
  python tools/t_table.py [V=v6g10t]
"""
import glob
import json
import os
import sys

V = sys.argv[1] if len(sys.argv) > 1 else "v6g10t"
R = "/root/gpufree-data/hnu_v2/runs"
TASKS = ["CrouchedTraverse", "ThreeStepPoleRun", "FiveStepPoleRun"]
CONDS = [("nominal", "0", ""), ("nominal+push", "0", "_push"), ("randomised", "1", ""), ("randomised+push", "1", "_push")]


def load(prefix, s, dr, tag, x):
    f = f"{R}/{prefix}_s{s}_dr{dr}/eval_student_{tag}{x}.json" if s else f"{R}/{prefix}_bank_teacher_dr{dr}/eval_teacher_teacher{x}.json"
    return json.load(open(f))["summary"] if os.path.exists(f) else None


models = [("teacher", None, None)]
for f in sorted(glob.glob(f"{R}/results_{V}_s*_dr0/eval_student_*.json")):
    if f.endswith("_push.json"):
        continue
    s = int(f.split(f"results_{V}_s")[1].split("_")[0])
    tag = os.path.basename(f)[len("eval_student_"):-5]
    models.append((f"{tag} s{s}", s, tag))
for crit, prefix in (("TASK (primary)", f"results_{V}"), ("TRACKING (secondary)", f"results_{V}trk")):
    print(f"\n===== {crit}: success % on TEST trials (train / val in brackets for the clean nominal case)")
    print("model        | " + " | ".join(f"{c:15s}" for c, _, _ in CONDS) + " | body err (test, nominal)")
    for lab, s, tag in models:
        cells, be = [], []
        for c, dr, x in CONDS:
            j = load(prefix, s, dr, tag, x)
            if j is None:
                cells.append("-")
                continue
            v = [100 * j[f"{t}/test"]["success"] for t in TASKS]
            cell = "/".join(f"{a:.0f}" for a in v)
            if c == "nominal":
                tv = "/".join(f"{100 * j[f'{t}/train']['success']:.0f}" for t in TASKS)
                vv = "/".join(f"{100 * j[f'{t}/val']['success']:.0f}" for t in TASKS)
                cell += f" [tr {tv} va {vv}]"
                be = [100 * j[f"{t}/test"]["body_err"] for t in TASKS]
            cells.append(cell)
        print(f"{lab:12s} | " + " | ".join(cells) + " | " + "/".join(f"{b:.1f}" for b in be))
    if crit.startswith("TASK"):
        print("  secondary (test, nominal): stakes all hit % W1/W2, foot-centre dist cm W1/W2, bridge no-contact %")
        for lab, s, tag in models:
            j = load(prefix, s, "0", tag, "")
            if j is None:
                continue
            sh = "/".join(f"{100 * j[f'{t}/test'].get('stakes_all_hit') :.0f}" if j[f"{t}/test"].get("stakes_all_hit") is not None else "-" for t in TASKS[1:])
            sd = "/".join(f"{100 * j[f'{t}/test'].get('stake_centre_dist_m'):.1f}" if j[f"{t}/test"].get("stake_centre_dist_m") is not None else "-" for t in TASKS[1:])
            print(f"   {lab:12s} stakes hit {sh}  centre dist {sd}  bridge no-contact {100 * j['CrouchedTraverse/test']['success_no_contact']:.0f}")
print("\n(cells: CrouchedTraverse / ThreeStepPoleRun / FiveStepPoleRun)")
