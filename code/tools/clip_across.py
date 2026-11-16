"""Success of one clip across levels / seeds / teacher, and the effect of a uniform exclusion rule.
  python tools/clip_across.py <clip> [V]
"""
import json
import os
import sys

clip = sys.argv[1]
V = sys.argv[2] if len(sys.argv) > 2 else "v6g10s"
H = "/root/gpufree-data/hnu_v2/runs"
T = clip.rsplit("_", 1)[0]
CONDS = [("理想", "0", ""), ("理想+推扰", "0", "_push"), ("鲁棒", "1", ""), ("鲁棒+推扰", "1", "_push")]


def rows(f):
    return {r["clip"]: r for r in json.load(open(f))["clips"]} if os.path.exists(f) else None


print(f"## {clip}: success % per level (seeds 1 / 2 / 3) and teacher\n")
print("| | " + " | ".join(c for c, _, _ in CONDS) + " |\n|---|---|---|---|---|")
tl = []
for _, dr, x in CONDS:
    r = rows(f"{H}/results_{V}_bank_teacher_dr{dr}/eval_teacher_teacher{x}.json")
    tl.append("-" if r is None else f"{100 * r[clip]['success']:.0f}")
print("| 教师 | " + " | ".join(tl) + " |")
for P in ["Z0", "Z1", "Z2", "Z3", "Z4"]:
    cells = []
    for _, dr, x in CONDS:
        v = []
        for s in (1, 2, 3):
            r = rows(f"{H}/results_{V}_s{s}_dr{dr}/eval_student_{P}{x}.json")
            v.append("-" if r is None else f"{100 * r[clip]['success']:.0f}")
        cells.append(" / ".join(v))
    print(f"| L{P[1]} | " + " | ".join(cells) + " |")

# overall action success with and without the clip, mean over finished seeds (same rule for every level)
print(f"\n## {T} success % with / without {clip} (mean over finished seeds)\n")
print("| | " + " | ".join(c for c, _, _ in CONDS) + " |\n|---|---|---|---|---|")
for P in ["teacher", "Z0", "Z1", "Z2", "Z3", "Z4"]:
    cells = []
    for _, dr, x in CONDS:
        fs = [f"{H}/results_{V}_bank_teacher_dr{dr}/eval_teacher_teacher{x}.json"] if P == "teacher" else \
             [f"{H}/results_{V}_s{s}_dr{dr}/eval_student_{P}{x}.json" for s in (1, 2, 3)]
        a, b = [], []
        for f in fs:
            r = rows(f)
            if r is None:
                continue
            tr = [v for v in r.values() if v["task"] == T]
            a.append(sum(v["success"] for v in tr) / len(tr))
            b.append(sum(v["success"] for v in tr if v["clip"] != clip) / (len(tr) - 1))
        cells.append("-" if not a else f"{100 * sum(a) / len(a):.1f} → {100 * sum(b) / len(b):.1f}")
    print(f"| {'教师' if P == 'teacher' else 'L' + P[1]} | " + " | ".join(cells) + " |")
