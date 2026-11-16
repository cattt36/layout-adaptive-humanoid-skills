"""Per-clip success of two students (or one student across conditions) for one action.
  python tools/clip_compare.py <task> <label=eval.json> <label=eval.json> ...
"""
import json
import sys

T = sys.argv[1]
runs = [a.split("=", 1) for a in sys.argv[2:]]
data = {lab: {r["clip"]: r for r in json.load(open(f))["clips"] if r["task"] == T} for lab, f in runs}
clips = sorted(next(iter(data.values())))
print(f"{'clip':18s} split  " + "  ".join(f"{lab:>12s}" for lab, _ in runs))
for c in clips:
    vals = [data[lab][c]["success"] for lab, _ in runs]
    if min(vals) < 1:
        print(f"{c:18s} {data[runs[0][0]][c]['split']:5s}  " + "  ".join(f"{100 * v:11.1f}%" for v in vals))
for lab, _ in runs:
    r = list(data[lab].values())
    print(f"{lab}: mean {100 * sum(x['success'] for x in r) / len(r):.1f}%  over {len(r)} clips x trials")
