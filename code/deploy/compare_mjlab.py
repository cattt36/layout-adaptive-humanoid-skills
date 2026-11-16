"""Per-clip comparison: mjlab evaluation (eval_student json) vs sim2sim result json.
  python deploy/compare_mjlab.py <eval_student.json> <sim2sim.json>
"""
import json
import sys

ev = {r["clip"]: r for r in json.load(open(sys.argv[1]))["clips"]}
s2 = {}
for r in json.load(open(sys.argv[2]))["rows"]:
    s2.setdefault(r["clip"], []).append(r)
for c in sorted(ev):
    a = ev[c]["success"]
    b = [x["success"] for x in s2.get(c, [])]
    sb = sum(b) / len(b) if b else float("nan")
    if a < 1 or sb < 1:
        print(f"{c:18s} mjlab {100 * a:5.1f}%  sim2sim {100 * sb:5.1f}%  causes {[x['cause'] for x in s2.get(c, []) if x['cause']]}")
