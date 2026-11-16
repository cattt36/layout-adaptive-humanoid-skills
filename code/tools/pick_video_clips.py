"""Per test clip, success of teacher and L0-L4 students (v4d seed 1), to choose representative video clips."""
import json
import os
import sys

R = sys.argv[1]
files = {"T": "eval_teacher_teacher.json", **{f"Z{i}": f"eval_student_Z{i}.json" for i in range(5)}}
data = {k: json.load(open(os.path.join(R, v))) for k, v in files.items()}
succ = {k: {c["clip"]: c["success"] for c in d["clips"]} for k, d in data.items()}
for c in data["Z4"]["clips"]:
    if c["split"] == "test":
        print(f"{c['clip']:18s}", "  ".join(f"{k} {succ[k][c['clip']]:.2f}" for k in files))
