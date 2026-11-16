"""Bridge-scene comparison table: per results dir and level, success (clean / push) per action and CrouchedTraverse
no-contact success (all clips and unseen-subject test clips)."""
import json
import os
import sys

print("| version | level | W1 | W2 | DuMu | DuMu no-contact | DuMu test | DuMu push | DuMu push no-contact |")
print("|---|---|---|---|---|---|---|---|---|")
for R in sys.argv[1:]:
    name = os.path.basename(os.path.normpath(R))
    for P in ["teacher", "Z0", "Z1", "Z2", "Z3", "Z4"]:
        f = os.path.join(R, f"eval_{'teacher_teacher' if P == 'teacher' else 'student_' + P}.json")
        fp = f.replace(".json", "_push.json")
        if not os.path.exists(f):
            continue
        s = json.load(open(f))["summary"]
        sp = json.load(open(fp))["summary"] if os.path.exists(fp) else {}
        g = lambda d, k, m="success": f"{100 * d[k][m]:.1f}" if k in d and m in d[k] else "-"  # noqa: E731
        print(f"| {name} | {'teacher' if P == 'teacher' else 'L' + P[1]} | {g(s, 'ThreeStepPoleRun/all')} | {g(s, 'FiveStepPoleRun/all')} | "
              f"{g(s, 'CrouchedTraverse/all')} | {g(s, 'CrouchedTraverse/all', 'success_no_contact')} | {g(s, 'CrouchedTraverse/test')} | "
              f"{g(sp, 'CrouchedTraverse/all')} | {g(sp, 'CrouchedTraverse/all', 'success_no_contact')} |")
