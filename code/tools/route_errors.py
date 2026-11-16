"""Router v2 errors on the benchmark (resolvable cases): confusion matrix per level (summed over latent seeds),
per condition, and example misrouted commands; plus sample level-owned texts of L4.
  python tools/route_errors.py <latent_root> [tag=ft2]
"""
import glob
import json
import os
import sys
from collections import Counter

root = sys.argv[1]
tag = sys.argv[2] if len(sys.argv) > 2 else "ft2"
ACTS = ["CrouchedTraverse", "ThreeStepPoleRun", "FiveStepPoleRun"]
for prof in ["Z0", "Z4"]:
    conf, cond, wrong = Counter(), Counter(), Counter()
    for f in sorted(glob.glob(os.path.join(root, "seed*", prof, f"language_routing_{tag}.json"))):
        for r in json.load(open(f, encoding="utf-8")):
            if r["abstain"]:
                continue
            conf[(r["expected"], r["routed"])] += 1
            if r["routed"] != r["expected"]:
                cond[r["id"].split("_")[0]] += 1
                wrong[(r["query"], r["expected"], r["routed"])] += 1
    print(f"\n== L{prof[1]}: confusion (rows = expected, cols = routed; 3 latent seeds x 16 commands per action)")
    print("              " + "  ".join(f"{a[:10]:>10s}" for a in ACTS))
    for e in ACTS:
        print(f"  {e[:10]:10s}  " + "  ".join(f"{conf[(e, r)]:10d}" for r in ACTS))
    print("  errors by condition prefix:", dict(cond))
    for (q, e, r), n in wrong.most_common(8):
        print(f"   x{n} [{e} -> {r}] {q}")
