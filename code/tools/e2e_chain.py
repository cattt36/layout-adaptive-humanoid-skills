"""End-to-end deployment chain per level: instruction -> z_sem routing -> z_skill -> student execution.

For every resolvable benchmark instruction the routed action comes from language_routing.json (eval_language.py with
HNU_LANG_DUMP=1, one per latent seed); the instruction succeeds when the route is correct AND the level's student
completes the action (closed-loop success of that student on the action's unseen-subject test clips).  Routing and
execution are evaluated independently, so E2E = mean_q [route correct] * S(level, expected action).

  python e2e_chain.py <latent_root> <results_dir> [<results_dir> ...]   (one results dir per RL seed)
"""
import glob
import json
import os
import sys

import numpy as np

root, res_dirs = sys.argv[1], sys.argv[2:]


def exec_success(rdir, prof, push):
    j = json.load(open(os.path.join(rdir, f"eval_student_{prof}{'_push' if push else ''}.json")))
    out = {}
    for c in j["clips"]:
        if c["split"] == "test":
            out.setdefault(c["task"], []).append(c["success"])
    return {k: float(np.mean(v)) for k, v in out.items()}


print("| Level | route acc (resolvable) | exec success (test, clean) | E2E clean | E2E push+noise |")
print("|---|---|---|---|---|")
for prof in ["Z0", "Z1", "Z2", "Z3", "Z4"]:
    rf = os.environ.get("HNU_ROUTING_FILE", "language_routing.json")  # language_routing_ft.json: fine-tuned router
    routes = [json.load(open(f, encoding="utf-8")) for f in sorted(glob.glob(os.path.join(root, "seed*", prof, rf)))]
    rows = {"route": [], "exec": [], "e2e": [], "e2e_push": []}
    for rd in res_dirs:
        if not os.path.exists(os.path.join(rd, f"eval_student_{prof}.json")):
            continue
        S, Sp = exec_success(rd, prof, False), exec_success(rd, prof, True)
        rows["exec"].append(np.mean(list(S.values())))
        for r in routes:
            q = [x for x in r if not x["abstain"]]
            ok = np.array([x["routed"] == x["expected"] for x in q], float)
            rows["route"].append(ok.mean())
            rows["e2e"].append(np.mean([o * S[x["expected"]] for o, x in zip(ok, q)]))
            rows["e2e_push"].append(np.mean([o * Sp[x["expected"]] for o, x in zip(ok, q)]))
    f = lambda k: f"{100 * np.mean(rows[k]):.1f} ± {100 * np.std(rows[k]):.1f}" if rows[k] else "n/a"  # noqa: E731
    print(f"| L{prof[1]} | {f('route')} | {f('exec')} | {f('e2e')} | {f('e2e_push')} |  (n_latent={len(routes)}, n_rl={len(rows['exec'])})")
