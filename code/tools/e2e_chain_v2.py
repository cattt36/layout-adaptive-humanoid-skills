"""Pre-registered secondary experiment A (end-to-end simulated deployment), fold 1:
instruction -> z_sem -> router v2 with abstention -> z_skill -> student.  Per level and latent seed the routing of the
sealed test instruction set (language_routing_testv2.json, scored once ) is combined with the closed-loop
success (criterion v2) of the level's students on the expected action's TEST clips:
  in-scope instruction (abstain = false): success = routed correctly AND not rejected, times S(level, action)
  -> routing errors and wrong rejections count as failures;  out-of-scope (abstain = true): correct rejection rate.
  python e2e_chain_v2.py <latent_root> <results_dir> [<results_dir> ...]     (one results dir per RL seed)"""
import glob
import json
import os
import sys

import numpy as np

root, res_dirs = sys.argv[1], sys.argv[2:]
RF = os.environ.get("HNU_ROUTING_FILE", "language_routing_testv2.json")


def exec_success(rdir, prof):
    j = json.load(open(os.path.join(rdir, f"eval_student_{prof}.json")))
    out = {}
    for c in j["clips"]:
        if c["split"] == "test":
            out.setdefault(c["task"], []).append(c.get("success_v2", c["success"]))
    return {k: float(np.mean(v)) for k, v in out.items()}


print(f"routing file: {RF};  results: {res_dirs}")
print("| Level | in-scope instructions | route correct | execution success (test) | **E2E success** | out-of-scope: correct rejection |")
print("|---|---|---|---|---|---|")
for prof in ["Z0", "Z1", "Z2", "Z3", "Z4"]:
    routes = [json.load(open(f, encoding="utf-8")) for f in sorted(glob.glob(os.path.join(root, "seed*", prof, RF)))]
    R = {"route": [], "exec": [], "e2e": [], "rej": []}
    n_in = 0
    for rd in res_dirs:
        if not os.path.exists(os.path.join(rd, f"eval_student_{prof}.json")):
            continue
        S = exec_success(rd, prof)
        R["exec"].append(np.mean(list(S.values())))
        for r in routes:
            ins = [x for x in r if not x["abstain"]]
            oos = [x for x in r if x["abstain"]]
            n_in = len(ins)
            ok = np.array([(not x.get("rejected", False)) and x["routed"] == x["expected"] for x in ins], float)
            R["route"].append(ok.mean())
            R["e2e"].append(np.mean([o * S.get(x["expected"], 0.0) for o, x in zip(ok, ins)]))
            if oos:
                R["rej"].append(np.mean([bool(x.get("rejected", False)) for x in oos]))
    f = lambda k: f"{100 * np.mean(R[k]):.1f} ± {100 * np.std(R[k]):.1f}" if R[k] else "n/a"  # noqa: E731
    print(f"| L{prof[1]} | {n_in} | {f('route')} | {f('exec')} | **{f('e2e')}** | {f('rej')} |  (latent seeds {len(routes)}, RL seeds {len(R['exec'])})")
