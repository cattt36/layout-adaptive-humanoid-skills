"""student stage summary (tools/st_base.sh): test-clip evaluation (fold 1) and sim2sim per planned layout,
new students vs the deployed one.  Writes runs/st_base/summary.md.  Run with any python that has numpy."""
import json, os, numpy as np
H = "/root/gpufree-data/hnu_v2"; R = f"{H}/runs/st_base"; RS = "/root/gpufree-data/robot_sim_check"
names = [n for n in ("deployed", "S2k", "S4k") if os.path.exists(f"{R}/{n}_A_b1_dr0/eval_student_Z3.json")]
TASKS = ("ThreeStepPoleRun", "FiveStepPoleRun", "CrouchedTraverse")
out = ["# Student stage (post bases)\n", "## Test clips (fold 1), success v2 %, floor touches per rollout (stakes)\n"]
for cond in ("A", "B"):
    out += [f"\n### condition {cond}\n", "| student | bases | DR | " + " | ".join(TASKS) + " | touches 3PR / 5PR |",
            "|---|---|---|---|---|---|---|"]
    for n in names:
        for b in (1, 0):
            for dr in (0, 1):
                f = f"{R}/{n}_{cond}_b{b}_dr{dr}/eval_student_Z3.json"
                if not os.path.exists(f):
                    continue
                s = json.load(open(f))["summary"]
                g = lambda t, k: s.get(f"{t}/test", {}).get(k, float("nan"))
                out.append(f"| {n} | {b} | {dr} | " + " | ".join(f"{100 * g(t, 'success_v2'):.1f}" for t in TASKS)
                           + f" | {g(TASKS[0], 'floor_touch_events'):.2f} / {g(TASKS[1], 'floor_touch_events'):.2f} |")
outs = [n for n in ("deployed", "S2k", "S4k") if os.path.isdir(f"{RS}/out_base_{n}")]
lays = sorted({f.rsplit("_", 1)[0] for f in os.listdir(f"{RS}/out_base_deployed") if f.endswith(".json")})
out += ["\n## sim2sim per planned layout (bases in the scene; stakes unaffected), success v2 % A / B / C\n",
        "| layout | " + " | ".join(outs) + " | max drop vs deployed (deployed - new, > 0 = worse) |", "|---|" + "---|" * (len(outs) + 1)]
for l in lays:
    v = {}
    for n in outs:
        v[n] = []
        for c in "ABC":
            f = f"{RS}/out_base_{n}/{l}_{c}.json"
            v[n].append(100 * np.mean([r["success_v2"] for r in json.load(open(f))["rows"]]) if os.path.exists(f) else np.nan)
    drop = {n: max(np.array(v["deployed"]) - np.array(v[n])) for n in outs if n != "deployed"}
    out.append(f"| {l} | " + " | ".join(" / ".join(f"{x:.0f}" for x in v[n]) for n in outs) + " | "
               + ", ".join(f"{n} {d:+.0f}" for n, d in drop.items()) + " |")
for n in outs:  # handbook v7 expectations (tools/robot_plan_v7.py, HNU_PLAN_SIM): {layout id: [A %, C %, falls in C]}
    sim = {}
    for l in lays:
        rows = {c: json.load(open(f"{RS}/out_base_{n}/{l}_{c}.json"))["rows"] for c in "AC"
                if os.path.exists(f"{RS}/out_base_{n}/{l}_{c}.json")}
        if len(rows) == 2:
            sim[l.split("_")[0]] = [round(100 * np.mean([r["success_v2"] for r in rows[c]])) for c in "AC"] \
                + [sum(r["cause"] == "fall" for r in rows["C"])]
    json.dump(sim, open(f"{R}/plan_sim_{n}.json", "w"))
open(f"{R}/summary.md", "w").write("\n".join(out) + "\n")
print("\n".join(out))
