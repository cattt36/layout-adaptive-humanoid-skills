"""Execution layer, idea "simulation preview" (exploratory, offline).  At deployment the layout geometry is
known (perception / layout sheet), so the top-5 retrieved candidates can be rolled out a few times in simulation before
the robot moves, and the best one executed.  Estimated without new simulation from the E2E candidate sweep (16
episodes per layout x candidate, ideal env, planner clock): episodes 0..n-1 are the preview, episodes 8..15 the
evaluation (never used for the choice).  Same command set as the execution layer of tools/expA_plan.py (accepted and
correctly routed commands of the frozen router x test layouts, top-5 of the retrieval ranking).
Limitation: preview and evaluation share the simulator (no model mismatch); the real-robot gap is not covered.
  python preview_select.py [who=xm] [seed=1] [eval_dr=0] [preview_dr=eval_dr] > runs/preview_<who>.md
robustness check (tools/preview_dr_sweep.sh): eval_dr=1 scores execution on the randomised + latency sweep
(episodes 8-15); preview_dr=0 previews in the ideal simulator, preview_dr=1 in the randomised one (episodes 0..n-1)."""
import json
import os
import sys

import numpy as np

H = "/root/gpufree-data/hnu_v2"
WHO = sys.argv[1] if len(sys.argv) > 1 else "xm"
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 1
EV_DR = int(sys.argv[3]) if len(sys.argv) > 3 else 0
PV_DR = int(sys.argv[4]) if len(sys.argv) > 4 else EV_DR
LV = ["Z0", "Z1", "Z2", "Z3"]
TASKS = ["CrouchedTraverse", "ThreeStepPoleRun", "FiveStepPoleRun"]
TOPK, EVAL = 5, slice(8, 16)
NS = [1, 2, 4, 8]
rng = np.random.default_rng(0)
R = {}
for fold in [1, 2, 3, 4]:
    FS = "" if fold == 1 else f"_f{fold}"
    PJ = json.load(open(f"{H}/data/z_cand_plan_f{fold}/pairs.json"))
    spl, K = PJ["split"], PJ["K"]
    RANK = json.load(open(f"{H}/runs/expA_rank{FS}.json"))
    for P in LV:
        EP = {}
        for dr in {EV_DR, PV_DR}:
            ep = EP.setdefault(dr, {})
            for k in range(K):
                d = f"{H}/runs/results_candplan_{WHO}{FS}_k{k:02d}_s{SEED}_dr{dr}"
                f = f"{d}/qexec_{P}.npz"
                if not os.path.exists(f):
                    continue
                z = np.load(f)
                names = [str(s) for s in z["names"]]
                for L, c in PJ["pairs"].get(f"k{k:02d}", {}).items():
                    if L in names and spl.get(L) == "test":
                        e = z["success_v2"][z["clip"] == names.index(L)].astype(float)
                        if len(e) >= 16:
                            ep[(L, c)] = e[:16]
        ep, pep = EP[EV_DR], EP[PV_DR]  # execution episodes / preview episodes
        if not ep or not pep:
            continue
        for r in RANK.get(f"42/{P}", []):
            if r["rejected"] or r["routed"] != r["expected"]:
                continue
            for L in [n for n in spl if n.startswith(r["expected"]) and spl[n] == "test"]:
                cands = [c for c in r["ranking"] if (L, c) in ep and (L, c) in pep][:TOPK]
                if not cands:
                    continue
                ev = {c: ep[(L, c)][EVAL].mean() for c in cands}
                row = dict(task=r["expected"], fold=fold, rank1=ev[cands[0]], random=np.mean(list(ev.values())), oracle=max(ev.values()))
                for n in NS:
                    pv = {c: pep[(L, c)][:n].mean() for c in cands}
                    best = max(pv.values())
                    tied = [c for c in cands if pv[c] == best]  # ties -> retrieval order
                    row[f"prev{n}"] = ev[tied[0]]
                R.setdefault(P, []).append(row)

M = [("rank1", "检索第 1"), ("random", "前 5 随机")] + [(f"prev{n}", f"预演 {n} 次") for n in NS] + [("oracle", "前 5 最好（上限）")]
ENVN = {0: "理想", 1: "随机化 + 延迟"}
print(f"# 执行层：仿真预演选择（student = {WHO}，种子 {SEED}；预演环境：{ENVN[PV_DR]}，执行评估环境：{ENVN[EV_DR]}）\n")
print("预演用每对的第 1–n 次运行，评估只用第 9–16 次（与预演不重叠）。指令集合同执行层（冻结路由器接受且动作正确）× test 布局，"
      "前 5 个检索候选。参考：同一集合上 Q_cand 选择约 74–78%。\n")
for t in [None] + TASKS:
    print(f"## {t or '合计'}\n")
    print("| 层级 | n | " + " | ".join(n for _, n in M) + " |")
    print("|---|---|" + "---|" * len(M))
    for P in LV:
        rows = [x for x in R.get(P, []) if t is None or x["task"] == t]
        if rows:
            print(f"| L{P[1]} | {len(rows)} | " + " | ".join(f"{100 * np.mean([x[m] for x in rows]):.1f}" for m, _ in M) + " |")
    print()
