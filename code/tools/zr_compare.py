"""Fair external baseline ZR (explicit-reference student, run_zr_l40.sh / run_zr_l40_s23.sh) vs the new z students,
closed loop (; seeds 1-3 added ).  Folds 1-4, TEST trials, criterion v2 (task); conditions A (own
z / own reference, human clock) and B (library + planner plan4) x 4 environments.  Unit = (test trial, RL seed) as in
8.2 / 8.4: per trial the mean over the seeds, then the mean over trials; 95 % CI = bootstrap stratified by fold (trials
and seeds resampled within each fold, the same draw for all levels); differences paired on trials and seeds.
A (fold, seed) enters only when ZR, L0 and L3 all have the result.
  HNU_SEEDS="1 2 3" python zr_compare.py > runs/zr_compare.md        (default seeds: 1)"""
import json
import os

import numpy as np

H = "/root/gpufree-data/hnu_v2"
rng = np.random.default_rng(5)
SEEDS = [int(s) for s in os.environ.get("HNU_SEEDS", "1").split()]
LV = [("ZR", "ZR（显式参考）"), ("Z0", "L0"), ("Z3", "L3")]
# display names (server data keep the old ids until the server-side rename)
TASKS = [("CrouchedTraverse", "独木桥"), ("ThreeStepPoleRun", "三步桩"), ("FiveStepPoleRun", "三步桩/五步桩")]
ENVS = [("dr0", "", "理想"), ("dr1", "", "随机化 + 延迟"), ("dr0", "_push", "推扰"), ("dr1", "_push", "两者叠加")]
TEST = {F: set(json.load(open(f"{H}/data/clip_split_cv/fold{F}/split.json"))["test"]) for F in (1, 2, 3, 4)}


def load(cond, F, s, dr, push, P):
    stem = f"results_v6g10xm{'B' if cond == 'B' else ''}{'' if F == 1 else f'_f{F}'}_s{s}_{dr}"
    f = f"{H}/runs/{stem}/eval_student_{P}{push}.json"
    if not os.path.exists(f):
        return None
    return {c["clip"]: (c["task"], float(c["success_v2"])) for c in json.load(open(f))["clips"] if c["clip"] in TEST[F]}


def table(cond, dr, push):
    """T[F] = dict(seeds=[...], clips={clip: {"task": t, P: np.array over seeds}})"""
    T = {}
    for F in (1, 2, 3, 4):
        per = {}
        for s in SEEDS:
            d = {P: load(cond, F, s, dr, push, P) for P, _ in LV}
            if all(v is not None for v in d.values()):
                per[s] = d
        if not per:
            continue
        common = set.intersection(*(set(v) for d in per.values() for v in d.values()))
        sl = sorted(per)
        T[F] = dict(seeds=sl, clips={c: {"task": per[sl[0]]["ZR"][c][0]} | {P: np.array([per[s][P][c][1] for s in sl]) for P, _ in LV}
                                     for c in sorted(common)})
    return T


def est(T, P, task=None, pick=None):
    v = []
    for F, t in T.items():
        cl = t["clips"]
        keys, si = pick[F] if pick else (list(cl), np.arange(len(t["seeds"])))
        v += [cl[c][P][si].mean() for c in keys if task is None or cl[c]["task"] == task]
    return 100 * np.mean(v) if v else np.nan


def ci(T, a, b, reps=4000):
    d = []
    for _ in range(reps):
        pick = {}
        for F, t in T.items():
            keys = list(t["clips"])
            pick[F] = ([keys[i] for i in rng.integers(len(keys), size=len(keys))], rng.integers(len(t["seeds"]), size=len(t["seeds"])))
        d.append(est(T, a, pick=pick) - est(T, b, pick=pick))
    return np.percentile(d, [2.5, 97.5])


print(f"# 公平基线 ZR vs 新学生（RL 种子 {SEEDS}，4 折 test 试次，成功标准 v2；单位 =（试次，种子），按折分层 bootstrap，配对）\n")
for cond, cname in (("A", "条件 A（自身参考 / 自身 z + 人的时钟）"), ("B", "条件 B（库片段 + 规划器 plan4，可部署）")):
    print(f"## {cname}\n")
    print("| 环境 | ZR | L0 | L3 | ZR − L0 [95% CI] | ZR − L3 [95% CI] | " + " | ".join(f"{tn} ZR / L0 / L3" for _, tn in TASKS) + " |")
    print("|---|---|---|---|---|---|" + "---|" * len(TASKS))
    for dr, push, en in ENVS:
        T = table(cond, dr, push)
        if not T:
            continue
        a0, a3 = ci(T, "ZR", "Z0"), ci(T, "ZR", "Z3")
        print(f"| {en} | {est(T, 'ZR'):.1f} | {est(T, 'Z0'):.1f} | {est(T, 'Z3'):.1f} | "
              f"{est(T, 'ZR') - est(T, 'Z0'):+.1f} [{a0[0]:+.1f}, {a0[1]:+.1f}] | {est(T, 'ZR') - est(T, 'Z3'):+.1f} [{a3[0]:+.1f}, {a3[1]:+.1f}] | "
              + " | ".join(f"{est(T, 'ZR', t):.0f} / {est(T, 'Z0', t):.0f} / {est(T, 'Z3', t):.0f}" for t, _ in TASKS) + " |")
    print()
for cond in ("A", "B"):
    T = table(cond, "dr0", "")
    print(f"条件 {cond} 已有（折：种子）：{ {F: t['seeds'] for F, t in T.items()} }")
print("每个（试次，种子）32 次运行。")
