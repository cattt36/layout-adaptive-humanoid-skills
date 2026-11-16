"""8.3.2 pooled over all new student pairs (tools/deploy_v2_run.sh; L0 / L3 x folds 1-4 x RL seeds 1-3), .
Estimator as pre-registered for the closed loop: unit = (test clip, RL seed); per clip the mean over seeds, then the
mean over clips; 95 % CI = bootstrap stratified by fold (clips and seeds resampled within each fold), L3 - L0 paired.
sim2sim settings from deploy_exports/v6g10xm_f<F>_s<S>/L*/sweeps/<name>.json (8 trials per clip); experiment C from
runs/results_v6g10xmC<d>[_f<F>]_s<S>_dr0 (mjlab, 32 rollouts per clip).
  python deploy_pool.py > runs/deploy_pool.md"""
import json
import os

import numpy as np

H = "/root/gpufree-data/hnu_v2"
REPS = 2000
rng = np.random.default_rng(3)
TASKS = [("CrouchedTraverse", "独木桥"), ("ThreeStepPoleRun", "三步桩"), ("FiveStepPoleRun", "五步桩")]
GROUPS = [
    ("基准（0–20 ms 延迟）", ["base"]),
    ("固定 20 ms 延迟", ["lat20"]),
    ("感知固定偏差 m", ["perc_bias0.02", "perc_bias0.05", "perc_bias0.08", "perc_bias0.12"]),
    ("感知逐帧噪声 m", ["perc_noise0.02", "perc_noise0.05", "perc_noise0.10"]),
    ("摆放：整体平移 m", ["shift0.02", "shift0.05", "shift0.08", "shift0.12"]),
    ("摆放：每个桩独立偏移 m（三步桩/五步桩）", ["stake_jitter0.02", "stake_jitter0.05", "stake_jitter0.08"]),
    ("搭建：横梁高度 m（独木桥）", ["bridge_dz-0.05", "bridge_dz-0.03", "bridge_dz0.03", "bridge_dz0.05"]),
    ("起始朝向误差（度）", ["yaw3", "yaw5", "yaw10"]),
    ("新增（保守）：桩距整体缩放（三步桩/五步桩，重新规划）", ["spacing0.90", "spacing0.95", "spacing1.05", "spacing1.10"]),
    ("新增（保守）：起点到第一根桩的距离 m（三步桩/五步桩，重新规划）", ["start-0.10", "start-0.05", "start+0.05", "start+0.10"]),
    ("新增（保守）：桥柱间距变化 m（独木桥）", ["post-0.05", "post-0.03", "post+0.03", "post+0.05"]),
    ("里程计漂移（无外部定位，补充）", ["drift0.01", "drift0.02", "drift0.05", "drift0.10"]),
]
TEST = {F: set(json.load(open(f"{H}/data/clip_split_cv/fold{F}/split.json"))["test"]) for F in (1, 2, 3, 4)}


def sweep(F, S, L, name):
    f = f"{H}/deploy_exports/v6g10xm_f{F}_s{S}/{L}/sweeps/{name}.json"
    if not os.path.exists(f):
        return None
    per = {}
    for r in json.load(open(f))["rows"]:
        if r["clip"] in TEST[F]:
            per.setdefault((r["clip"], r["task"]), []).append(r["success_v2"])
    return {k: float(np.mean(v)) for k, v in per.items()}


def expc(F, S, L, d):
    f = f"{H}/runs/results_v6g10xmC{d}{'' if F == 1 else f'_f{F}'}_s{S}_dr0/eval_student_Z{L[1]}.json"
    if not os.path.exists(f):
        return None
    return {(c["clip"], c["task"]): float(c["success_v2"]) for c in json.load(open(f))["clips"] if c["clip"] in TEST[F]}


def table(getter, L):
    """fold -> {(clip, task): {seed: value}}"""
    out = {}
    for F in (1, 2, 3, 4):
        for S in (1, 2, 3):
            v = getter(F, S, L)
            for k, x in (v or {}).items():
                out.setdefault(F, {}).setdefault(k, {})[S] = x
    return out


def est(T, task=None, pick=None):
    v = []
    for F, cl in T.items():
        keys = [k for k in cl if task is None or k[1] == task] if pick is None else [k for k in pick[F][0] if k in cl and (task is None or k[1] == task)]
        for k in keys:
            ss = cl[k] if pick is None else {s: cl[k][s] for s in pick[F][1] if s in cl[k]}
            if not ss:
                ss = cl[k]
            v.append(np.mean(list(ss.values())))
    return 100 * np.mean(v) if v else np.nan


def boot(T3, T0):
    d = []
    for _ in range(REPS):
        pick = {}
        for F, cl in T3.items():
            ks = sorted(cl)
            seeds = sorted({s for x in cl.values() for s in x})
            pick[F] = ([ks[i] for i in rng.integers(len(ks), size=len(ks))], list(rng.choice(seeds, len(seeds))))
        d.append(est(T3, pick=pick) - est(T0, pick=pick))
    return np.nanpercentile(d, [2.5, 97.5])


def n_units(T):
    return sum(len(c) for c in T.values()), sorted({(F, s) for F, cl in T.items() for x in cl.values() for s in x})


T_any = table(lambda F, S, L: sweep(F, S, L, "base"), "L3")
nu, pairs = n_units(T_any)
print("# 8.3.2 部署条件（全部新学生 L0 / L3 汇总；sim2sim：CPU MuJoCo，0–20 ms 延迟，成功标准 v2，每片段 8 次）\n")
print(f"已有的（折，种子）：{pairs}；单位 =（test 试次，RL 种子），先对种子取平均再对试次取平均；95% CI 为按折分层 bootstrap。"
      "链路 = 库 z + 规划器 plan4。\n")
for gname, names in GROUPS:
    print(f"## {gname}\n")
    print("| 设置 | L3 | L0 | L3 − L0 [95% CI] | " + " | ".join(f"L3 {tn} | L0 {tn}" for _, tn in TASKS) + " |")
    print("|---|---|---|---|" + "---|---|" * len(TASKS))
    for nm in names:
        T3, T0 = table(lambda F, S, L: sweep(F, S, L, nm), "L3"), table(lambda F, S, L: sweep(F, S, L, nm), "L0")
        if not T3 or not T0:
            continue
        lo, hi = boot(T3, T0)
        lab = nm
        for a, b in (("perc_bias", ""), ("perc_noise", ""), ("stake_jitter", ""), ("shift", ""), ("bridge_dz", ""), ("yaw", ""),
                     ("drift", ""), ("lat20", "20 ms"), ("base", "–"), ("spacing", "×"), ("start", ""), ("post", "")):
            lab = lab.replace(a, b)
        print(f"| {lab} | {est(T3):.1f} | {est(T0):.1f} | {est(T3) - est(T0):+.1f} [{lo:+.1f}, {hi:+.1f}] | "
              + " | ".join(f"{est(T3, t):.0f} | {est(T0, t):.0f}" for t, _ in TASKS) + " |")
    print()
print("## 动捕中断（实验 C；障碍物特征冻结时长 s；mjlab 评估，每片段 32 次）\n")
print("| 冻结时长 | L3 | L0 | L3 − L0 [95% CI] | " + " | ".join(f"L3 {tn} | L0 {tn}" for _, tn in TASKS) + " |")
print("|---|---|---|---|" + "---|---|" * len(TASKS))
for d in ("0", "0.25", "0.5", "1.0"):
    T3, T0 = table(lambda F, S, L: expc(F, S, L, d), "L3"), table(lambda F, S, L: expc(F, S, L, d), "L0")
    if T3 and T0:
        lo, hi = boot(T3, T0)
        print(f"| {d} | {est(T3):.1f} | {est(T0):.1f} | {est(T3) - est(T0):+.1f} [{lo:+.1f}, {hi:+.1f}] | "
              + " | ".join(f"{est(T3, t):.0f} | {est(T0, t):.0f}" for t, _ in TASKS) + " |")
