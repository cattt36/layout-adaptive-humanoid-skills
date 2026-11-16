"""Closed-loop success with the pre-registered uncertainty (runs/preregistration_v6g10m.md, CV supplement): unit =
(trial, RL seed); point estimate = mean over the test trials of the per-trial mean over seeds; 95 % CI = bootstrap
stratified by fold (resample test trials within each fold, then the seeds), 10000 replicates; level differences
L_k - L0 paired within (trial, seed).  Uses whatever (fold, seed) results exist (interim) or the full design.
  python closed_loop_ci.py [seeds=1,2,3] [reps=10000] > runs/closed_loop_ci.md"""
import json
import os
import sys

import numpy as np

H = "/root/gpufree-data/hnu_v2"
SEEDS = [int(s) for s in (sys.argv[1] if len(sys.argv) > 1 else "1,2,3").split(",")]
REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 10000
LV = ["Z0", "Z1", "Z2", "Z3"]
CONDS = [("理想", "0", ""), ("动力学随机化 + 延迟", "1", ""), ("推扰", "0", "_push"), ("随机化 + 推扰", "1", "_push")]
TASKS = [(None, "合计"), ("CrouchedTraverse", "独木桥"), ("ThreeStepPoleRun", "三步桩"), ("FiveStepPoleRun", "五步桩")]
rng = np.random.default_rng()


STEM = os.environ.get("HNU_CI_STEM", "results_v6g10n")  # results_v6g10xm (condition A) / results_v6g10xmB (B)


def load(F, s, dr, P, x):
    stem = STEM + ("" if F == 1 else f"_f{F}")
    p = f"{H}/runs/{stem}_s{s}_dr{dr}/eval_student_{P}{x}.json"
    if not os.path.exists(p):
        return None
    return {c["clip"]: (c["task"], c["success_v2"]) for c in json.load(open(p))["clips"] if c["split"] == "test"}


def data(dr, x, task):
    """fold -> (trials, seeds available for all levels, array [level, trial, seed])."""
    out = {}
    for F in (1, 2, 3, 4):
        seeds = [s for s in SEEDS if all(load(F, s, dr, P, x) is not None for P in LV)]
        if not seeds:
            continue
        ref = load(F, seeds[0], dr, "Z0", x)
        trials = sorted(c for c, (t, _) in ref.items() if task is None or t == task)
        if not trials:
            continue
        A = np.array([[[load(F, s, dr, P, x)[c][1] for s in seeds] for c in trials] for P in LV])
        out[F] = (trials, seeds, A)
    return out


def estimate(d, idx=None):
    vals = []
    for F, (trials, seeds, A) in d.items():
        ti, si = (np.arange(len(trials)), np.arange(len(seeds))) if idx is None else idx[F]
        vals.append(A[:, ti][:, :, si].mean(2))  # [level, trial]
    return np.concatenate(vals, 1).mean(1)  # per level


def boot(d):
    est = []
    for _ in range(REPS):
        idx = {F: (rng.integers(0, len(t), len(t)), rng.integers(0, len(s), len(s))) for F, (t, s, _) in d.items()}
        est.append(estimate(d, idx))
    return np.array(est)


units = {F: s for F, (t, s, _) in data("0", "", None).items()}
print(f"# 闭环成功率（{STEM}）：均值 [95% 置信区间]（预注册方法：单位为（试次，种子），按折分层 bootstrap，{REPS} 次）\n")
print("已有的（折：种子）：" + "；".join(f"第 {F} 折：{units[F]}" for F in sorted(units)) + "\n")
for cn, dr, x in CONDS:
    print(f"## {cn}\n")
    print("| 动作 | L0 | L1 | L2 | L3 | L3 − L0（配对） | L2 − L0 | L1 − L0 | 趋势（每级，配对） |")
    print("|---|---|---|---|---|---|---|---|---|")
    C = np.array([-1.5, -0.5, 0.5, 1.5]) / 5.0  # linear trend contrast = least-squares slope per level
    for t, tn in TASKS:
        d = data(dr, x, t)
        if not d:
            continue
        e, B = estimate(d), boot(d)
        ci = lambda v: f"{100 * np.percentile(v, 2.5):.1f}, {100 * np.percentile(v, 97.5):.1f}"  # noqa: E731
        cells = [f"{100 * e[i]:.1f} [{ci(B[:, i])}]" for i in range(4)]
        diffs = [f"{100 * (e[k] - e[0]):+.1f} [{ci(B[:, k] - B[:, 0])}]" for k in (3, 2, 1)]
        trend = f"{100 * float(C @ e):+.2f} [{ci(B @ C)}]"
        print(f"| {tn} | " + " | ".join(cells) + " | " + " | ".join(diffs) + f" | {trend} |")
    print()
print("非劣效判定（预注册）：L3 − L0 的 95% CI 下界 > −5 个百分点；优越性：下界 > 0。"
      "趋势：成功率对层级 0–3 的最小二乘斜率（每级百分点），配对 bootstrap；CI 下界 > 0 为单调上升。")
