"""8.4 execution-layer ablation, closed loop: new students (cross-trial z augmentation, results_v6g10xm*) vs
old students (conventional distillation, results_v6g10n*), L0-L3, conditions A (own z, human clock) and B (library z +
planner plan4), 4 environments.  Unit = (test trial, RL seed); per trial the mean over the seeds of each student type,
then the mean over trials; 95 % CI = bootstrap stratified by fold (trials and seeds resampled within each fold); the
difference new - old is paired on the trials.
  python new_vs_old.py [reps=10000] > runs/new_vs_old.md
HNU_LEVELS (default Z0,Z1,Z2,Z3) and HNU_FOLDS (default 1,2,3,4); HNU_LEVELS=ZR compares the explicit-
reference students with (results_v6g10xm*) and without (results_v6g10n*) rhythm decoupling."""
import json
import os
import sys

import numpy as np

H = "/root/gpufree-data/hnu_v2"
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
rng = np.random.default_rng(16)
LV = os.environ.get("HNU_LEVELS", "Z0,Z1,Z2,Z3").split(",")
FOLDS = [int(f) for f in os.environ.get("HNU_FOLDS", "1,2,3,4").split(",")]
ENVS = [("0", "", "理想"), ("1", "", "随机化 + 延迟"), ("0", "_push", "推扰"), ("1", "_push", "两者叠加")]
TASKS = [(None, "合计"), ("CrouchedTraverse", "独木桥"), ("ThreeStepPoleRun", "三步桩"), ("FiveStepPoleRun", "五步桩")]


def load(stem, F, s, dr, P, x):
    p = f"{H}/runs/{stem}{'' if F == 1 else f'_f{F}'}_s{s}_dr{dr}/eval_student_{P}{x}.json"
    if not os.path.exists(p):
        return None
    return {c["clip"]: (c["task"], c["success_v2"]) for c in json.load(open(p))["clips"] if c["split"] == "test"}


def data(cond, dr, x, P, task):
    """fold -> (trials, new array [trial, seed_n], old array [trial, seed_o])"""
    out = {}
    for F in FOLDS:
        arrs = []
        for stem in ((f"results_v6g10xm{'B' if cond == 'B' else ''}"), (f"results_v6g10n{'B' if cond == 'B' else ''}")):
            runs = {s: load(stem, F, s, dr, P, x) for s in (1, 2, 3)}
            arrs.append({s: r for s, r in runs.items() if r is not None})
        if not arrs[0] or not arrs[1]:
            continue
        common = set.intersection(*(set(r) for a in arrs for r in a.values()))
        trials = sorted(c for c in common if task is None or arrs[0][next(iter(arrs[0]))][c][0] == task)
        if not trials:
            continue
        out[F] = (trials, *[np.array([[a[s][c][1] for s in sorted(a)] for c in trials]) for a in arrs])
    return out


def est(d, idx=None):
    vn, vo = [], []
    for F, (trials, An, Ao) in d.items():
        if idx is None:
            vn.append(An.mean(1)); vo.append(Ao.mean(1))
        else:
            ti, sn, so = idx[F]
            vn.append(An[ti][:, sn].mean(1)); vo.append(Ao[ti][:, so].mean(1))
    return np.concatenate(vn).mean(), np.concatenate(vo).mean()


def boot(d):
    out = []
    for _ in range(REPS):
        idx = {F: (rng.integers(0, len(t), len(t)), rng.integers(0, An.shape[1], An.shape[1]), rng.integers(0, Ao.shape[1], Ao.shape[1]))
               for F, (t, An, Ao) in d.items()}
        out.append(est(d, idx))
    return np.array(out)


ci = lambda v: f"[{100 * np.percentile(v, 2.5):+.1f}, {100 * np.percentile(v, 97.5):+.1f}]"  # noqa: E731
print(f"# 8.4 新学生 vs 旧学生（闭环；单位 =（test 试次，种子）；按折分层 bootstrap {REPS} 次；差值按试次配对）\n")
for cond, cname in (("A", "条件 A（自身 z + 人的时钟）"), ("B", "条件 B（库 z + 规划器 plan4，可部署）")):
    print(f"## {cname}\n")
    for dr, x, en in ENVS:
        print(f"### {en}\n")
        print("| 动作 | " + " | ".join(f"{'L' + P[1] if P[0] == 'Z' and P[1].isdigit() else P} 新 / 旧 / 新 − 旧" for P in LV) + " |")
        print("|---|" + "---|" * len(LV))
        for t, tn in TASKS:
            cells = []
            for P in LV:
                d = data(cond, dr, x, P, t)
                if not d:
                    cells.append("–")
                    continue
                n, o = est(d)
                B = boot(d)
                cells.append(f"{100 * n:.1f} / {100 * o:.1f} / {100 * (n - o):+.1f} {ci(B[:, 0] - B[:, 1])}")
            print(f"| {tn} | " + " | ".join(cells) + " |")
        print()
seeds = {cond: {F: (sorted(s for s in (1, 2, 3) if load(f"results_v6g10xm{'B' if cond == 'B' else ''}", F, s, '0', LV[0], '')),
                    sorted(s for s in (1, 2, 3) if load(f"results_v6g10n{'B' if cond == 'B' else ''}", F, s, '0', LV[0], '')))
                for F in FOLDS} for cond in "AB"}
print("已有种子（折：新 / 旧）：" + "；".join(f"{c} " + ", ".join(f"f{F} {v[0]}/{v[1]}" for F, v in seeds[c].items()) for c in "AB"))
