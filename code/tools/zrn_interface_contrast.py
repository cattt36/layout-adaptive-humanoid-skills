"""Interface contrast of the ER-conv analysis plan (runs/zrn_analysis_plan.md, secondary, descriptive; ):
(ER-dec - ER-conv) - (z-dec - z-conv) at L0 and L3.
  closed loop: conditions A and B, 4 environments, estimator of tools/new_vs_old.py (unit = (test trial, RL seed), seeds
               averaged per trial, trials paired across the four groups, seeds resampled per group)
  end to end : preview + preview refusal, router v4, estimator of tools/e2e_refusal_pool.py (unit = (instruction x
               layout, RL seed), the same resampled pairs and seeds for all four groups)
Fold-stratified bootstrap.  z-dec / z-conv = results_v6g10xm* / results_v6g10n* at level Z0 or Z3, ER-dec / ER-conv =
the same at level ZR (end to end: groups xm, n, zr, zrn of runs/expA_plan*_v4_zr_zrn_rows.json in hnu_v2_4090).
  [HNU_FOLDS=2,3,4] python zrn_interface_contrast.py [reps=10000] > runs/zrn_interface_contrast[_f234].md"""
import json
import os
import sys

import numpy as np

H = "/root/gpufree-data/hnu_v2/runs"
H4 = "/root/gpufree-data/hnu_v2_4090/runs"
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
FOLDS = [int(f) for f in os.environ.get("HNU_FOLDS", "1,2,3,4").split(",")]
rng = np.random.default_rng(23)
ENVS = [("0", "", "理想"), ("1", "", "随机化 + 延迟"), ("0", "_push", "推扰"), ("1", "_push", "两者叠加")]
NAMES = ["z 解耦", "z 不解耦", "ER 解耦", "ER 不解耦"]
ci = lambda v: f"[{100 * np.percentile(v, 2.5):+.1f}, {100 * np.percentile(v, 97.5):+.1f}]"  # noqa: E731


def load(stem, F, s, dr, P, x):
    p = f"{H}/{stem}{'' if F == 1 else f'_f{F}'}_s{s}_dr{dr}/eval_student_{P}{x}.json"
    if not os.path.exists(p):
        return None
    return {c["clip"]: float(c["success_v2"]) for c in json.load(open(p))["clips"] if c["split"] == "test"}


def cl_data(cond, dr, x, P):
    """fold -> [array (trial, seed)] for z-dec, z-conv, ER-dec, ER-conv on the common test trials"""
    B = "B" if cond == "B" else ""
    out = {}
    for F in FOLDS:
        groups = []
        for stem, lev in ((f"results_v6g10xm{B}", P), (f"results_v6g10n{B}", P), (f"results_v6g10xm{B}", "ZR"), (f"results_v6g10n{B}", "ZR")):
            runs = {s: load(stem, F, s, dr, lev, x) for s in (1, 2, 3)}
            groups.append({s: r for s, r in runs.items() if r is not None})
        if not all(groups):
            continue
        common = sorted(set.intersection(*(set(r) for g in groups for r in g.values())))
        if common:
            out[F] = [np.array([[g[s][c] for s in sorted(g)] for c in common]) for g in groups]
    return out


def cl_est(d, idx=None):
    v = [[], [], [], []]
    for F, arrs in d.items():
        for j, A in enumerate(arrs):
            if idx is None:
                v[j].append(A.mean(1))
            else:
                ti, ss = idx[F]
                v[j].append(A[ti][:, ss[j]].mean(1))
    return np.array([np.concatenate(a).mean() for a in v])


def cl_boot(d):
    out = []
    for _ in range(REPS):
        idx = {F: (rng.integers(0, len(a[0]), len(a[0])), [rng.integers(0, A.shape[1], A.shape[1]) for A in a]) for F, a in d.items()}
        out.append(cl_est(d, idx))
    return np.array(out)


D = {}
for F in FOLDS:
    for s in (1, 2, 3):
        f = f"{H4}/expA_plan{'' if F == 1 else f'_f{F}'}_s{s}_v4_zr_zrn_rows.json"
        if os.path.exists(f):
            D[(F, s)] = json.load(open(f))


def table(who, P):
    out = {}
    for (F, s), d in D.items():
        for x in d.get(f"{who}/{P}", []):
            if x["ok"] and "pv8" not in x:
                continue
            out.setdefault(F, {}).setdefault((x["instr"], x["layout"]), {})[s] = x
    return out


def e2e_est(T, pick=None):
    v = []
    for F, pairs in T.items():
        keys = list(pairs) if pick is None else [k for k in pick[F][0] if k in pairs]
        for k in keys:
            ss = pairs[k] if pick is None else ({s: pairs[k][s] for s in pick[F][1] if s in pairs[k]} or pairs[k])
            v.append(np.mean([x["pv8"] if x["ok"] and x.get("pv_accepted") else 0.0 for x in ss.values()]))
    return np.mean(v) if v else np.nan


def resample(T):
    pick = {}
    for F, pairs in T.items():
        ins, lay = sorted({k[0] for k in pairs}), sorted({k[1] for k in pairs})
        seeds = sorted({s for p in pairs.values() for s in p})
        bi, bl = rng.choice(ins, len(ins)), rng.choice(lay, len(lay))
        pick[F] = ([(i, l) for i in bi for l in bl if (i, l) in pairs], list(rng.choice(seeds, len(seeds))))
    return pick


def row(lab, m, B):
    ez, er = m[0] - m[1], m[2] - m[3]
    bz, br = B[:, 0] - B[:, 1], B[:, 2] - B[:, 3]
    return (f"| {lab} | " + " / ".join(f"{100 * a:.1f}" for a in m) + f" | {100 * ez:+.1f} {ci(bz)} | {100 * er:+.1f} {ci(br)} | "
            f"{100 * (er - ez):+.1f} {ci(br - bz)} |")


print(f"# 接口对比（分析计划次要项，描述性）：(ER 解耦 − ER 不解耦) − (z 解耦 − z 不解耦)；折 {FOLDS}；按折分层 bootstrap {REPS} 次\n")
HEAD = "| {} | " + " / ".join(NAMES) + " (%) | z：解耦 − 不解耦 | ER：解耦 − 不解耦 | 接口对比 ER − z |\n|---|---|---|---|---|"
for cond, cname in (("A", "闭环条件 A（自身技能 + 人的时钟）"), ("B", "闭环条件 B（库技能 + 规划器，可部署）")):
    print(f"## {cname}\n")
    print(HEAD.format("环境 / 层级"))
    for dr, x, en in ENVS:
        for P in ("Z0", "Z3"):
            d = cl_data(cond, dr, x, P)
            if d:
                print(row(f"{en}，L{P[1]}", cl_est(d), cl_boot(d)))
    print()
print("## 端到端（预演 + 预演拒识，路由器 v4；ER 用相应层级的路由）\n")
print(HEAD.format("层级"))
for P in ("Z0", "Z3"):
    Ts = [table(w, P) for w in ("xm", "n", "zr", "zrn")]
    if all(Ts):
        m = np.array([e2e_est(T) for T in Ts])
        B = []
        for _ in range(REPS):
            p = resample(Ts[0])
            B.append([e2e_est(T, p) for T in Ts])
        print(row(f"L{P[1]}", m, np.array(B)))
seeds = {F: sorted(s for (G, s) in D if G == F) for F in FOLDS}
print(f"\n端到端已有种子：{seeds}")
