"""E2E with simulation preview and refusal (user request).  Rows of tools/expA_plan.py (keys q8, pv8, ...,
written when the per-rollout files exist); every method scored on rollouts 8-15 of each (layout, candidate) pair, the
preview uses rollouts 0-3.  Methods (selection + refusal):
  Q            : Q_cand choice, no refusal
  Q+Qrej       : Q_cand choice, refuse if the best Q score < tau (val layouts, 10 % quantile; tools/expA_plan.py)
  PV           : preview choice (all top-5 candidates x 4 rollouts), no refusal
  PV+Qrej      : preview choice, Q refusal
  PV+PVrej     : preview choice, refuse if the best candidate's preview success < 2/4   (adopted)
Per student type and level: end-to-end success over all in-scope (command x test layout) pairs (instruction refusal,
wrong action or execution refusal = failure), execution refusal rate and success of the executed commands over the
accepted & correctly routed pairs, fall rate per command (refused = no fall).  Estimator as tools/expA_pool.py (unit =
(command x layout, RL seed); seeds averaged; fold-stratified bootstrap).
  HNU_RTAG=v4 [HNU_WHO_EXTRA=zr] python e2e_refusal_pool.py > runs/e2e_refusal[_v4][_zr].md
HNU_REF_WHO (default xm) = reference group of the between-group comparisons in section 3, HNU_REPS
(default 2000) = bootstrap replicates, HNU_WHO_ONLY = restrict to these groups; e.g. HNU_WHO_EXTRA=zr,zrn
HNU_WHO_ONLY=zr,zrn HNU_REF_WHO=zr HNU_REPS=10000 for the primary comparison ER-dec - ER-conv.
HNU_FOLDS (default 1,2,3,4), e.g. HNU_FOLDS=2,3,4 for the folds not used in recipe selection."""
import json
import os

import numpy as np

H = "/root/gpufree-data/hnu_v2/runs"
RT = os.environ.get("HNU_RTAG", "")
EXTRA = [w for w in os.environ.get("HNU_WHO_EXTRA", "").split(",") if w]
SUF = (f"_{RT}" if RT else "") + ("_" + "_".join(EXTRA) if EXTRA else "")
WHO = [("xm", "新学生"), ("n", "旧学生")] + [(w, {"zr": "公平基线 ZR", "zrn": "ZR 不解耦"}.get(w, w)) for w in EXTRA]
ONLY = [w for w in os.environ.get("HNU_WHO_ONLY", "").split(",") if w]  # restrict all sections to these groups
WHO = [w for w in WHO if not ONLY or w[0] in ONLY]
REF = os.environ.get("HNU_REF_WHO", "xm")
LV = ["Z0", "Z1", "Z2", "Z3"]
REPS = int(os.environ.get("HNU_REPS", "2000"))
FOLDS = [int(f) for f in os.environ.get("HNU_FOLDS", "1,2,3,4").split(",")]
rng = np.random.default_rng(7)
METHODS = [("q", "Q", "q8", "accepted_none"), ("qq", "Q + Q 拒识", "q8", "accepted"),
           ("pv", "预演", "pv8", "accepted_none"), ("pvq", "预演 + Q 拒识", "pv8", "accepted"),
           ("pvpv", "**预演 + 预演拒识（采用）**", "pv8", "pv_accepted")]
D = {}
for F in FOLDS:
    for s in (1, 2, 3):
        f = f"{H}/expA_plan{'' if F == 1 else f'_f{F}'}_s{s}{SUF}_rows.json"
        if os.path.exists(f):
            D[(F, s)] = json.load(open(f))


def table(who, P):
    out = {}
    for (F, s), d in D.items():
        for x in d.get(f"{who}/{P}", []):
            if x["ok"] and "pv8" not in x:
                continue  # no per-rollout data for this pair
            out.setdefault(F, {}).setdefault((x["instr"], x["layout"]), {})[s] = x
    return out


def val(x, key, gate, kind):
    """kind: e2e (all pairs), acc (executed share among ok), exe (success of executed), fall (per ok command)"""
    ok = x["ok"]
    run = ok and (gate == "accepted_none" or bool(x.get(gate)))
    if kind == "e2e":
        return x[key] if run else 0.0
    if not ok:
        return None
    if kind == "acc":
        return 1.0 if run else 0.0
    if kind == "exe":
        return x[key] if run else None
    if kind == "fall":
        f = x.get(key + "_fall", np.nan)
        return (f if np.isfinite(f) else None) if run else 0.0


def est(T, key, gate, kind, pick=None):
    v = []
    for F, pairs in T.items():
        keys = list(pairs) if pick is None else [k for k in pick[F][0] if k in pairs]
        for k in keys:
            ss = pairs[k] if pick is None else ({s: pairs[k][s] for s in pick[F][1] if s in pairs[k]} or pairs[k])
            vals = [val(x, key, gate, kind) for x in ss.values()]
            vals = [a for a in vals if a is not None]
            if vals:
                v.append(np.mean(vals))
    return 100 * np.mean(v) if v else np.nan


def resample(T):
    pick = {}
    for F, pairs in T.items():
        ins, lay = sorted({k[0] for k in pairs}), sorted({k[1] for k in pairs})
        seeds = sorted({s for p in pairs.values() for s in p})
        bi, bl = rng.choice(ins, len(ins)), rng.choice(lay, len(lay))
        pick[F] = ([(i, l) for i in bi for l in bl if (i, l) in pairs], list(rng.choice(seeds, len(seeds))))
    return pick


def boot_diff(Ta, ma, Tb, mb):
    d = []
    for _ in range(REPS):
        p = resample(Ta)
        d.append(est(Ta, ma[2], ma[3], "e2e", p) - est(Tb, mb[2], mb[3], "e2e", p))
    return np.nanpercentile(d, [2.5, 97.5])


def boot_ci(T, m):
    b = [est(T, m[2], m[3], "e2e", resample(T)) for _ in range(REPS)]
    return np.nanpercentile(b, [2.5, 97.5])


have = sorted(D)
print(f"# 端到端：仿真预演选择与拒识（路由器 {RT or 'v2（冻结）'}；已有（折，种子）：{have}）\n")
print("选择 / 拒识方案见脚本说明；所有方案的执行结果统一用每对“布局 × 候选”的第 9–16 次运行评估（预演用第 1–4 次）。"
      "端到端 = 全部应执行指令 × test 布局（指令层拒识、选错动作、执行层拒识都算失败）。\n")
print("## 1. 端到端成功率 %（95% CI）\n")
print("| 学生 | 层级 | " + " | ".join(m[1] for m in METHODS) + " |")
print("|---|---|" + "---|" * len(METHODS))
for who, wn in WHO:
    for P in LV:
        T = table(who, P)
        if not T:
            continue
        cells = []
        for m in METHODS:
            e = est(T, m[2], m[3], "e2e")
            if m[0] in ("q", "pvpv"):
                lo, hi = boot_ci(T, m)
                cells.append(f"{e:.1f} [{lo:.1f}, {hi:.1f}]")
            else:
                cells.append(f"{e:.1f}")
        print(f"| {wn} | L{P[1]} | " + " | ".join(cells) + " |")
print("\n## 2. 执行层：拒识率 / 被执行指令的成功率 / 每条指令的摔倒率 %（接受且动作正确的指令 × test 布局）\n")
print("| 学生 | 层级 | " + " | ".join(f"{m[1]}：拒识 / 成功 / 摔倒" for m in METHODS) + " |")
print("|---|---|" + "---|" * len(METHODS))
for who, wn in WHO:
    for P in LV:
        T = table(who, P)
        if not T:
            continue
        cells = [f"{100 - est(T, m[2], m[3], 'acc'):.0f} / {est(T, m[2], m[3], 'exe'):.1f} / {est(T, m[2], m[3], 'fall'):.1f}" for m in METHODS]
        print(f"| {wn} | L{P[1]} | " + " | ".join(cells) + " |")
print("\n## 3. 配对比较（端到端，百分点，95% CI）\n")
print("| 比较 | 差值 | 95% CI |")
print("|---|---|---|")
Q, PVPV = METHODS[0], METHODS[4]
for who, wn in WHO:
    for P in ("Z0", "Z3"):
        T = table(who, P)
        if T:
            lo, hi = boot_diff(T, PVPV, T, Q)
            print(f"| {wn} L{P[1]}：预演 + 预演拒识 − Q | {est(T, PVPV[2], PVPV[3], 'e2e') - est(T, Q[2], Q[3], 'e2e'):+.1f} | [{lo:+.1f}, {hi:+.1f}] |")
for who, wn in WHO:
    T3, T0 = table(who, "Z3"), table(who, "Z0")
    if T3 and T0:
        lo, hi = boot_diff(T3, PVPV, T0, PVPV)
        print(f"| {wn}：L3 − L0（预演 + 预演拒识） | {est(T3, PVPV[2], PVPV[3], 'e2e') - est(T0, PVPV[2], PVPV[3], 'e2e'):+.1f} | [{lo:+.1f}, {hi:+.1f}] |")
RN = dict(WHO)[REF]
for other, on in [w for w in WHO if w[0] != REF]:
    for P in LV:
        Tn, To = table(REF, P), table(other, P)
        if Tn and To:
            lo, hi = boot_diff(Tn, PVPV, To, PVPV)
            print(f"| L{P[1]}：{RN} − {on}（预演 + 预演拒识） | {est(Tn, PVPV[2], PVPV[3], 'e2e') - est(To, PVPV[2], PVPV[3], 'e2e'):+.1f} | [{lo:+.1f}, {hi:+.1f}] |")
