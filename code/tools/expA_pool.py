"""Full chain (new execution layer) over all students: folds 1-4 x L0-L3 x RL seeds 1-3, new (xm) and old (n)
(user request; rows written by tools/expA_plan.py, runs/expA_plan[_f<k>]_s<seed>_rows.json).
Estimator as pre-registered for the closed loop: unit = (command x test layout, RL seed); point estimate = mean over the
(command, layout) pairs of the per-pair mean over seeds; 95 % CI = bootstrap stratified by fold (commands, layouts and
seeds resampled within each fold).  Layers:
  instruction : frozen router per fold and level (latent seed 42) on the sealed set -- independent of the student
  execution   : pairs whose command was accepted and routed correctly
  end to end  : all in-scope pairs, refusal / wrong action = failure
  python expA_pool.py > runs/expA_pool.md"""
import json
import os

import numpy as np

H = "/root/gpufree-data/hnu_v2/runs"
WHO = [("xm", "新 student"), ("n", "旧 student")]
LV = ["Z0", "Z1", "Z2", "Z3"]
TASKS = [("CrouchedTraverse", "独木桥"), ("ThreeStepPoleRun", "三步桩"), ("FiveStepPoleRun", "五步桩")]
REPS = 2000
rng = np.random.default_rng(1)
cases = {c["id"]: c for c in json.load(open(f"{H}/../code/data_lang/router_test_v2_sealed.json", encoding="utf-8"))["cases"]}
# HNU_RTAG=v3 (router v3) / HNU_QZ=1 (Q_cand + predicted-z feature) read the matching rows of expA_plan.py
RT = os.environ.get("HNU_RTAG", "")
EXTRA = [w for w in os.environ.get("HNU_WHO_EXTRA", "").split(",") if w]  # external baseline (tools/rwt_sweep.sh)
WHO += [(w, {"rwtt": "基线：检索 + 时间对齐 + 跟踪", "rwts": "基线：检索 + 时空对齐 + 跟踪",
             "zr": "公平基线 ZR（显式参考学生）"}.get(w, w)) for w in EXTRA]
XS = "_rwt" if EXTRA in (["rwtt", "rwts"], ["rwtt"], ["rwts"]) else ("_" + "_".join(EXTRA) if EXTRA else "")
SUF = (f"_{RT}" if RT else "") + ("_qz" if os.environ.get("HNU_QZ", "") == "1" else "") + XS \
    + (f"_evdr{os.environ['HNU_EVAL_DR']}" if os.environ.get("HNU_EVAL_DR", "0") != "0" else "")
D = {}
for F in (1, 2, 3, 4):
    for s in (1, 2, 3):
        f = f"{H}/expA_plan{'' if F == 1 else f'_f{F}'}_s{s}{SUF}_rows.json"
        if os.path.exists(f):
            D[(F, s)] = json.load(open(f))


def table(who, P, task=None):
    """fold -> {(instr, layout): {seed: row}}"""
    out = {}
    for (F, s), d in D.items():
        for x in d.get(f"{who}/{P}", []):
            if task is None or x["task"] == task:
                out.setdefault(F, {}).setdefault((x["instr"], x["layout"]), {})[s] = x
    return out


def value(x, layer, m):
    if layer == "e2e":
        return x[m] if x["ok"] else 0.0
    return x[m] if x["ok"] else None


def estimate(T, layer, m, pick=None):
    v = []
    for F, pairs in T.items():
        if pick is not None and F not in pick:
            continue
        keys = list(pairs) if pick is None else [k for k in pick[F][0] if k in pairs]
        for key in keys:
            seeds = pairs[key] if pick is None else {s: pairs[key][s] for s in pick[F][1] if s in pairs[key]}
            if not seeds and EXTRA:  # baseline tracker has one "seed": keep it whatever seeds the resample drew
                seeds = pairs[key]
            vals = [value(x, layer, m) for x in seeds.values()]
            vals = [a for a in vals if a is not None]
            if vals:
                v.append(np.mean(vals))
    return np.mean(v) if v else np.nan


def resample(T):
    pick = {}
    for F, pairs in T.items():
        instr = sorted({k[0] for k in pairs})
        lays = sorted({k[1] for k in pairs})
        seeds = sorted({s for p in pairs.values() for s in p})
        bi, bl, bs = rng.choice(instr, len(instr)), rng.choice(lays, len(lays)), list(rng.choice(seeds, len(seeds)))
        pick[F] = ([(i, l) for i in bi for l in bl if (i, l) in pairs], bs)
    return pick


def boot(Ts, layer, m):
    """Ts: list of tables sharing the resampling (paired comparisons); returns [REPS, len(Ts)]."""
    out = []
    for _ in range(REPS):
        pick = resample(Ts[0])
        out.append([estimate(T, layer, m, pick) for T in Ts])
    return np.array(out)


ci = lambda b: f"[{100 * np.nanpercentile(b, 2.5):.1f}, {100 * np.nanpercentile(b, 97.5):.1f}]"  # noqa: E731
have = sorted(D)
print("# 完整链路（新执行层）：全部 student 汇总\n")
print(f"已有的（折，种子）：{have}。单位为（指令 × test 布局，RL 种子），先对种子取平均再对配对取平均；95% CI 为按折分层 bootstrap"
      f"（{REPS} 次）。三步桩/五步桩的踩桩时机和独木桥的进桥时间都由几何规划器给出，不使用真人时序。\n")
print("## 1. 指令层（冻结路由器，每折、每层级；与 student 和 RL 种子无关）\n")
print("| 层级 | 应执行指令被接受（覆盖率） | 接受后选对动作 | 越界 / 歧义指令被拒绝 |")
print("|---|---|---|---|")
for P in LV:
    acc = cor = tot = ref = nref = 0
    for F in sorted({F for F, _ in D}):
        rk = json.load(open(f"{H}/expA_rank{'' if F == 1 else f'_f{F}'}{'_' + RT if RT else ''}.json")).get(f"42/{P}", [])
        root = f"{H}/latent_v5t" if F == 1 else f"{H}/latent_cv_f{F}"
        route = json.load(open(f"{root}/seed42/{P}/language_routing_test{RT or 'v2'}.json", encoding="utf-8"))
        tot += len(rk)
        acc += sum(not r["rejected"] for r in rk)
        cor += sum((not r["rejected"]) and r["routed"] == r["expected"] for r in rk)
        oos = [x for x in route if cases[x["id"]]["should_abstain"]]
        ref += sum(x["rejected"] for x in oos)
        nref += len(oos)
    if tot:
        print(f"| L{P[1]} | {100 * acc / tot:.1f}% | {100 * cor / max(acc, 1):.1f}% | {100 * ref / max(nref, 1):.1f}% |")
print("\n（按折汇总计数；每折的封存测试集相同，路由器为该折的编码器。）\n")
for layer, title in (("exec", "2. 执行层（被接受且动作正确的指令 × test 布局）"), ("e2e", "3. 端到端（全部应执行指令 × test 布局）")):
    print(f"## {title}，Q_cand 选择，成功率 %\n")
    print("| student | 层级 | 成功率 [95% CI] | " + " | ".join(t for _, t in TASKS) + " | 前 5 几何最近 | 检索第 1 | 前 5 实际最好（上限） |")
    print("|---|---|---|---|---|---|---|---|---|")
    for who, wn in WHO:
        for P in LV:
            T = table(who, P)
            if not T:
                continue
            tk = [estimate(table(who, P, t), layer, "qcand") for t, _ in TASKS]
            print(f"| {wn} | L{P[1]} | {100 * estimate(T, layer, 'qcand'):.1f} {ci(boot([T], layer, 'qcand')[:, 0])} | "
                  + " | ".join(f"{100 * v:.1f}" for v in tk)
                  + f" | {100 * estimate(T, layer, 'geo'):.1f} | {100 * estimate(T, layer, 'rank1'):.1f} | {100 * estimate(T, layer, 'oracle'):.1f} |")
    print()
print("## 4. 配对比较（端到端，Q_cand）\n")
print("| 比较 | 差值（百分点） | 95% CI | 非劣效（下限 > −5） | 优越（下限 > 0） |")
print("|---|---|---|---|---|")
for who, wn in WHO:
    T0, T3 = table(who, "Z0"), table(who, "Z3")
    if T0 and T3:
        b = boot([T3, T0], "e2e", "qcand")
        d = b[:, 0] - b[:, 1]
        print(f"| {wn}：L3 − L0 | {100 * (estimate(T3, 'e2e', 'qcand') - estimate(T0, 'e2e', 'qcand')):+.1f} | {ci(d)} | "
              f"{'是' if np.nanpercentile(d, 2.5) > -0.05 else '否'} | {'是' if np.nanpercentile(d, 2.5) > 0 else '否'} |")
for w, wn in [x for x in WHO if x[0] in EXTRA]:
    for layer in ("exec", "e2e"):
        for P in LV:
            Tn, Tb = table("xm", P), table(w, P)
            if Tn and Tb:
                b = boot([Tn, Tb], layer, "qcand")
                d = b[:, 0] - b[:, 1]
                print(f"| L{P[1]}：新 student − {wn}（{'执行层' if layer == 'exec' else '端到端'}） | "
                      f"{100 * (estimate(Tn, layer, 'qcand') - estimate(Tb, layer, 'qcand')):+.1f} | {ci(d)} | "
                      f"{'是' if np.nanpercentile(d, 2.5) > -0.05 else '否'} | {'是' if np.nanpercentile(d, 2.5) > 0 else '否'} |")
for P in LV:
    Tn, To = table("xm", P), table("n", P)
    if Tn and To:
        b = boot([Tn, To], "e2e", "qcand")
        d = b[:, 0] - b[:, 1]
        print(f"| L{P[1]}：新 − 旧 | {100 * (estimate(Tn, 'e2e', 'qcand') - estimate(To, 'e2e', 'qcand')):+.1f} | {ci(d)} | "
              f"{'是' if np.nanpercentile(d, 2.5) > -0.05 else '否'} | {'是' if np.nanpercentile(d, 2.5) > 0 else '否'} |")
