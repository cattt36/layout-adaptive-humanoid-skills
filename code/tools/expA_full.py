"""Experiment A, full chain (runs/preregistration_expA_impl.md, frozen , sha a50635a2...):
instruction -> z_sem -> router v2 (abstain) -> retrieve z_skill (top-5 library clips) -> Q_exec -> student on the
held-out layouts.  Inputs: candidate sweep (tools/cand_sweep.sh), rankings (tools/expA_rank.py), own-z reference
(results_v6g10n_C0_s<seed>_dr0).  Prints a markdown report.
  python expA_full.py [seeds=1] > runs/expA_full_s1.md"""
import json
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

H = "/root/gpufree-data/hnu_v2"
SEEDS = [int(s) for s in (sys.argv[1] if len(sys.argv) > 1 else "1").split(",")]
LV = ["Z0", "Z1", "Z2", "Z3"]  # L4 dropped from the study (user decision ; deviation 9)
TASKS = ["CrouchedTraverse", "ThreeStepPoleRun", "FiveStepPoleRun"]
TN = {"CrouchedTraverse": "独木桥", "ThreeStepPoleRun": "三步桩", "FiveStepPoleRun": "五步桩"}
GEO = {"ThreeStepPoleRun": f"{H}/data/ref_stakes_s93_h10/ThreeStepPoleRun", "FiveStepPoleRun": f"{H}/data/ref_stakes_s83_h10/FiveStepPoleRun",
       "CrouchedTraverse": f"{H}/data/ref_bridge_video_lift/CrouchedTraverse"}
TOPK = 5
cfg = json.load(open(f"{H}/data/z_cand_f1/cands.json"))
LIB, K = cfg["library"], cfg["K"]
split = json.load(open(f"{H}/data/clip_split_trial/split_v2.json"))
SPL = {n: k for k in ("train", "val", "test") for n in split[k]}
RANK = json.load(open(f"{H}/runs/expA_rank.json"))
rng = np.random.default_rng(0)


# ------------------------------------------------------------------ layouts (start frame; as tools/make_z_library.py)
def yaw_of(q):
    w, x, y, z = q
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def layout(name):
    t = name.split("_")[0]
    ref = np.load(os.path.join(GEO[t], f"{name}_motion.npz"))
    p0, yaw0 = ref["body_pos_w"][0, 0], yaw_of(ref["body_quat_w"][0, 0])
    c, s = np.cos(-yaw0), np.sin(-yaw0)
    rot = lambda xy: [c * (xy[0] - p0[0]) - s * (xy[1] - p0[1]), s * (xy[0] - p0[0]) + c * (xy[1] - p0[1])]  # noqa: E731
    if t == "CrouchedTraverse":
        parts = json.load(open(os.path.join(GEO[t], f"{name}_bridge.json")))["parts"]
        pts = [rot(parts["plank"]["center"][:2])] + [rot(parts[p]["center"][:2]) for p in sorted(parts) if p.startswith("post")]
        under = parts["plank"]["center"][2] - parts["plank"]["half"][2]
        yaw = float(np.angle(np.exp(1j * (parts["plank"]["yaw"] - yaw0))))
        return np.array(pts), np.r_[np.ravel(pts), under, yaw], dict(under=under, yaw=yaw)
    j = json.load(open(os.path.join(GEO[t], f"{name}_stakes.json")))
    pts = np.array([rot(p[:2]) for p in j["stake_xyz_50hz"]])
    return pts, np.ravel(pts), {}


LAY = {}
for t in TASKS:
    for n in sorted(SPL):
        if n.startswith(t) and os.path.exists(os.path.join(GEO[t], f"{n}_motion.npz")):
            LAY[n] = layout(n)


def gdist(a, b):
    pa, _, ea = LAY[a]
    pb, _, eb = LAY[b]
    if len(pa) != len(pb):
        return 1e9
    d = float(np.linalg.norm(pa - pb, axis=1).mean())
    if ea:
        d += 3.0 * abs(ea["under"] - eb["under"]) + 0.5 * abs(np.angle(np.exp(1j * (ea["yaw"] - eb["yaw"]))))
    return d


# ------------------------------------------------------------------ sweep results
def load_sweep(seed, P):
    succ, fall, q0 = {}, {}, {}
    for k in range(K):
        d = f"{H}/runs/results_v6g10n_cand_k{k:02d}_s{seed}_dr0"
        if not os.path.exists(f"{d}/eval_student_{P}.json"):
            return None
        rows = {r["clip"]: r for r in json.load(open(f"{d}/eval_student_{P}.json"))["clips"]}
        z = np.load(f"{d}/qexec_{P}.npz")
        names = [str(s) for s in z["names"]]
        cl = z["clip"]
        q_start = z["q"][0].mean(1)
        for ci, n in enumerate(names):
            lib = LIB[n.split("_")[0]]
            if n not in LAY or k >= len(lib) or lib[k] == n:
                continue
            m = cl == ci
            succ[(n, lib[k])] = rows[n]["success_v2"]
            fall[(n, lib[k])] = float(z["fell"][m].mean())
            q0[(n, lib[k])] = float(q_start[m].mean())
    own_f = f"{H}/runs/results_v6g10n_C0_s{seed}_dr0/eval_student_{P}.json"
    own = {r["clip"]: r["success_v2"] for r in json.load(open(own_f))["clips"]} if os.path.exists(own_f) else {}
    return succ, fall, q0, own


def zfeat(P, clip, cache={}):
    if (P, clip) not in cache:
        g = np.load(f"{H}/runs/latent_v5t/seed42/{P}/train_latents.npz", allow_pickle=True)
        tr = np.array([str(t) for t in g["trial_id"]])
        for c in set(tr):
            m = tr == c
            cache[(P, c)] = np.r_[g["z_quality"][m].mean(0), g["z_quality"][m].min(0), g["z_skill"][m].mean(0)]
    return cache[(P, clip)]


def feat(P, L, c):
    _, dl, _ = LAY[L]
    _, dc, _ = LAY[c]
    return np.r_[dl, dc, dl - dc, gdist(L, c), zfeat(P, c)]


def fit_qcand(P, succ):
    models = {}
    for t in TASKS:
        X, y, w = [], [], []
        for (L, c), s in succ.items():
            if L.startswith(t) and SPL.get(L) == "train":
                f = feat(P, L, c)
                X += [f, f]
                y += [1, 0]
                w += [s, 1 - s]
        sc = StandardScaler().fit(np.array(X))
        m = LogisticRegression(C=1.0, max_iter=5000).fit(sc.transform(np.array(X)), np.array(y), sample_weight=np.array(w))
        models[t] = (sc, m)
    return lambda L, c: float(models[L.split("_")[0]][1].predict_proba(models[L.split("_")[0]][0].transform(feat(P, L, c)[None]))[0, 1])


# ------------------------------------------------------------------ E2E
METHODS = [("own", "片段自身的 z（参考，即上午的实现）"), ("oracle", "前 5 候选中实际最好（上限）"),
           ("qcand", "**Q_cand（候选级可执行性，主方法）**"), ("q0", "Q_exec⁰（起始状态）"), ("geo", "几何最近"),
           ("rank1", "检索排名第 1（链路去掉 Q_exec）"), ("random", "随机（期望值）")]


def e2e(P, lseed, data, qc, tau, topk=TOPK):
    succ, fall, q0, own = data
    rows = []
    for r in RANK[f"{lseed}/{P}"]:
        ok = (not r["rejected"]) and r["routed"] == r["expected"]
        layouts = [n for n in LAY if n.startswith(r["expected"]) and SPL[n] == "test"]
        for L in layouts:
            v = {}
            if ok:
                cands = r["ranking"][:topk] if topk else r["ranking"]
                s = {c: succ[(L, c)] for c in cands}
                qs = {c: qc(L, c) for c in cands}
                pick = dict(qcand=max(qs, key=qs.get), q0=max(cands, key=lambda c: q0[(L, c)]),
                            geo=min(cands, key=lambda c: gdist(L, c)), rank1=cands[0], oracle=max(s, key=s.get))
                v = {m: s[c] for m, c in pick.items()}
                v["random"] = float(np.mean(list(s.values())))
                v["own"] = own.get(L, np.nan)
                v["fall_qcand"] = fall[(L, pick["qcand"])]
                v["accepted"] = max(qs.values()) >= tau
            rows.append(dict(task=r["expected"], instr=r["id"], layout=L, ok=ok, **v))
    return rows


def val_tau(P, lseed, qc):
    vals = []
    for r in RANK[f"{lseed}/{P}"]:
        if r["rejected"] or r["routed"] != r["expected"]:
            continue
        for L in [n for n in LAY if n.startswith(r["expected"]) and SPL[n] == "val"]:
            vals.append(max(qc(L, c) for c in r["ranking"][:TOPK]))
    return float(np.quantile(vals, 0.10)), len(vals)


def score(rows, m, reject=False):
    v = []
    for x in rows:
        if not x["ok"]:
            v.append(0.0)
        elif reject and not x["accepted"]:
            v.append(0.0)
        else:
            v.append(x[m])
    return np.array(v, float)


def boot(rows_by_level, m, reps=10000, reject=False):
    """Bootstrap over instructions and test layouts (pairs keep their instruction / layout identity)."""
    lv = list(rows_by_level)
    instr = sorted({x["instr"] for x in rows_by_level[lv[0]]})
    lays = sorted({x["layout"] for x in rows_by_level[lv[0]]})
    idx = {P: {(x["instr"], x["layout"]): i for i, x in enumerate(rows_by_level[P])} for P in lv}
    val = {P: score(rows_by_level[P], m, reject) for P in lv}
    out = {P: [] for P in lv}
    for _ in range(reps):
        bi = rng.choice(instr, len(instr))
        bl = set(rng.choice(lays, len(lays)))
        wl = {l: c for l, c in zip(*np.unique(rng.choice(lays, len(lays)), return_counts=True))}
        for P in lv:
            acc, wsum = 0.0, 0.0
            for i in bi:
                for l, c in wl.items():
                    j = idx[P].get((i, l))
                    if j is not None:
                        acc += c * val[P][j]
                        wsum += c
            out[P].append(acc / max(wsum, 1e-9))
    return {P: np.array(v) for P, v in out.items()}


print("# 实验 A（完整链路）：指令 → z_sem → 路由 → 检索 z_skill → Q_exec → 学生执行（第 1 折，test 布局）\n")
print("实施细则：runs/preregistration_expA_impl.md（定稿，sha a50635a2…），这里严格按细则计算。")
print(f"RL 种子：{SEEDS}；latent 种子 42（主分析）；候选 = 检索排名前 {TOPK} 的库片段；每个布局 × 候选执行 16 次，理想条件，成功标准 v2。\n")
for seed in SEEDS:
    R, TAU, DATA, QC = {}, {}, {}, {}
    for P in LV:
        d = load_sweep(seed, P)
        if d is None:
            print(f"- 种子 {seed} {P}：候选扫描不完整，跳过")
            continue
        DATA[P] = d
        QC[P] = fit_qcand(P, d[0])
        TAU[P], nv = val_tau(P, 42, QC[P])
        R[P] = e2e(P, 42, d, QC[P], TAU[P])
    if not R:
        continue
    print(f"## RL 种子 {seed}\n")
    print("### 端到端成功率 %（应执行指令 × test 布局；路由错误或拒识算失败）\n")
    print("| 选择方法 | " + " | ".join(f"L{P[1]}" for P in R) + " |")
    print("|---|" + "---|" * len(R))
    for m, name in METHODS:
        print(f"| {name} | " + " | ".join(f"{100 * np.nanmean(score(R[P], m)):.1f}" for P in R) + " |")
    print()
    B = boot(R, "qcand")
    print("**主指标（Q_cand）的 95% CI 与 L3 − L0**：" + "；".join(
        f"L{P[1]} {100 * np.mean(score(R[P], 'qcand')):.1f} [{100 * np.percentile(B[P], 2.5):.1f}, {100 * np.percentile(B[P], 97.5):.1f}]" for P in R))
    if "Z0" in R and "Z3" in R:
        dlt = B["Z3"] - B["Z0"]
        print(f"\nL3 − L0 = {100 * (np.mean(score(R['Z3'], 'qcand')) - np.mean(score(R['Z0'], 'qcand'))):+.1f} 个百分点，"
              f"95% CI [{100 * np.percentile(dlt, 2.5):+.1f}, {100 * np.percentile(dlt, 97.5):+.1f}]；"
              f"非劣效（CI 下限 > −5）：{'是' if np.percentile(dlt, 2.5) > -0.05 else '否'}；"
              f"优越（CI 下限 > 0）：{'是' if np.percentile(dlt, 2.5) > 0 else '否'}")
    print("\n### Q_cand 相对“检索排名第 1”的配对提升（同一指令 × 布局）\n")
    print("| 层级 | 提升（百分点） | 95% CI |")
    print("|---|---|---|")
    for P in R:
        Bq, Br = boot({P: R[P]}, "qcand", 4000), None
        b1 = boot({P: R[P]}, "rank1", 4000)
        d = (Bq[P] - b1[P]) * 100
        print(f"| L{P[1]} | {100 * (np.mean(score(R[P], 'qcand')) - np.mean(score(R[P], 'rank1'))):+.1f} | "
              f"[{np.percentile(d, 2.5):+.1f}, {np.percentile(d, 97.5):+.1f}] |")
    print("\n### 加上 Q_cand 拒识（τ 在 val 布局上取覆盖率 ≥ 90% 的最大值）\n")
    print("| 层级 | τ | 端到端成功率 % | 覆盖率 % | 被接受时成功率 % | 执行中摔倒率 %（不拒识 → 拒识） |")
    print("|---|---|---|---|---|---|")
    for P in R:
        okr = [x for x in R[P] if x["ok"]]
        acc = [x for x in okr if x["accepted"]]
        f_all = np.mean([x["fall_qcand"] for x in okr]) if okr else np.nan
        f_acc = np.sum([x["fall_qcand"] for x in acc]) / max(len(okr), 1)
        print(f"| L{P[1]} | {TAU[P]:.2f} | {100 * np.mean(score(R[P], 'qcand', True)):.1f} | {100 * len(acc) / max(len(okr), 1):.1f} | "
              f"{100 * np.mean([x['qcand'] for x in acc]):.1f} | {100 * f_all:.1f} → {100 * f_acc:.1f} |")
    print("\n### 分动作（Q_cand，%）\n")
    print("| 层级 | " + " | ".join(TN[t] for t in TASKS) + " | 路由正确率 % |")
    print("|---|---|---|---|---|")
    for P in R:
        cells = [f"{100 * np.mean(score([x for x in R[P] if x['task'] == t], 'qcand')):.1f}" for t in TASKS]
        rt = np.mean([x["ok"] for x in R[P]])
        print(f"| L{P[1]} | " + " | ".join(cells) + f" | {100 * rt:.1f} |")
    print("\n### 敏感性：候选 = 全部同动作库片段（%）\n")
    print("| 选择方法 | " + " | ".join(f"L{P[1]}" for P in R) + " |")
    print("|---|" + "---|" * len(R))
    RA = {P: e2e(P, 42, DATA[P], QC[P], TAU[P], topk=None) for P in R}
    for m, name in METHODS:
        if m in ("own",):
            continue
        print(f"| {name} | " + " | ".join(f"{100 * np.nanmean(score(RA[P], m)):.1f}" for P in R) + " |")
    print("\n### 敏感性：latent 种子 43 / 44 的路由与检索（Q_cand，%）\n")
    print("| latent 种子 | " + " | ".join(f"L{P[1]}" for P in R) + " |")
    print("|---|" + "---|" * len(R))
    for ls in (42, 43, 44):
        print(f"| {ls} | " + " | ".join(f"{100 * np.mean(score(e2e(P, ls, DATA[P], QC[P], TAU[P]), 'qcand')):.1f}" for P in R) + " |")
    print("\n### Q_cand 在 test 布局上的预测质量（全部候选；与实际成功率的秩相关）\n")
    print("| 层级 | Spearman（Q_cand） | Spearman（Q_exec⁰） | Spearman（−几何距离） |")
    print("|---|---|---|---|")
    from scipy.stats import spearmanr
    for P in R:
        succ, _, q0, _ = DATA[P]
        keys = [k for k in succ if SPL.get(k[0]) == "test"]
        s = [succ[k] for k in keys]
        print(f"| L{P[1]} | {spearmanr([QC[P](*k) for k in keys], s)[0]:.2f} | {spearmanr([q0[k] for k in keys], s)[0]:.2f} | "
              f"{spearmanr([-gdist(*k) for k in keys], s)[0]:.2f} |")
    # descriptive (added before the sweep finished; not a primary metric): Q_cand bins -> actual outcome
    print("\n### Q_cand 越高，执行越成功、越安全吗？（test 布局 × 全部候选，按 Q_cand 五等分，描述性）\n")
    print("| 层级 | 档（Q_cand 从低到高） | Q_cand 范围 | 实际成功率 % | 摔倒率 % | 配对数 |")
    print("|---|---|---|---|---|---|")
    for P in R:
        succ, fall, _, _ = DATA[P]
        keys = [k for k in succ if SPL.get(k[0]) == "test"]
        qv = np.array([QC[P](*k) for k in keys])
        sv = np.array([succ[k] for k in keys])
        fv = np.array([fall[k] for k in keys])
        o = np.argsort(qv, kind="stable")
        for bi, part in enumerate(np.array_split(o, 5)):
            print(f"| L{P[1]} | {bi + 1} | {qv[part].min():.2f}–{qv[part].max():.2f} | {100 * sv[part].mean():.1f} | "
                  f"{100 * fv[part].mean():.1f} | {len(part)} |")
    print()
