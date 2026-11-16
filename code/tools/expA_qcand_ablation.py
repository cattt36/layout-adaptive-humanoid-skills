"""Experiment A, EXPLORATORY feature ablation of the candidate-level executability estimator Q_cand (user
request; whether it goes into the paper is the user's decision).  Same data, model (L2 logistic regression, C = 1,
standardised features, one model per action, trained on TRAINING layouts only) and evaluation as tools/expA_full.py;
only the input features change:
  full     : layout geometry (target, candidate, difference, distance) + z_skill mean + z_quality mean / min
  no_zq    : geometry + z_skill            (does z_quality add anything?)
  no_zs    : geometry + z_quality          (does z_skill add anything?)
  geo      : geometry only                 (no latent at all)
  zq       : z_quality only                (the latent's executability code alone)
  zq_dist  : z_quality + geometric distance
Metrics on the TEST layouts: Spearman(Q, success) over all (layout, candidate) pairs; success / fall rate of the top
fifth of Q; end-to-end success with the frozen chain (routing of latent seed 42, top-5 retrieval) and with all library
candidates.
  python expA_qcand_ablation.py [seed=1] > runs/expA_qcand_ablation_s1.md"""
import os
import sys

import numpy as np

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "expA_full.py")
seed_arg = sys.argv[1] if len(sys.argv) > 1 else "1"
sys.argv = [sys.argv[0], seed_arg]
exec(open(SRC, encoding="utf-8").read().split('print("# 实验 A（完整链路）')[0])  # definitions only

from scipy.stats import spearmanr  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

VARIANTS = [("full", "完整（几何 + z_skill + z_quality）"), ("no_zq", "去掉 z_quality（几何 + z_skill）"),
            ("no_zs", "去掉 z_skill（几何 + z_quality）"), ("geo", "只用几何"), ("zq", "只用 z_quality"),
            ("zq_dist", "z_quality + 几何距离")]


def feat_v(P, L, c, v):
    _, dl, _ = LAY[L]
    _, dc, _ = LAY[c]
    geo = np.r_[dl, dc, dl - dc, gdist(L, c)]
    z = zfeat(P, c)
    zq, zs = z[:24], z[24:]
    return {"full": np.r_[geo, zq, zs], "no_zq": np.r_[geo, zs], "no_zs": np.r_[geo, zq], "geo": geo, "zq": zq,
            "zq_dist": np.r_[gdist(L, c), zq]}[v]


def fit_v(P, succ, v):
    models = {}
    for t in TASKS:
        X, y, w = [], [], []
        for (L, c), s in succ.items():
            if L.startswith(t) and SPL.get(L) == "train":
                f = feat_v(P, L, c, v)
                X += [f, f]
                y += [1, 0]
                w += [s, 1 - s]
        sc = StandardScaler().fit(np.array(X))
        models[t] = (sc, LogisticRegression(C=1.0, max_iter=5000).fit(sc.transform(np.array(X)), np.array(y),
                                                                        sample_weight=np.array(w)))
    return lambda L, c: float(models[L.split("_")[0]][1].predict_proba(
        models[L.split("_")[0]][0].transform(feat_v(P, L, c, v)[None]))[0, 1])


for seed in SEEDS:
    print(f"# Q_cand 特征消融（探索性；第 1 折 RL 种子 {seed}，test 布局；模型与评估同 expA_full.py，只改输入特征）\n")
    res = {}
    for P in LV:
        data = load_sweep(seed, P)
        if data is None:
            continue
        succ, fall, _, _ = data
        keys = [k for k in succ if SPL.get(k[0]) == "test"]
        s = np.array([succ[k] for k in keys])
        fl = np.array([fall[k] for k in keys])
        for v, _ in VARIANTS:
            qc = fit_v(P, succ, v)
            q = np.array([qc(*k) for k in keys])
            top = np.argsort(-q)[: max(len(q) // 5, 1)]
            e5 = np.mean(score(e2e(P, 42, data, qc, 0.0), "qcand"))
            ea = np.mean(score(e2e(P, 42, data, qc, 0.0, topk=None), "qcand"))
            res[(v, P)] = (spearmanr(q, s)[0], 100 * s[top].mean(), 100 * fl[top].mean(), 100 * e5, 100 * ea)
    cols = [P for P in LV if ("full", P) in res]
    for idx, (title, fmt) in enumerate([("Spearman（Q 与实际成功率，test 布局 × 全部候选）", "{:.2f}"),
                                        ("Q 最高的 1/5 配对：实际成功率 %", "{:.1f}"),
                                        ("Q 最高的 1/5 配对：摔倒率 %", "{:.1f}"),
                                        ("端到端成功率 %（前 5 个候选，冻结的链路）", "{:.1f}"),
                                        ("端到端成功率 %（全部同动作库片段作候选）", "{:.1f}")]):
        print(f"## {title}\n")
        print("| 输入特征 | " + " | ".join(f"L{P[1]}" for P in cols) + " | 平均 |")
        print("|---|" + "---|" * (len(cols) + 1))
        for v, name in VARIANTS:
            vals = [res[(v, P)][idx] for P in cols]
            print(f"| {name} | " + " | ".join(fmt.format(x) for x in vals) + f" | {fmt.format(np.mean(vals))} |")
        print()
    print("参照（同一数据）：几何最近规则与随机选择的端到端成功率见 runs/expA_full_s1.md。")
