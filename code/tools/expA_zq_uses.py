"""EXPLORATORY uses of z_quality (user request; the user decides whether anything goes into the paper).
Data: the experiment-A candidate sweep (tools/cand_sweep.sh; every layout x every same-action library clip, 16
rollouts, nominal) and the fold-1 latents (latent seed 42).  Metrics fixed BEFORE running:

A  Safety-oriented abstention in the frozen chain (in-scope, correctly routed instructions x TEST layouts, top-5
   retrieved candidates).  Selection is fixed: the candidate with the highest predicted success.  Rejection scores
   compared at equal coverage:
     S_geo    : predicted success, geometry-only success model (also the selector)
     S_full   : predicted success, full model (geometry + z_skill + z_quality; also the selector)
     S_safe   : predicted FALL risk of the selected candidate, fall model on geometry + z_quality (selector = geo)
     S_safe0  : predicted FALL risk, fall model on geometry only (selector = geo)     <- control for S_safe
   Reported: fall rate among executed attempts and success among accepted at 100/90/80/70/60/50 % coverage, and the
   mean fall rate over coverage 50-100 %.  All models: L2 logistic regression (C = 1), standardised, per action,
   trained on TRAINING layouts only (fall model: label = share of falling rollouts).

B  Skill transferability: for every library clip, its mean success over the TEST layouts of its action.  Predictors
   (Spearman per action, then averaged over the three actions):
     zq       : ridge (alpha 1) from z_quality (mean, min), leave-one-clip-out, fitted to the clips' mean success on
                the TRAINING layouts (never test)
     zs       : the same from z_skill (mean)
     typical  : - mean geometric distance of the clip's layout to the training layouts (no latent)
     ceiling  : the clip's own mean success on the training layouts (not available for a new skill; reference)
  python expA_zq_uses.py [seed=1] > runs/expA_zq_uses_s1.md"""
import os
import sys

import numpy as np

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "expA_full.py")
seed_arg = sys.argv[1] if len(sys.argv) > 1 else "1"
sys.argv = [sys.argv[0], seed_arg]
exec(open(SRC, encoding="utf-8").read().split('print("# 实验 A（完整链路）')[0])  # definitions only

from scipy.stats import spearmanr  # noqa: E402
from sklearn.linear_model import LogisticRegression, Ridge  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402


def feat_v(P, L, c, v):
    _, dl, _ = LAY[L]
    _, dc, _ = LAY[c]
    geo = np.r_[dl, dc, dl - dc, gdist(L, c)]
    z = zfeat(P, c)
    return {"geo": geo, "full": np.r_[geo, z[:24], z[24:]], "geo_zq": np.r_[geo, z[:24]]}[v]


def fit(P, label, v):
    models = {}
    for t in TASKS:
        X, y, w = [], [], []
        for (L, c), s in label.items():
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


COV = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5)
SCORES = [("S_geo", "成功概率（只用几何）"), ("S_full", "成功概率（完整：几何 + z_skill + z_quality）"),
          ("S_safe0", "摔倒风险（只用几何）"), ("S_safe", "摔倒风险（几何 + z_quality）")]
for seed in SEEDS:
    print(f"# z_quality 的两种用法（探索性；第 1 折 RL 种子 {seed}；评估标准在运行前写定，见脚本说明）\n")
    A, B = {}, {}
    for P in LV:
        data = load_sweep(seed, P)
        if data is None:
            continue
        succ, fall, _, _ = data
        q_geo, q_full = fit(P, succ, "geo"), fit(P, succ, "full")
        f_geo, f_gzq = fit(P, fall, "geo"), fit(P, fall, "geo_zq")
        # ---------------- A
        pairs = []
        for r in RANK[f"42/{P}"]:
            if r["rejected"] or r["routed"] != r["expected"]:
                continue
            for L in [n for n in LAY if n.startswith(r["expected"]) and SPL[n] == "test"]:
                cands = r["ranking"][:TOPK]
                cg = max(cands, key=lambda c: q_geo(L, c))
                cf = max(cands, key=lambda c: q_full(L, c))
                pairs.append(dict(
                    S_geo=(q_geo(L, cg), succ[(L, cg)], fall[(L, cg)]),
                    S_full=(q_full(L, cf), succ[(L, cf)], fall[(L, cf)]),
                    S_safe0=(1 - f_geo(L, cg), succ[(L, cg)], fall[(L, cg)]),
                    S_safe=(1 - f_gzq(L, cg), succ[(L, cg)], fall[(L, cg)])))
        for sname, _ in SCORES:
            arr = np.array([p[sname] for p in pairs])  # acceptance score, success, fall
            o = np.argsort(-arr[:, 0], kind="stable")
            row = {}
            for cv in COV:
                k = max(int(round(cv * len(o))), 1)
                acc = o[:k]
                row[cv] = (100 * arr[acc, 2].mean(), 100 * arr[acc, 1].mean())
            fine = [100 * arr[o[:max(int(round(cv * len(o))), 1)], 2].mean() for cv in np.linspace(0.5, 1.0, 11)]
            A[(P, sname)] = (row, float(np.mean(fine)), len(pairs))
        # ---------------- B
        res = {"zq": [], "zs": [], "typical": [], "ceiling": []}
        for t in TASKS:
            lib = [c for c in LIB[t] if c in LAY]
            tr_l = [n for n in LAY if n.startswith(t) and SPL[n] == "train"]
            te_l = [n for n in LAY if n.startswith(t) and SPL[n] == "test"]
            ttr = {c: np.mean([succ[(L, c)] for L in tr_l if L != c and (L, c) in succ]) for c in lib}
            tte = {c: np.mean([succ[(L, c)] for L in te_l if (L, c) in succ]) for c in lib}
            y_te = [tte[c] for c in lib]
            for key, sl in (("zq", slice(0, 24)), ("zs", slice(24, None))):
                pred = []
                for c in lib:
                    others = [o_ for o_ in lib if o_ != c]
                    X = np.array([zfeat(P, o_)[sl] for o_ in others])
                    sc = StandardScaler().fit(X)
                    m = Ridge(alpha=1.0).fit(sc.transform(X), [ttr[o_] for o_ in others])
                    pred.append(float(m.predict(sc.transform(zfeat(P, c)[sl][None]))[0]))
                res[key].append(spearmanr(pred, y_te)[0])
            res["typical"].append(spearmanr([-np.mean([gdist(c, L) for L in tr_l if L != c]) for c in lib], y_te)[0])
            res["ceiling"].append(spearmanr([ttr[c] for c in lib], y_te)[0])
        B[P] = {k: (float(np.nanmean(v)), [round(float(x), 2) for x in v]) for k, v in res.items()}
    cols = [P for P in LV if P in B]
    print("## A. 面向安全的拒识（选择规则固定：成功概率最高的候选；比较用哪个分数决定拒识）\n")
    print(f"配对数（正确路由的指令 × test 布局）：{A[(cols[0], 'S_geo')][2]}\n")
    for P in cols:
        print(f"**L{P[1]}**\n")
        print("| 拒识分数 | " + " | ".join(f"覆盖 {int(c * 100)}%：摔倒 / 成功" for c in COV) + " | 摔倒率平均（覆盖 50–100%） |")
        print("|---|" + "---|" * (len(COV) + 1))
        for sname, label in SCORES:
            row, auc, _ = A[(P, sname)]
            print(f"| {label} | " + " | ".join(f"{row[c][0]:.1f} / {row[c][1]:.1f}" for c in COV) + f" | {auc:.1f} |")
        print()
    print("## B. 技能通用性：预测库中每个动作在 test 布局上的平均成功率（Spearman，三个动作取平均；括号内为独木桥 / 三步桩 / 五步桩）\n")
    print("| 预测方式 | " + " | ".join(f"L{P[1]}" for P in cols) + " |")
    print("|---|" + "---|" * len(cols))
    for key, label in (("zq", "z_quality（留一交叉验证，拟合训练布局上的成功率）"), ("zs", "z_skill（同上）"),
                       ("typical", "布局典型程度（不用 latent）"), ("ceiling", "参考：该动作在训练布局上的实际成功率")):
        print(f"| {label} | " + " | ".join(f"{B[P][key][0]:.2f}（{' / '.join(str(x) for x in B[P][key][1])}）" for P in cols) + " |")
    print("\n每个动作的库片段数：独木桥 18、三步桩 14、五步桩 19；Spearman 的样本量小，波动大。")
