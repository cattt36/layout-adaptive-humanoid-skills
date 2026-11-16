"""Execution layer, idea "executability prior" (exploratory, offline).  Only L3 has a supervised z_quality
(executability of the clip from simulated expert rollouts; AUROC 0.93 vs 0.70 without supervision).  Question: does
the candidate's z_quality predict how well that library clip TRANSFERS to other layouts, better at L3 than at L0?
  target   : t_c = mean success of library candidate c over the TRAINING layouts of the E2E candidate sweep
  predictor: ridge on the clip mean / min of z_quality (fitted on the library, leave-one-clip-out R^2 and Spearman)
  selection: on the test layouts, among the top-5 retrieved candidates (frozen router v2 ranking), pick the highest
             predicted t_c ("zq prior"); reference: pick the highest measured t_c ("empirical prior", level-agnostic,
             needs the training-layout sweep), Q_cand ~74-78 %, oracle ~90 %
  python zq_prior.py [who=xm] [seed=1] > runs/zq_prior_<who>.md"""
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.preprocessing import StandardScaler

H = "/root/gpufree-data/hnu_v2"
WHO = sys.argv[1] if len(sys.argv) > 1 else "xm"
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 1
LV = ["Z0", "Z1", "Z2", "Z3"]
TASKS = ["CrouchedTraverse", "ThreeStepPoleRun", "FiveStepPoleRun"]
AL = np.logspace(-2, 4, 25)
R = {}
for fold in [1, 2, 3, 4]:
    FS = "" if fold == 1 else f"_f{fold}"
    lroot = f"{H}/runs/latent_v5t" if fold == 1 else f"{H}/runs/latent_cv_f{fold}"
    PJ = json.load(open(f"{H}/data/z_cand_plan_f{fold}/pairs.json"))
    spl, K, LIB = PJ["split"], PJ["K"], PJ["library"]
    RANK = json.load(open(f"{H}/runs/expA_rank{FS}.json"))
    for P in LV:
        succ = {}
        for k in range(K):
            f = f"{H}/runs/results_candplan_{WHO}{FS}_k{k:02d}_s{SEED}_dr0/eval_student_{P}.json"
            if not os.path.exists(f):
                continue
            res = {r["clip"]: r["success_v2"] for r in json.load(open(f))["clips"]}
            for L, c in PJ["pairs"].get(f"k{k:02d}", {}).items():
                if L in res:
                    succ[(L, c)] = float(res[L])
        if not succ:
            continue
        g = np.load(f"{lroot}/seed42/{P}/train_latents.npz", allow_pickle=True)
        tr = np.array([str(t) for t in g["trial_id"]])
        zq = {c: np.r_[g["z_quality"][tr == c].mean(0), g["z_quality"][tr == c].min(0)] for c in set(tr)}
        for t in TASKS:
            lib = [c for c in LIB[t] if c in zq]
            tc = {c: np.mean([s for (L, cc), s in succ.items() if cc == c and spl.get(L) == "train" and L != c] or [np.nan]) for c in lib}
            lib = [c for c in lib if np.isfinite(tc[c])]
            if len(lib) < 6:
                continue
            X = StandardScaler().fit_transform(np.stack([zq[c] for c in lib]))
            y = np.array([tc[c] for c in lib])
            yh = cross_val_predict(RidgeCV(alphas=AL), X, y, cv=LeaveOneOut())
            R.setdefault((P, t, "r2"), []).append(1 - ((y - yh) ** 2).sum() / ((y - y.mean()) ** 2).sum())
            R.setdefault((P, t, "rho"), []).append(spearmanr(yh, y)[0])
            pred = dict(zip(lib, yh))  # leave-one-out predictions: no candidate predicts itself
            for r in RANK.get(f"42/{P}", []):
                if r["rejected"] or r["routed"] != r["expected"] or r["expected"] != t:
                    continue
                for L in [n for n in spl if n.startswith(t) and spl[n] == "test"]:
                    cands = [c for c in r["ranking"] if (L, c) in succ and c in pred][:5]
                    if len(cands) < 2:
                        continue
                    R.setdefault((P, t, "zq"), []).append(succ[(L, max(cands, key=pred.get))])
                    R.setdefault((P, t, "emp"), []).append(succ[(L, max(cands, key=tc.get))])
                    R.setdefault((P, t, "rank1"), []).append(succ[(L, cands[0])])
                    R.setdefault((P, t, "rand"), []).append(np.mean([succ[(L, c)] for c in cands]))
                    R.setdefault((P, t, "oracle"), []).append(max(succ[(L, c)] for c in cands))

m = lambda P, t, k: np.nanmean(sum((R.get((P, tt, k), []) for tt in ([t] if t else TASKS)), []))  # noqa: E731
print(f"# 执行层：z_quality 能否预测候选的“可迁移性”（student = {WHO}，种子 {SEED}；离线）\n")
print("目标 = 库候选在训练布局上的平均成功率；预测 = z_quality 的岭回归（留一）。选择 = 在 test 布局前 5 个检索候选中选预测最高者。\n")
for t in [None] + TASKS:
    print(f"## {t or '合计'}\n")
    print("| 层级 | R²（留一） | Spearman ρ | 选：z_quality 先验 | 选：实测先验（参考） | 检索第 1 | 前 5 随机 | 前 5 最好 |")
    print("|---|---|---|---|---|---|---|---|")
    for P in LV:
        print(f"| L{P[1]} | {m(P, t, 'r2'):.2f} | {m(P, t, 'rho'):+.2f} | {100 * m(P, t, 'zq'):.1f} | {100 * m(P, t, 'emp'):.1f} | "
              f"{100 * m(P, t, 'rank1'):.1f} | {100 * m(P, t, 'rand'):.1f} | {100 * m(P, t, 'oracle'):.1f} |")
    print()
