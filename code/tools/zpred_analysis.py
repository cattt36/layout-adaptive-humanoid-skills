"""Execution layer, idea "predicted z" (exploratory, offline).  Success of a borrowed z falls steeply with
its distance to the test trial's OWN z (tools/zdep_analysis.py), but the own z is unknown at deployment.  Here a ridge
regression layout geometry -> trial z (clip mean of [z_skill, z_quality]) is fitted per fold, level and action on the
LIBRARY (training) clips only; for a test layout the candidate closest to the predicted z is chosen.
Reported per level (E2E candidate sweep: every test layout x every library candidate, planner clock, 16 runs, ideal):
  R2      : held-out (leave-one-clip-out within the library) R^2 of geometry -> z, and on the test trials
  rho     : within-layout Spearman correlation of success with -distance (predicted z | geometry | own z = oracle)
  pick-k  : mean success of the k candidates nearest by predicted z vs nearest by geometry (k = 1, 3), whole library
  python zpred_analysis.py [who=xm] [seeds=1] > runs/zpred_<who>.md"""
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
SEEDS = [int(s) for s in (sys.argv[2] if len(sys.argv) > 2 else "1").split(",")]
LV = ["Z0", "Z1", "Z2", "Z3"]
TASKS = ["CrouchedTraverse", "ThreeStepPoleRun", "FiveStepPoleRun"]
GEO = {"ThreeStepPoleRun": f"{H}/data/ref_stakes_s93_h10/ThreeStepPoleRun", "FiveStepPoleRun": f"{H}/data/ref_stakes_s83_h10/FiveStepPoleRun",
       "CrouchedTraverse": f"{H}/data/ref_bridge_video_lift/CrouchedTraverse"}
ALPHAS = np.logspace(-2, 3, 21)


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
    d = np.diff(pts, axis=0)
    return pts, np.r_[np.ravel(pts), np.linalg.norm(d, axis=1)], {}


def gdist(A, B):
    (pa, _, ea), (pb, _, eb) = A, B
    if len(pa) != len(pb):
        return np.nan
    d = float(np.linalg.norm(pa - pb, axis=1).mean())
    if ea:
        d += 3.0 * abs(ea["under"] - eb["under"]) + 0.5 * abs(np.angle(np.exp(1j * (ea["yaw"] - eb["yaw"]))))
    return d


def zmeans(lroot, P):
    out = {}
    for part in ["train", "val", "test"]:
        f = f"{lroot}/seed42/{P}/{part}_latents.npz"
        if not os.path.exists(f):
            continue
        g = np.load(f, allow_pickle=True)
        tr = np.array([str(t) for t in g["trial_id"]])
        z = np.concatenate([g["z_skill"], g["z_quality"]], 1)
        for c in set(tr):
            out.setdefault(c, z[tr == c].mean(0))
    return out


def r2(Y, Yh):
    return 1 - ((Y - Yh) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum()


R = {}  # (P, metric) -> list
for fold in [1, 2, 3, 4]:
    FS = "" if fold == 1 else f"_f{fold}"
    lroot = f"{H}/runs/latent_v5t" if fold == 1 else f"{H}/runs/latent_cv_f{fold}"
    PJ = json.load(open(f"{H}/data/z_cand_plan_f{fold}/pairs.json"))
    spl, K, LIB = PJ["split"], PJ["K"], PJ["library"]
    LAY = {n: layout(n) for n in spl}
    for P in LV:
        Z = zmeans(lroot, P)
        succ = {}
        for seed in SEEDS:
            for k in range(K):
                f = f"{H}/runs/results_candplan_{WHO}{FS}_k{k:02d}_s{seed}_dr0/eval_student_{P}.json"
                if not os.path.exists(f):
                    continue
                res = {r["clip"]: r["success_v2"] for r in json.load(open(f))["clips"]}
                for L, c in PJ["pairs"].get(f"k{k:02d}", {}).items():
                    if L in res and spl.get(L) == "test":
                        succ.setdefault((L, c), []).append(float(res[L]))
        if not succ:
            continue
        for t in TASKS:
            lib = [c for c in LIB[t] if c in Z and c in LAY]
            dims = [len(LAY[c][1]) for c in lib]
            nd = max(set(dims), key=dims.count)  # clips with a different stake count cannot share one regression
            lib = [c for c in lib if len(LAY[c][1]) == nd]
            Xl = np.stack([LAY[c][1] for c in lib])
            Yl = np.stack([Z[c] for c in lib])
            sx, sy = StandardScaler().fit(Xl), StandardScaler().fit(Yl)
            m = RidgeCV(alphas=ALPHAS).fit(sx.transform(Xl), sy.transform(Yl))
            Yloo = cross_val_predict(RidgeCV(alphas=ALPHAS), sx.transform(Xl), sy.transform(Yl), cv=LeaveOneOut())
            R.setdefault((P, t, "r2_loo"), []).append(r2(sy.transform(Yl), Yloo))
            tests = [L for L in spl if spl[L] == "test" and L.startswith(t) and L in Z and len(LAY[L][1]) == nd]
            if tests:
                Yt = sy.transform(np.stack([Z[L] for L in tests]))
                R.setdefault((P, t, "r2_test"), []).append(r2(Yt, m.predict(sx.transform(np.stack([LAY[L][1] for L in tests])))))
            for L in tests:
                cands = [c for c in lib if (L, c) in succ]
                if len(cands) < 4:
                    continue
                s = np.array([np.mean(succ[(L, c)]) for c in cands])
                zh = m.predict(sx.transform(LAY[L][1][None]))[0]
                dp = np.array([np.linalg.norm(sy.transform(Z[c][None])[0] - zh) for c in cands])
                dg = np.array([gdist(LAY[L], LAY[c]) for c in cands])
                do = np.array([np.linalg.norm(Z[c] - Z[L]) for c in cands])
                for name, d in (("pred", dp), ("geo", dg), ("own", do)):
                    if np.std(s) > 0:
                        R.setdefault((P, t, f"rho_{name}"), []).append(spearmanr(-d, s)[0])
                    o = np.argsort(d)
                    for kk in (1, 3):
                        R.setdefault((P, t, f"pick{kk}_{name}"), []).append(s[o[:kk]].mean())
                R.setdefault((P, t, "all"), []).append(s.mean())

mean = lambda P, t, k: np.nanmean(sum((R.get((P, tt, k), []) for tt in ([t] if t else TASKS)), []))  # noqa: E731
print(f"# 执行层：用“几何 → z”回归选候选（student = {WHO}，种子 {SEEDS}；离线，基于 E2E 候选扫描）\n")
print("回归只用该折技能库（训练试次）；预测 test 布局的 z，选与预测 z 最近的库候选。成功率为 16 次的平均，%。\n")
for t in [None] + TASKS:
    print(f"## {t or '合计（三个动作）'}\n")
    print("| 层级 | R² 留一（库内） | R² test | ρ 预测 z | ρ 几何 | ρ 自身 z（上限） | 选 1：预测 z | 选 1：几何 | 选 3：预测 z | 选 3：几何 | 选 1：自身 z（上限） | 全库平均 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for P in LV:
        print(f"| L{P[1]} | {mean(P, t, 'r2_loo'):.2f} | {mean(P, t, 'r2_test'):.2f} | {mean(P, t, 'rho_pred'):+.2f} | "
              f"{mean(P, t, 'rho_geo'):+.2f} | {mean(P, t, 'rho_own'):+.2f} | {100 * mean(P, t, 'pick1_pred'):.1f} | "
              f"{100 * mean(P, t, 'pick1_geo'):.1f} | {100 * mean(P, t, 'pick3_pred'):.1f} | {100 * mean(P, t, 'pick3_geo'):.1f} | "
              f"{100 * mean(P, t, 'pick1_own'):.1f} | {100 * mean(P, t, 'all'):.1f} |")
    print()
