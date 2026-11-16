"""Execution layer: a better candidate selector "Q2" (exploratory, offline; same for every level).
Q_cand (tools/expA_plan.py) is no better than "geometrically nearest": pointwise logistic regression on raw layout
coordinates + z features, few training layouts, and it spends its capacity on how hard a LAYOUT is although the choice
only needs the order of the candidates WITHIN a layout.  Q2:
  features : how much the planner has to change the candidate for this layout (stakes: raw swing-distance ratios
             mean / min / max and how many fall outside the clip range [0.6, 1.6], shift of the first stance onset,
             final compression, duration ratio; beam: planned / own entry frame, duration ratio, plank height and yaw
             difference), start offset and heading difference, start-relative geometric distance (expA_plan.gdist),
             and the candidate's track record (mean success on the OTHER training layouts)
  model    : pairwise ranking logistic regression within each training layout (pairs with |success difference| >= 0.1,
             weighted by the difference), one model per action; C chosen by 5-fold leave-layouts-out CV on the
             training layouts (criterion: success of the top-1 candidate of the held-out layouts)
Scored on the test layouts exactly like the execution layer of expA_plan.py (accepted and correctly routed commands of
the frozen router x test layouts, top-5 retrieved candidates).  Also "Q2 top-2 + preview": the two best candidates by
Q2, two simulated preview episodes each (episodes 0-1), executed = better preview; all methods of that table are scored
on episodes 8-15 only.
  python q2_explore.py [who=xm] [seed=1] > runs/q2_<who>.md"""
import json
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

H = "/root/gpufree-data/hnu_v2"
WHO = sys.argv[1] if len(sys.argv) > 1 else "xm"
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 1
LV = ["Z0", "Z1", "Z2", "Z3"]
TASKS = ["CrouchedTraverse", "ThreeStepPoleRun", "FiveStepPoleRun"]
GEO = {"ThreeStepPoleRun": f"{H}/data/ref_stakes_s93_h10/ThreeStepPoleRun", "FiveStepPoleRun": f"{H}/data/ref_stakes_s83_h10/FiveStepPoleRun"}
BRIDGE = f"{H}/data/ref_bridge_video_lift/CrouchedTraverse"
CS = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
rng = np.random.default_rng(0)


def yaw_of(q):
    w, x, y, z = q
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def wrap(a):
    return float(np.angle(np.exp(1j * a)))


INFO = {}


def info(n):
    if n not in INFO:
        t = n.split("_")[0]
        if t == "CrouchedTraverse":
            m = np.load(os.path.join(BRIDGE, f"{n}_motion.npz"))
            p = json.load(open(os.path.realpath(os.path.join(BRIDGE, f"{n}_bridge.json"))))["parts"]["plank"]
            c, yaw, half = np.array(p["center"]), p["yaw"], p["half"]
            d, nn = np.array([np.cos(yaw), np.sin(yaw)]), np.array([-np.sin(yaw), np.cos(yaw)])
            rel = m["body_pos_w"][:, :, :2] - c[:2]
            inside = (np.abs(rel @ nn) < 0.25) & (np.abs(rel @ d) < half[0])
            entry = int(np.where(inside.any(1))[0][0]) if inside.any() else -1
            p0 = m["body_pos_w"][0, 0, :2]
            INFO[n] = dict(p0=p0, yaw0=yaw_of(m["body_quat_w"][0, 0]), T=len(m["body_pos_w"]), entry=entry,
                           dist=float(min(np.linalg.norm(c[:2] + s * d * half[0] - p0) for s in (1, -1))),
                           under=float(c[2] - half[2]), pyaw=float(yaw), pc=c[:2])
        else:
            m = np.load(os.path.join(GEO[t], f"{n}_motion.npz"))
            j = json.load(open(os.path.join(GEO[t], f"{n}_stakes.json")))
            INFO[n] = dict(p0=m["body_pos_w"][0, 0, :2], yaw0=yaw_of(m["body_quat_w"][0, 0]), T=len(m["body_pos_w"]),
                           pts=np.array([p[:2] for p in j["stake_xyz_50hz"]]),
                           win=np.array([(s["stance"][0] * 50.0 / j["fps"], s["stance"][1] * 50.0 / j["fps"]) for s in j["stakes"]]))
    return INFO[n]


def rel_pts(n):
    """stake / plank points relative to the start pose (as expA_plan.layout)."""
    i = info(n)
    c, s = np.cos(-i["yaw0"]), np.sin(-i["yaw0"])
    R = np.array([[c, -s], [s, c]])
    pts = i["pts"] if "pts" in i else i["pc"][None]
    return (pts - i["p0"]) @ R.T


def feats(L, c, REG, RD, prior):
    a, b = info(L), info(c)
    t = L.split("_")[0]
    common = [np.linalg.norm(a["p0"] - b["p0"]), abs(wrap(a["yaw0"] - b["yaw0"])), np.log(a["T"] / b["T"]), prior]
    pa, pb = rel_pts(L), rel_pts(c)
    gd = float(np.linalg.norm(pa - pb, axis=1).mean()) if len(pa) == len(pb) else 5.0
    if t == "CrouchedTraverse":
        pred = float(np.clip(np.polyval(RD, a["dist"]), 10, a["T"] - 30))
        ent = max(b["entry"], 1)
        return np.r_[common, gd, np.log(pred / ent), abs(a["under"] - b["under"]), abs(wrap(a["pyaw"] - b["pyaw"]))]
    seg = lambda p0, p: np.linalg.norm(np.diff(np.vstack([p0, p]), axis=0), axis=1)  # noqa: E731
    r = seg(a["p0"], a["pts"]) / np.maximum(seg(b["p0"], b["pts"]), 1e-3)
    lr = np.log(np.maximum(r, 1e-3))
    ws = b["win"]
    a0 = float(np.polyval(REG[t], np.linalg.norm(a["pts"][0] - a["p0"])))
    gaps = np.diff(np.concatenate([[0.0], ws.flatten()]))[0::2]
    stance = ws[:, 1] - ws[:, 0]
    tt, out = 0.0, []
    for k in range(len(ws)):
        s0 = a0 if k == 0 else tt + gaps[k] * np.clip(r[k], 0.6, 1.6)
        out.append(s0 + stance[k])
        tt = out[-1]
    comp = max(out[-1] / max(a["T"] - 11, 1), 1.0)
    return np.r_[common, gd, lr.mean(), lr.min(), lr.max(), np.sum((r < 0.6) | (r > 1.6)), (a0 - ws[0, 0]) / 50.0,
                 abs(a0 - ws[0, 0]) / 50.0, np.log(comp)]


def fit_rank(X, y, groups, C):
    """pairwise ranking within groups -> weight vector (on standardised features)."""
    sc = StandardScaler().fit(X)
    Xs = sc.transform(X)
    D, Y, W = [], [], []
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        for i in idx:
            for j in idx:
                if i < j and abs(y[i] - y[j]) >= 0.1:
                    D += [Xs[i] - Xs[j], Xs[j] - Xs[i]]
                    Y += [int(y[i] > y[j]), int(y[j] > y[i])]
                    W += [abs(y[i] - y[j])] * 2
    if len(set(Y)) < 2:
        return lambda Z: np.zeros(len(Z))
    m = LogisticRegression(C=C, fit_intercept=False, max_iter=5000).fit(np.array(D), np.array(Y), sample_weight=np.array(W))
    return lambda Z: m.decision_function(sc.transform(Z))


R = {}
for fold in [1, 2, 3, 4]:
    FS = "" if fold == 1 else f"_f{fold}"
    PJ = json.load(open(f"{H}/data/z_cand_plan_f{fold}/pairs.json"))
    spl, K, LIB = PJ["split"], PJ["K"], PJ["library"]
    RANK = json.load(open(f"{H}/runs/expA_rank{FS}.json"))
    rows_f = f"{H}/runs/expA_plan{FS}_s{SEED}_rows.json"
    OLD = json.load(open(rows_f)) if os.path.exists(rows_f) else {}
    REG = {t: np.polyfit([np.linalg.norm(info(n)["pts"][0] - info(n)["p0"]) for n in LIB[t]], [info(n)["win"][0, 0] for n in LIB[t]], 1)
           for t in GEO}
    bl = [n for n in LIB["CrouchedTraverse"] if info(n)["entry"] > 0]
    RD = np.polyfit([info(n)["dist"] for n in bl], [info(n)["entry"] for n in bl], 1)
    for P in LV:
        succ, ep = {}, {}
        for k in range(K):
            d = f"{H}/runs/results_candplan_{WHO}{FS}_k{k:02d}_s{SEED}_dr0"
            if not os.path.exists(f"{d}/eval_student_{P}.json"):
                continue
            res = {r["clip"]: r["success_v2"] for r in json.load(open(f"{d}/eval_student_{P}.json"))["clips"]}
            z = np.load(f"{d}/qexec_{P}.npz") if os.path.exists(f"{d}/qexec_{P}.npz") else None
            names = [str(s) for s in z["names"]] if z is not None else []
            for L, c in PJ["pairs"].get(f"k{k:02d}", {}).items():
                if L in res:
                    succ[(L, c)] = float(res[L])
                    if L in names:
                        e = z["success_v2"][z["clip"] == names.index(L)].astype(float)
                        if len(e) >= 16:
                            ep[(L, c)] = e[:16]
        if not succ:
            continue
        for t in TASKS:
            lib = [c for c in LIB[t]]
            train_L = sorted({L for (L, c) in succ if L.startswith(t) and spl.get(L) == "train"})

            def prior(c, excl):
                v = [succ[(L, c)] for L in train_L if L != excl and (L, c) in succ]
                return float(np.mean(v)) if v else 0.5

            X, y, g = [], [], []
            for L in train_L:
                for c in lib:
                    if (L, c) in succ:
                        X.append(feats(L, c, REG, RD, prior(c, L)))
                        y.append(succ[(L, c)])
                        g.append(L)
            X, y, g = np.array(X), np.array(y), np.array(g)
            if len(set(g)) < 6:
                continue
            best_C, best_v = CS[0], -1
            for C in CS:
                v = []
                for tr, te in GroupKFold(n_splits=5).split(X, y, g):
                    f = fit_rank(X[tr], y[tr], g[tr], C)
                    s = f(X[te])
                    for L in np.unique(g[te]):
                        m = g[te] == L
                        v.append(y[te][m][np.argmax(s[m])])
                if np.mean(v) > best_v + 1e-9:
                    best_C, best_v = C, np.mean(v)
            q2 = fit_rank(X, y, g, best_C)
            R.setdefault((P, t, "C"), []).append(best_C)
            old = {(x["instr"], x["layout"]): x for x in OLD.get(f"{WHO}/{P}", []) if x.get("ok")}
            for r in RANK.get(f"42/{P}", []):
                if r["rejected"] or r["routed"] != r["expected"] or r["expected"] != t:
                    continue
                for L in [n for n in spl if n.startswith(t) and spl[n] == "test"]:
                    cands = [c for c in r["ranking"] if (L, c) in succ][:5]
                    if len(cands) < 2:
                        continue
                    sc = q2(np.array([feats(L, c, REG, RD, prior(c, None)) for c in cands]))
                    pri = [prior(c, None) for c in cands]
                    gd = [feats(L, c, REG, RD, 0)[4] for c in cands]
                    pick = dict(q2=cands[int(np.argmax(sc))], prior=cands[int(np.argmax(pri))], geo=cands[int(np.argmin(gd))],
                                rank1=cands[0], oracle=max(cands, key=lambda c: succ[(L, c)]))
                    for mth, c in pick.items():
                        R.setdefault((P, t, mth), []).append(succ[(L, c)])
                    R.setdefault((P, t, "random"), []).append(np.mean([succ[(L, c)] for c in cands]))
                    if (r["id"], L) in old:
                        R.setdefault((P, t, "qold"), []).append(old[(r["id"], L)]["qcand"])
                    if all((L, c) in ep for c in cands):  # preview table: everything scored on episodes 8-15
                        ev = {c: ep[(L, c)][8:].mean() for c in cands}
                        top2 = [cands[i] for i in np.argsort(-sc)[:2]]
                        hyb = max(top2, key=lambda c: (ep[(L, c)][:2].mean(), -top2.index(c)))
                        pv4 = max(cands, key=lambda c: (ep[(L, c)][:4].mean(), -cands.index(c)))
                        for mth, v in dict(e_q2=ev[pick["q2"]], e_geo=ev[pick["geo"]], e_hyb=ev[hyb], e_pv4=ev[pv4],
                                           e_oracle=max(ev.values())).items():
                            R.setdefault((P, t, mth), []).append(v)

m = lambda P, t, k: 100 * np.nanmean(sum((R.get((P, tt, k), []) for tt in ([t] if t else TASKS)), []) or [np.nan])  # noqa: E731
print(f"# 执行层：候选选择器 Q2（student = {WHO}，种子 {SEED}；离线，基于 E2E 候选扫描）\n")
print("比较集合同 expA_plan 执行层（冻结路由器接受且动作正确的指令 × test 布局，前 5 个检索候选）。Q2 只用训练布局拟合，"
      "正则强度由训练布局上的留布局交叉验证选择。\n")
for t in [None] + TASKS:
    print(f"## {t or '合计'}\n")
    print("| 层级 | 旧 Q | **Q2** | 只用口碑 | 几何最近 | 检索第 1 | 前 5 随机 | 前 5 最好 | ‖ | 第 9–16 次评估：Q2 | 几何 | Q2 前 2 + 各预演 2 次 | 全部预演 4 次 | 上限 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for P in LV:
        print(f"| L{P[1]} | {m(P, t, 'qold'):.1f} | **{m(P, t, 'q2'):.1f}** | {m(P, t, 'prior'):.1f} | {m(P, t, 'geo'):.1f} | "
              f"{m(P, t, 'rank1'):.1f} | {m(P, t, 'random'):.1f} | {m(P, t, 'oracle'):.1f} | ‖ | {m(P, t, 'e_q2'):.1f} | "
              f"{m(P, t, 'e_geo'):.1f} | {m(P, t, 'e_hyb'):.1f} | {m(P, t, 'e_pv4'):.1f} | {m(P, t, 'e_oracle'):.1f} |")
    print()
print("交叉验证选出的正则强度 C：" + "；".join(f"L{P[1]} {t[:5]} {R.get((P, t, 'C'))}" for P in LV for t in TASKS))
