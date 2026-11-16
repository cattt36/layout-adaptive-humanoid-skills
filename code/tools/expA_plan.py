"""Full chain with the new execution layer: instruction -> z_sem -> router (abstain) -> retrieve top-5
library clips -> geometric planner per candidate (tools/make_z_cands_plan.py) -> Q_cand choice (+ rejection) ->
student.  Reported in two layers (user decision ):
  instruction layer : in-scope commands accepted (coverage), action correct among accepted, out-of-scope / ambiguous
                      commands refused (frozen router, sealed test set)
  execution layer   : success of the executed candidate over (accepted and correctly routed command x test layout)
  end to end        : over all (in-scope command x test layout); refusal / wrong action = failure
Q_cand as in tools/expA_full.py (logistic regression on layout geometry + candidate z_quality / z_skill, fitted on the
TRAINING layouts of the sweep; rejection threshold = 10 % quantile of the best candidate score on the val layouts).
Students: new (xm, cross-trial z, mix) and old (n, v6g10n), fold 1, both under the planner clock.
  python expA_plan.py [seed=1] [fold=1] > runs/expA_plan[_f<k>]_s1.md   (folds 2-4: tools/e2e_router_folds.sh, expA_rank.py <k>)"""
import json
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

H = "/root/gpufree-data/hnu_v2"
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 1
FOLD = int(sys.argv[2]) if len(sys.argv) > 2 else 1
FS = "" if FOLD == 1 else f"_f{FOLD}"
LROOT = f"{H}/runs/latent_v5t" if FOLD == 1 else f"{H}/runs/latent_cv_f{FOLD}"
LV = ["Z0", "Z1", "Z2", "Z3"]
WHO = [("xm", "新 student（跨试次 z 增广，混合）"), ("n", "旧 student")]
TASKS = ["CrouchedTraverse", "ThreeStepPoleRun", "FiveStepPoleRun"]
TN = {"CrouchedTraverse": "独木桥", "ThreeStepPoleRun": "三步桩", "FiveStepPoleRun": "五步桩"}
GEO = {"ThreeStepPoleRun": f"{H}/data/ref_stakes_s93_h10/ThreeStepPoleRun", "FiveStepPoleRun": f"{H}/data/ref_stakes_s83_h10/FiveStepPoleRun",
       "CrouchedTraverse": f"{H}/data/ref_bridge_video_lift/CrouchedTraverse"}
TOPK = 5
# variants (defaults reproduce the frozen analysis):
#   HNU_RTAG=v3 : router v3 (expA_rank*_v3.json, language_routing_testv3.json)
#   HNU_QZ=1    : Q_cand gets one more feature, the distance of the candidate z to the z PREDICTED from the layout
#                 geometry (ridge on the library clips; leave-one-out for library layouts), + selection rule "zpred"
#   HNU_WHO_EXTRA=rwtt,rwts : + external baseline "retrieve - warp - track" (tools/rwt_sweep.sh), rows file *_rwt
RT = os.environ.get("HNU_RTAG", "")
QZ = os.environ.get("HNU_QZ", "") == "1"
EXTRA = [w for w in os.environ.get("HNU_WHO_EXTRA", "").split(",") if w]
WHO += [(w, {"rwtt": "基线：检索 + 时间对齐 + 跟踪", "rwts": "基线：检索 + 时空对齐 + 跟踪",
             "zr": "公平基线 ZR（显式参考学生）"}.get(w, w)) for w in EXTRA]
XS = "_rwt" if EXTRA in (["rwtt", "rwts"], ["rwtt"], ["rwts"]) else ("_" + "_".join(EXTRA) if EXTRA else "")
#   HNU_EVAL_DR=1 : (preview robustness check, tools/preview_dr_sweep.sh) Q_cand is still fitted / thresholded on the
#                   nominal sweep (as deployed), but every chosen candidate is scored on the randomised + latency sweep
EVAL_DR = int(os.environ.get("HNU_EVAL_DR", "0"))
SUF = (f"_{RT}" if RT else "") + ("_qz" if QZ else "") + XS + (f"_evdr{EVAL_DR}" if EVAL_DR else "")
PJ = json.load(open(f"{H}/data/z_cand_plan_f{FOLD}/pairs.json"))
LIB, K, SPL = PJ["library"], PJ["K"], PJ["split"]
RANK = json.load(open(f"{H}/runs/expA_rank{FS}{'_' + RT if RT else ''}.json"))
cases = {c["id"]: c for c in json.load(open(f"{H}/code/data_lang/router_test_v2_sealed.json", encoding="utf-8"))["cases"]}
rng = np.random.default_rng(0)


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


LAY = {n: layout(n) for n in SPL}


def gdist(a, b):
    pa, _, ea = LAY[a]
    pb, _, eb = LAY[b]
    if len(pa) != len(pb):
        return 1e9
    d = float(np.linalg.norm(pa - pb, axis=1).mean())
    if ea:
        d += 3.0 * abs(ea["under"] - eb["under"]) + 0.5 * abs(np.angle(np.exp(1j * (ea["yaw"] - eb["yaw"]))))
    return d


def load_sweep(who, P, dr=0):
    """-> succ, fall (per pair means) and, when the per-rollout file exists, ep / fe: the 16 rollouts' success_v2 and
    fall flags per pair (simulation preview = rollouts 0-3, execution scored on rollouts 8-15)."""
    succ, fall, ep, fe = {}, {}, {}, {}
    for k in range(K):
        d = f"{H}/runs/results_candplan_{who}{FS}_k{k:02d}_s{SEED}_dr{dr}"
        f = f"{d}/eval_student_{P}.json"
        if not os.path.exists(f):
            return None
        rows = {r["clip"]: r for r in json.load(open(f))["clips"]}
        q = f"{d}/qexec_{P}.npz"
        fz = np.load(q) if os.path.exists(q) else None
        names = [str(s) for s in fz["names"]] if fz is not None else []
        for n, c in PJ["pairs"].get(f"k{k:02d}", {}).items():
            if n not in rows:
                continue
            succ[(n, c)] = rows[n]["success_v2"]
            if fz is not None and n in names:
                sel = fz["clip"] == names.index(n)
                fall[(n, c)] = float(np.nanmean(fz["fell"][sel].astype(float))) if np.isfinite(fz["fell"][sel].astype(float)).any() else np.nan
                if sel.sum() >= 16:
                    ep[(n, c)] = fz["success_v2"][sel][:16].astype(float)
                    fe[(n, c)] = fz["fell"][sel][:16].astype(float)
    return succ, fall, ep, fe


def zfeat(P, clip, cache={}):
    if (P, clip) not in cache:
        g = np.load(f"{LROOT}/seed42/{P}/train_latents.npz", allow_pickle=True)
        tr = np.array([str(t) for t in g["trial_id"]])
        for c in set(tr):
            m = tr == c
            cache[(P, c)] = np.r_[g["z_quality"][m].mean(0), g["z_quality"][m].min(0), g["z_skill"][m].mean(0)]
    return cache[(P, clip)]


def zmean(P, clip, cache={}):
    if (P, clip) not in cache:
        g = np.load(f"{LROOT}/seed42/{P}/train_latents.npz", allow_pickle=True)
        tr = np.array([str(t) for t in g["trial_id"]])
        z = np.concatenate([g["z_skill"], g["z_quality"]], 1)
        for c in set(tr):
            cache[(P, c)] = z[tr == c].mean(0)
    return cache.get((P, clip))


def zpred_model(P, t, cache={}):
    """ridge geometry -> z on the library clips of action t; leave-one-out predictions for the library layouts"""
    if (P, t) not in cache:
        from sklearn.linear_model import RidgeCV
        from sklearn.model_selection import LeaveOneOut, cross_val_predict
        lib = [c for c in LIB[t] if c in LAY and zmean(P, c) is not None]
        dims = [len(LAY[c][1]) for c in lib]
        nd = max(set(dims), key=dims.count)
        lib = [c for c in lib if len(LAY[c][1]) == nd]
        sx, sy = StandardScaler().fit(np.stack([LAY[c][1] for c in lib])), StandardScaler().fit(np.stack([zmean(P, c) for c in lib]))
        X, Y = sx.transform(np.stack([LAY[c][1] for c in lib])), sy.transform(np.stack([zmean(P, c) for c in lib]))
        al = np.logspace(-2, 3, 21)
        m = RidgeCV(alphas=al).fit(X, Y)
        loo = dict(zip(lib, cross_val_predict(RidgeCV(alphas=al), X, Y, cv=LeaveOneOut())))
        cache[(P, t)] = (m, sx, sy, nd, loo, {})
    return cache[(P, t)]


def dzp(P, L, c):
    m, sx, sy, nd, loo, memo = zpred_model(P, L.split("_")[0])
    if (L, c) not in memo:
        zc = zmean(P, c)
        if len(LAY[L][1]) != nd or zc is None:
            memo[(L, c)] = np.nan
        else:
            zh = loo[L] if L in loo else m.predict(sx.transform(LAY[L][1][None]))[0]
            memo[(L, c)] = float(np.linalg.norm(sy.transform(zc[None])[0] - zh))
    return memo[(L, c)]


def feat(P, L, c):
    f = np.r_[LAY[L][1], LAY[c][1], LAY[L][1] - LAY[c][1], gdist(L, c), zfeat(P, c)]
    if QZ:
        d = dzp(P, L, c)
        f = np.r_[f, d if np.isfinite(d) else 10.0]
    return f


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
        models[t] = (sc, LogisticRegression(C=1.0, max_iter=5000).fit(sc.transform(np.array(X)), np.array(y), sample_weight=np.array(w)))
    return lambda L, c: float(models[L.split("_")[0]][1].predict_proba(models[L.split("_")[0]][0].transform(feat(P, L, c)[None]))[0, 1])


METHODS = [("oracle", "前 5 候选中实际最好（上限）"), ("qcand", "**Q_cand 选择（主方法）**"), ("geo", "前 5 候选中几何最近"),
           ("geoall", "全库几何最近（不经检索，参考）"), ("rank1", "检索排名第 1（不用 Q）"), ("random", "前 5 随机（期望）")]
if QZ:
    METHODS.insert(3, ("zpred", "前 5 中预测 z 最近"))


def rows_for(P, data, qc, tau, evdata=None):
    succ, fall = (evdata if evdata is not None else data)[:2]  # values (and the oracle) from the evaluation sweep
    pep = data[2]  # preview rollouts (0-3) from the nominal sweep
    eep, efe = (evdata if evdata is not None else data)[2:4]  # execution rollouts (8-15)
    out = []
    for r in RANK[f"42/{P}"]:
        ok = (not r["rejected"]) and r["routed"] == r["expected"]
        for L in [n for n in SPL if n.startswith(r["expected"]) and SPL[n] == "test"]:
            v = dict(task=r["expected"], instr=r["id"], layout=L, ok=ok)
            if ok:
                cands = [c for c in r["ranking"] if (L, c) in succ][:TOPK]
                allc = [c for c in LIB[r["expected"]] if (L, c) in succ]
                if not cands:
                    continue
                qs = {c: qc(L, c) for c in cands}
                pick = dict(qcand=max(qs, key=qs.get), geo=min(cands, key=lambda c: gdist(L, c)), rank1=cands[0],
                            oracle=max(cands, key=lambda c: succ[(L, c)]), geoall=min(allc, key=lambda c: gdist(L, c)))
                if QZ:
                    pick["zpred"] = min(cands, key=lambda c: dzp(P, L, c) if np.isfinite(dzp(P, L, c)) else 1e9)
                v.update({m: succ[(L, c)] for m, c in pick.items()})
                v["random"] = float(np.mean([succ[(L, c)] for c in cands]))
                v["accepted"] = max(qs.values()) >= tau
                v["fall"] = fall.get((L, pick["qcand"]), np.nan)
                # simulation preview (all top-5 candidates, rollouts 0-3; ties -> retrieval order) and the
                # two refusal rules; every method of this block is scored on rollouts 8-15 (disjoint from the preview)
                if all((L, c) in pep and (L, c) in eep for c in cands):
                    pv = {c: float(pep[(L, c)][:4].mean()) for c in cands}
                    cpv = max(cands, key=lambda c: (pv[c], -cands.index(c)))
                    cq = pick["qcand"]
                    ex = lambda c: float(eep[(L, c)][8:].mean())  # noqa: E731
                    fl = lambda c: float(np.nanmean(efe[(L, c)][8:])) if np.isfinite(efe[(L, c)][8:]).any() else np.nan  # noqa: E731
                    v.update(q8=ex(cq), pv8=ex(cpv), q8_fall=fl(cq), pv8_fall=fl(cpv), pv_best=pv[cpv],
                             pv_accepted=bool(pv[cpv] >= 0.5), oracle8=max(ex(c) for c in cands))
            out.append(v)
    return out


def e2e(rows, m, reject=False):
    return np.array([0.0 if (not x["ok"] or (reject and not x["accepted"])) else x[m] for x in rows])


def execl(rows, m, reject=False):
    return np.array([x[m] for x in rows if x["ok"] and (not reject or x["accepted"])])


def boot(rows, m, layer, reps=5000):
    instr, lays = sorted({x["instr"] for x in rows}), sorted({x["layout"] for x in rows})
    val = {(x["instr"], x["layout"]): x for x in rows}
    out = []
    for _ in range(reps):
        bi, bl = rng.choice(instr, len(instr)), rng.choice(lays, len(lays))
        v = []
        for i in bi:
            for l in bl:
                x = val.get((i, l))
                if x is None:
                    continue
                if layer == "e2e":
                    v.append(x[m] if x["ok"] else 0.0)
                elif x["ok"]:
                    v.append(x[m])
        out.append(np.mean(v) if v else np.nan)
    return np.array(out)


ci = lambda b: f"[{100 * np.nanpercentile(b, 2.5):.1f}, {100 * np.nanpercentile(b, 97.5):.1f}]"  # noqa: E731
print(f"# 完整链路（新执行层：规划器 + 跨试次 z 增广 student），第 {FOLD} 折种子 {SEED}，{sum(v == 'test' for v in SPL.values())} 个 test 布局\n")
print("指令层用冻结的路由器（封存测试集，latent 种子 42）；执行层每个布局 × 候选执行 16 次，理想条件，成功标准 v2；"
      "三步桩/五步桩的踩桩时机和独木桥的进桥时间都由几何规划器给出，不使用任何真人时序。\n")
print("## 1. 指令层（与 student 无关）\n")
print("| 层级 | 应执行指令被接受（覆盖率） | 接受后选对动作 | 越界 / 歧义指令被拒绝 |")
print("|---|---|---|---|")
for P in LV:
    rs = RANK[f"42/{P}"]
    acc = [r for r in rs if not r["rejected"]]
    route = json.load(open(f"{LROOT}/seed42/{P}/language_routing_test{RT or 'v2'}.json", encoding="utf-8"))
    oos = [x for x in route if cases[x["id"]]["should_abstain"]]
    print(f"| L{P[1]} | {100 * len(acc) / len(rs):.1f}%（{len(acc)}/{len(rs)}） | "
          f"{100 * np.mean([r['routed'] == r['expected'] for r in acc]):.1f}% | {100 * np.mean([x['rejected'] for x in oos]):.1f}% |")
print()
RES = {}
for who, wn in WHO:
    for P in LV:
        d = load_sweep(who, P)
        if d is None:
            continue
        qc = fit_qcand(P, d[0])
        vals = [max(qc(L, c) for c in [c for c in r["ranking"] if (L, c) in d[0]][:TOPK])
                for r in RANK[f"42/{P}"] if not r["rejected"] and r["routed"] == r["expected"]
                for L in [n for n in SPL if n.startswith(r["expected"]) and SPL[n] == "val"]
                if any((L, c) in d[0] for c in r["ranking"])]
        tau = float(np.quantile(vals, 0.10))
        de = load_sweep(who, P, EVAL_DR) if EVAL_DR else None
        if EVAL_DR and de is None:
            continue
        RES[(who, P)] = (rows_for(P, d, qc, tau, de), tau)
json.dump({f"{who}/{P}": [{k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in x.items()} for x in rows]
           for (who, P), (rows, _) in RES.items()}, open(f"{H}/runs/expA_plan{FS}_s{SEED}{SUF}_rows.json", "w"))
print("## 2. 执行层（被接受且动作正确的指令 × test 布局；成功率 %）\n")
print("| student | 层级 | " + " | ".join(n for _, n in METHODS) + " |")
print("|---|---|" + "---|" * len(METHODS))
for who, wn in WHO:
    for P in LV:
        if (who, P) not in RES:
            continue
        rows = RES[(who, P)][0]
        print(f"| {wn} | L{P[1]} | " + " | ".join(f"{100 * execl(rows, m).mean():.1f}" for m, _ in METHODS) + " |")
print("\n### 执行层，Q_cand 主方法：95% CI、分动作、拒识\n")
print("| student | 层级 | 成功率 [95% CI] | 独木桥 | 三步桩 | 五步桩 | 加拒识后：接受时成功率 | 接受比例 | 摔倒率（不拒识 → 拒识） |")
print("|---|---|---|---|---|---|---|---|---|")
for who, wn in WHO:
    for P in LV:
        if (who, P) not in RES:
            continue
        rows, tau = RES[(who, P)]
        okr = [x for x in rows if x["ok"]]
        tk = [100 * np.mean([x["qcand"] for x in okr if x["task"] == t]) for t in TASKS]
        accr = [x for x in okr if x["accepted"]]
        f0 = np.nanmean([x["fall"] for x in okr])
        f1 = np.nanmean([x["fall"] if x["accepted"] else 0.0 for x in okr])
        print(f"| {wn} | L{P[1]} | {100 * execl(rows, 'qcand').mean():.1f} {ci(boot(rows, 'qcand', 'exec'))} | "
              + " | ".join(f"{v:.1f}" for v in tk)
              + f" | {100 * np.mean([x['qcand'] for x in accr]):.1f} | {100 * len(accr) / len(okr):.1f}% | {100 * f0:.1f} → {100 * f1:.1f} |")
print("\n## 3. 端到端（全部应执行指令 × test 布局；拒识 / 选错动作算失败；%）\n")
print("| student | 层级 | Q_cand [95% CI] | Q_cand + 拒识 | 前 5 几何最近 | 检索排名第 1 |")
print("|---|---|---|---|---|---|")
for who, wn in WHO:
    for P in LV:
        if (who, P) not in RES:
            continue
        rows = RES[(who, P)][0]
        print(f"| {wn} | L{P[1]} | {100 * e2e(rows, 'qcand').mean():.1f} {ci(boot(rows, 'qcand', 'e2e'))} | "
              f"{100 * e2e(rows, 'qcand', True).mean():.1f} | {100 * e2e(rows, 'geo').mean():.1f} | {100 * e2e(rows, 'rank1').mean():.1f} |")
print("\n参考：原 E2E（实验 A，旧 student，人的踩桩时钟，种子 1）Q_cand：L0 37.1、L3 36.9。")
