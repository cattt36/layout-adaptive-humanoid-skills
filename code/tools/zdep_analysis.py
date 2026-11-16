"""Does the student rely more on z's trial-specific content at higher levels?
Uses the E2E candidate sweep (every test layout x every library candidate of the same action, planner clock, 16 runs,
ideal env).  For each level, success of (layout L, candidate c) is regressed within layout (layout fixed effects) on
  dz : distance between candidate z and the test trial's own z (same level's encoder; clip mean of [z_skill, z_quality];
       scaled by the median dz of that level / fold / action, so levels are comparable)
  dg : geometric layout distance (as in expA_plan.gdist; level-independent; scaled the same way)
  dt : |log duration ratio| (candidate clip vs test trial)
A steeper (more negative) slope at L3 than at L0 means L3 students are hurt more by a borrowed z that differs from the
trial's own.  CI: bootstrap over test layouts (stratified by fold).
  python zdep_analysis.py [who=xm] [seeds=1] > runs/zdep_<who>.md"""
import json
import os
import sys

import numpy as np

H = "/root/gpufree-data/hnu_v2"
WHO = sys.argv[1] if len(sys.argv) > 1 else "xm"
SEEDS = [int(s) for s in (sys.argv[2] if len(sys.argv) > 2 else "1").split(",")]
LV = ["Z0", "Z1", "Z2", "Z3"]
TASKS = ["CrouchedTraverse", "ThreeStepPoleRun", "FiveStepPoleRun"]
GEO = {"ThreeStepPoleRun": f"{H}/data/ref_stakes_s93_h10/ThreeStepPoleRun", "FiveStepPoleRun": f"{H}/data/ref_stakes_s83_h10/FiveStepPoleRun",
       "CrouchedTraverse": f"{H}/data/ref_bridge_video_lift/CrouchedTraverse"}
rng = np.random.default_rng(0)


def yaw_of(q):
    w, x, y, z = q
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def layout(name):
    t = name.split("_")[0]
    ref = np.load(os.path.join(GEO[t], f"{name}_motion.npz"))
    T = ref["body_pos_w"].shape[0]
    p0, yaw0 = ref["body_pos_w"][0, 0], yaw_of(ref["body_quat_w"][0, 0])
    c, s = np.cos(-yaw0), np.sin(-yaw0)
    rot = lambda xy: [c * (xy[0] - p0[0]) - s * (xy[1] - p0[1]), s * (xy[0] - p0[0]) + c * (xy[1] - p0[1])]  # noqa: E731
    if t == "CrouchedTraverse":
        parts = json.load(open(os.path.join(GEO[t], f"{name}_bridge.json")))["parts"]
        pts = [rot(parts["plank"]["center"][:2])] + [rot(parts[p]["center"][:2]) for p in sorted(parts) if p.startswith("post")]
        under = parts["plank"]["center"][2] - parts["plank"]["half"][2]
        yaw = float(np.angle(np.exp(1j * (parts["plank"]["yaw"] - yaw0))))
        return np.array(pts), dict(under=under, yaw=yaw), T
    j = json.load(open(os.path.join(GEO[t], f"{name}_stakes.json")))
    return np.array([rot(p[:2]) for p in j["stake_xyz_50hz"]]), {}, T


def gdist(A, B):
    (pa, ea, _), (pb, eb, _) = A, B
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


rows = []  # fold, seed, level, task, layout, cand, succ, dz, dg, dt
for fold in [1, 2, 3, 4]:
    FS = "" if fold == 1 else f"_f{fold}"
    lroot = f"{H}/runs/latent_v5t" if fold == 1 else f"{H}/runs/latent_cv_f{fold}"
    PJ = json.load(open(f"{H}/data/z_cand_plan_f{fold}/pairs.json"))
    spl, K = PJ["split"], PJ["K"]
    lay = {}
    for P in LV:
        Z = zmeans(lroot, P)
        for seed in SEEDS:
            for k in range(K):
                f = f"{H}/runs/results_candplan_{WHO}{FS}_k{k:02d}_s{seed}_dr0/eval_student_{P}.json"
                if not os.path.exists(f):
                    continue
                res = {r["clip"]: r["success_v2"] for r in json.load(open(f))["clips"]}
                for L, c in PJ["pairs"].get(f"k{k:02d}", {}).items():
                    if L not in res or spl.get(L) != "test" or L == c or L not in Z or c not in Z:
                        continue
                    for n in (L, c):
                        if n not in lay:
                            lay[n] = layout(n)
                    rows.append(dict(fold=fold, seed=seed, P=P, task=L.split("_")[0], L=L, c=c, s=float(res[L]),
                                     dz=float(np.linalg.norm(Z[L] - Z[c])), dg=gdist(lay[L], lay[c]),
                                     dt=abs(np.log(lay[c][2] / lay[L][2]))))

print(f"# z 依赖度分析：借用 z 与本试次差异越大，成功率掉多少（student = {WHO}，种子 {SEEDS}）\n")
print("数据：E2E 候选扫描（每个 test 布局 × 库中同动作的全部候选，规划器时钟，16 次，理想环境）。"
      "斜率 = 在同一布局内，差异增加一个“中位数”时成功率变化的百分点（布局固定效应线性回归）；95% CI 为按折分层、对布局 bootstrap。\n")
have = sorted({(r["fold"], r["seed"], r["P"]) for r in rows})
print("已有（折，种子，层级）：" + "，".join(f"f{f}s{s}{p}" for f, s, p in have) + "\n")


def prep(sub, var):
    sub = [r for r in sub if np.isfinite(r[var])]
    # scale by median within (fold, task) for this level
    med = {}
    for r in sub:
        med.setdefault((r["fold"], r["task"]), []).append(r[var])
    med = {k: np.median(v) if np.median(v) > 0 else 1.0 for k, v in med.items()}
    return [(r["fold"], r["L"], r[var] / med[(r["fold"], r["task"])], r["s"]) for r in sub]


def slope(data):
    by = {}
    for f, L, x, y in data:
        by.setdefault(L, []).append((x, y))
    xs, ys = [], []
    for v in by.values():
        a = np.array(v)
        xs.append(a[:, 0] - a[:, 0].mean())
        ys.append(a[:, 1] - a[:, 1].mean())
    x, y = np.concatenate(xs), np.concatenate(ys)
    return 100 * float((x * y).sum() / (x * x).sum()) if (x * x).sum() > 0 else np.nan


def boot(datas, reps=2000):
    """datas: dict level -> data; resample layouts (stratified by fold) jointly across levels."""
    lays = {}
    for d in datas.values():
        for f, L, _, _ in d:
            lays.setdefault(f, set()).add(L)
    lays = {f: sorted(v) for f, v in lays.items()}
    idx = {P: {} for P in datas}
    for P, d in datas.items():
        for i, (f, L, _, _) in enumerate(d):
            idx[P].setdefault(L, []).append(i)
    out = {P: [] for P in datas}
    for _ in range(reps):
        pick = [L for f, v in lays.items() for L in rng.choice(v, len(v))]
        for P, d in datas.items():
            bd = []
            for j, L in enumerate(pick):
                bd += [(0, f"{L}#{j}", d[i][2], d[i][3]) for i in idx[P].get(L, [])]
            out[P].append(slope(bd))
    return {P: np.array(v) for P, v in out.items()}


ci = lambda b: f"[{np.nanpercentile(b, 2.5):.1f}, {np.nanpercentile(b, 97.5):.1f}]"  # noqa: E731
NAMES = {"dz": "z 距离（借用 z vs 本试次 z）", "dg": "几何距离（障碍物相对起点）", "dt": "时长差 |log 比值|"}
for var in ["dz", "dg", "dt"]:
    print(f"## {NAMES[var]}\n")
    for tsel, tname in [(None, "合计")] + [(t, t) for t in TASKS]:
        datas = {P: prep([r for r in rows if r["P"] == P and (tsel is None or r["task"] == tsel)], var) for P in LV}
        datas = {P: d for P, d in datas.items() if d}
        if "Z0" not in datas or "Z3" not in datas:
            continue
        B = boot(datas)
        cells = [f"L{P[1]} {slope(datas[P]):+.1f} {ci(B[P])}" for P in LV if P in datas]
        diff = B["Z3"] - B["Z0"]
        print(f"- **{tname}**：" + "；".join(cells)
              + f"；**L3 − L0 = {slope(datas['Z3']) - slope(datas['Z0']):+.1f} {ci(diff)}**")
    print()

# raw: success on nearest vs farthest third of candidates (by dz) per level
print("## 成功率：按 z 距离三等分（每个布局内排序）\n")
print("| 层级 | 最近 1/3 | 中间 1/3 | 最远 1/3 | 最近 − 最远 |")
print("|---|---|---|---|---|")
for P in LV:
    sub = [r for r in rows if r["P"] == P]
    if not sub:
        continue
    by = {}
    for r in sub:
        by.setdefault((r["seed"], r["L"]), []).append(r)
    b = [[], [], []]
    for v in by.values():
        v = sorted(v, key=lambda r: r["dz"])
        for i, part in enumerate(np.array_split(np.arange(len(v)), 3)):
            b[i] += [v[j]["s"] for j in part]
    m = [100 * np.mean(x) for x in b]
    print(f"| L{P[1]} | {m[0]:.1f} | {m[1]:.1f} | {m[2]:.1f} | {m[0] - m[2]:+.1f} |")
