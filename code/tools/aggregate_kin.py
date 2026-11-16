"""L3-kin ablation summary (tools/run_l3kin.sh): per fold x latent seed (12 encoders) the offline 8.1 metrics and the
linear probes of L2, L3-kin (L3 without human forces / moments) and L3; paired differences
  L3-kin - L2 : contribution of human kinematics + gait timing
  L3 - L3-kin : contribution of human dynamics (joint moments, ground reaction forces)
Self-check: the Z0-Z2 rows of the L3-kin paper roots must equal those of the standard paper roots.
  python aggregate_kin.py > runs/l3kin_summary.md"""
import csv
import json
import os

import numpy as np

H = "/root/gpufree-data/hnu_v2"
PR = f"{H}/paper_root"
FS = [(F, S) for F in (1, 2, 3, 4) for S in (42, 43, 44)]


def std_root(F, S):
    return f"{PR}/latent_v5t_seed{S}" if F == 1 else f"{PR}/latent_cv_f{F}_seed{S}"


def metrics(root):
    """level -> metric -> value (mean over actions for the latent metrics, task macro for retrieval)."""
    o = f"{root}/Results/8.1_Offline_Representation/outputs"
    out = {}
    f = f"{o}/offline_retrieval_macro_by_level.csv"
    if os.path.exists(f):
        for r in csv.DictReader(open(f, encoding="utf-8-sig")):
            lv = r["supervision_level"]
            for k, col in (("R@1", "task_macro_r_at_1"), ("MRR", "task_macro_mrr")):
                out.setdefault(lv, {})[f"检索 {r['protocol']} {k}"] = float(r[col])
    f = f"{o}/offline_latent_metrics.csv"
    if os.path.exists(f):
        acc = {}
        for r in csv.DictReader(open(f, encoding="utf-8-sig")):
            for k, col in (("步态相位 BA", "phase_balanced_accuracy"), ("部署语义余弦", "deploy_semantic_cosine"),
                           ("执行质量 AUROC", "quality_macro_auroc")):
                if r.get(col) not in ("", None):
                    acc.setdefault((r["supervision_level"], k), []).append(float(r[col]))
        for (lv, k), v in acc.items():
            out.setdefault(lv, {})[k] = float(np.mean(v))
    return out


std = {fs: metrics(std_root(*fs)) for fs in FS}
kin = {fs: metrics(f"{PR}/latent_kin_f{fs[0]}_seed{fs[1]}") for fs in FS}
print("# L3-kin 消融：人体运动学 vs 人体动力学（4 折 × 3 latent 种子，离线指标与线性探针）\n")
print("L3-kin = L3 去掉所有人体力和力矩信号（关节力矩、地面反力目标），保留人体运动学、步长和步态时序。其余设置与 L3 完全相同。\n")
# self-check
dev = []
for fs in FS:
    for lv in ("L0", "L1", "L2"):
        for k, v in std[fs].get(lv, {}).items():
            if k in kin[fs].get(lv, {}):
                dev.append(abs(kin[fs][lv][k] - v))
print(f"自检：L3-kin 目录中 L0–L2 的数值与标准结果的最大差异 = {max(dev) if dev else float('nan'):.4f}（应接近 0；比较了 {len(dev)} 个数值）\n")
print("## 离线指标\n")
print("| 指标 | L2 | L3-kin | L3 | L3-kin − L2（人体运动学） | L3 − L3-kin（人体动力学） |")
print("|---|---|---|---|---|---|")
keys = sorted({k for fs in FS for k in std[fs].get("L3", {})})
for k in keys:
    rows = [(std[fs]["L2"][k], kin[fs]["L3"][k], std[fs]["L3"][k]) for fs in FS
            if k in std[fs].get("L2", {}) and k in kin[fs].get("L3", {}) and k in std[fs].get("L3", {})]
    if not rows:
        continue
    a = np.array(rows)
    d1, d2 = a[:, 1] - a[:, 0], a[:, 2] - a[:, 1]
    print(f"| {k} | {a[:, 0].mean():.3f} | {a[:, 1].mean():.3f} | {a[:, 2].mean():.3f} | "
          f"{d1.mean():+.3f}（{int((d1 > 0).sum())}/{len(d1)}） | {d2.mean():+.3f}（{int((d2 > 0).sum())}/{len(d2)}） |")
print("\n## 线性探针（test 试次帧；R² 或 BA）\n")
ps = json.load(open(f"{H}/runs/probe_z_cv.json")) if os.path.exists(f"{H}/runs/probe_z_cv.json") else {}
pk = json.load(open(f"{H}/runs/probe_z_kin.json")) if os.path.exists(f"{H}/runs/probe_z_kin.json") else {}
NAME = {"stake": "下一桩位置 R²", "ttd": "距着地时间 R²", "disp": "0.5 s 骨盆位移 R²", "height": "骨盆高度 R²",
        "support": "支撑状态 BA", "task": "动作类别 BA"}
print("| 探针 | L2 | L3-kin | L3 | L3-kin − L2 | L3 − L3-kin |")
print("|---|---|---|---|---|---|")
for k, name in NAME.items():
    rows = [(ps[f"f{F}_s{S}_Z2"][k], pk[f"f{F}_s{S}_Z3K"][k], ps[f"f{F}_s{S}_Z3"][k]) for F, S in FS
            if f"f{F}_s{S}_Z2" in ps and f"f{F}_s{S}_Z3K" in pk and f"f{F}_s{S}_Z3" in ps]
    if not rows:
        continue
    a = np.array(rows)
    d1, d2 = a[:, 1] - a[:, 0], a[:, 2] - a[:, 1]
    print(f"| {name} | {a[:, 0].mean():.3f} | {a[:, 1].mean():.3f} | {a[:, 2].mean():.3f} | "
          f"{d1.mean():+.3f}（{int((d1 > 0).sum())}/{len(d1)}） | {d2.mean():+.3f}（{int((d2 > 0).sum())}/{len(d2)}） |")
print("\n括号内为 12 个编码器中差值为正的个数。")
