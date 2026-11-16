"""Decomposition of the human-data contribution (tools/run_l3kin.sh, tools/run_l3rom.sh): per fold x latent seed (12
encoders) offline 8.1 metrics and linear probes of
  L2      : robot evidence only
  L3-rom  : + human joint ranges of motion, step length, obstacle relation (no gait timing, no forces / moments)
  L3-kin  : + human gait timing (contact, touchdown / lift-off times from the force plates, segment duration)
  L3      : + human dynamics (OpenSim joint moments, ground-reaction targets)
Paired differences: L3-rom - L2 (human spatial kinematics), L3-kin - L3-rom (human gait timing), L3 - L3-kin (human
dynamics).  Self-check: Z0-Z2 rows of the ablation paper roots equal the standard ones.
  python aggregate_human.py > runs/l3rom_summary.md"""
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
    o = f"{root}/Results/8.1_Offline_Representation/outputs"
    out = {}
    f = f"{o}/offline_retrieval_macro_by_level.csv"
    if os.path.exists(f):
        for r in csv.DictReader(open(f, encoding="utf-8-sig")):
            for k, col in (("R@1", "task_macro_r_at_1"), ("MRR", "task_macro_mrr")):
                out.setdefault(r["supervision_level"], {})[f"检索 {r['protocol']} {k}"] = float(r[col])
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
rom = {fs: metrics(f"{PR}/latent_rom_f{fs[0]}_seed{fs[1]}") for fs in FS}
print("# 人体数据贡献的拆分：空间运动学 / 步态时序 / 动力学（4 折 × 3 latent 种子）\n")
print("- L3-rom = L2 + 人体关节活动范围、步长、与障碍物的空间关系（没有步态时序，没有力和力矩）")
print("- L3-kin = L3-rom + 人体步态时序（力台确定的触地、离地时刻、接触状态，片段时长）")
print("- L3 = L3-kin + 人体动力学（OpenSim 关节力矩、地面反力目标）\n")
dev = [abs(r[fs][lv][k] - std[fs][lv][k]) for r in (kin, rom) for fs in FS for lv in ("L0", "L1", "L2")
       for k in std[fs].get(lv, {}) if k in r[fs].get(lv, {})]
print(f"自检：消融目录中 L0–L2 与标准结果的最大差异 = {max(dev) if dev else float('nan'):.4f}（比较 {len(dev)} 个数值）\n")


def row(name, a):
    d1, d2, d3 = a[:, 1] - a[:, 0], a[:, 2] - a[:, 1], a[:, 3] - a[:, 2]
    c = lambda d: f"{d.mean():+.3f}（{int((d > 0).sum())}/{len(d)}）"  # noqa: E731
    return (f"| {name} | {a[:, 0].mean():.3f} | {a[:, 1].mean():.3f} | {a[:, 2].mean():.3f} | {a[:, 3].mean():.3f} | "
            f"{c(d1)} | {c(d2)} | {c(d3)} |")


HDR = ("| 指标 | L2 | L3-rom | L3-kin | L3 | 空间运动学（L3-rom − L2） | 步态时序（L3-kin − L3-rom） | 动力学（L3 − L3-kin） |\n"
       "|---|---|---|---|---|---|---|---|")
print("## 离线指标\n")
print(HDR)
for k in sorted({k for fs in FS for k in std[fs].get("L3", {})}):
    rows = [(std[fs]["L2"][k], rom[fs]["L3"][k], kin[fs]["L3"][k], std[fs]["L3"][k]) for fs in FS
            if all(k in d[fs].get(lv, {}) for d, lv in ((std, "L2"), (rom, "L3"), (kin, "L3"), (std, "L3")))]
    if rows:
        print(row(k, np.array(rows)))
print("\n## 线性探针\n")
ld = lambda p: json.load(open(p)) if os.path.exists(p) else {}  # noqa: E731
ps, pk, pr = ld(f"{H}/runs/probe_z_cv.json"), ld(f"{H}/runs/probe_z_kin.json"), ld(f"{H}/runs/probe_z_rom.json")
NAME = {"stake": "下一桩位置 R²", "ttd": "距着地时间 R²", "disp": "0.5 s 骨盆位移 R²", "height": "骨盆高度 R²",
        "support": "支撑状态 BA", "task": "动作类别 BA"}
print(HDR)
for k, name in NAME.items():
    rows = [(ps[f"f{F}_s{S}_Z2"][k], pr[f"f{F}_s{S}_Z3R"][k], pk[f"f{F}_s{S}_Z3K"][k], ps[f"f{F}_s{S}_Z3"][k]) for F, S in FS
            if f"f{F}_s{S}_Z2" in ps and f"f{F}_s{S}_Z3R" in pr and f"f{F}_s{S}_Z3K" in pk and f"f{F}_s{S}_Z3" in ps]
    if rows:
        print(row(name, np.array(rows)))
print("\n括号内为 12 个编码器中差值为正的个数。")
