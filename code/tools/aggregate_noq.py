"""Factorisation ablation summary (tools/run_l3noq.sh): L3 vs L3-noQ (= L3 without executability supervision of the
z_quality branch) per fold x latent seed (12 encoders), offline 8.1 metrics and linear probes of z_skill; paired
difference L3 - L3-noQ (positive = the executability branch helps).  Self-check: Z0-Z2 rows equal the standard ones.
  python aggregate_noq.py > runs/l3noq_summary.md"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
H = "/root/gpufree-data/hnu_v2"
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "aggregate_human.py"), encoding="utf-8")
     .read().split("std = {fs")[0])  # metrics(), std_root(), FS, PR
std = {fs: metrics(std_root(*fs)) for fs in FS}
noq = {fs: metrics(f"{PR}/latent_noq_f{fs[0]}_seed{fs[1]}") for fs in FS}
print("# 因子化消融：去掉 z_quality 的可执行性监督（L3-noQ）对 L3 的影响（4 折 × 3 latent 种子）\n")
print("L3-noQ = L3 的训练中把质量相关的三个损失（质量回归与标签、标签排序、质量协方差）设为 0，其余完全相同。\n")
dev = [abs(noq[fs][lv][k] - std[fs][lv][k]) for fs in FS for lv in ("L0", "L1", "L2")
       for k in std[fs].get(lv, {}) if k in noq[fs].get(lv, {})]
print(f"自检：L0–L2 与标准结果的最大差异 = {max(dev) if dev else float('nan'):.4f}（比较 {len(dev)} 个数值）\n")
print("## 离线指标\n")
print("| 指标 | L0 | L3-noQ | L3 | L3 − L3-noQ（质量分支的作用） |")
print("|---|---|---|---|---|")
for k in sorted({k for fs in FS for k in std[fs].get("L3", {})}):
    rows = [(std[fs]["L0"].get(k, np.nan), noq[fs]["L3"][k], std[fs]["L3"][k]) for fs in FS
            if k in noq[fs].get("L3", {}) and k in std[fs].get("L3", {})]
    if rows:
        a = np.array(rows, float)
        d = a[:, 2] - a[:, 1]
        print(f"| {k} | {np.nanmean(a[:, 0]):.3f} | {a[:, 1].mean():.3f} | {a[:, 2].mean():.3f} | "
              f"{d.mean():+.3f}（{int((d > 0).sum())}/{len(d)}） |")
print("\n## z_skill 的线性探针\n")
ld = lambda p: json.load(open(p)) if os.path.exists(p) else {}  # noqa: E731
ps, pq = ld(f"{H}/runs/probe_z_cv.json"), ld(f"{H}/runs/probe_z_noq.json")
NAME = {"stake": "下一桩位置 R²", "ttd": "距着地时间 R²", "disp": "0.5 s 骨盆位移 R²", "height": "骨盆高度 R²",
        "support": "支撑状态 BA", "task": "动作类别 BA"}
print("| 探针 | L0 | L3-noQ | L3 | L3 − L3-noQ |")
print("|---|---|---|---|---|")
for k, name in NAME.items():
    rows = [(ps[f"f{F}_s{S}_Z0"][k], pq[f"f{F}_s{S}_Z3Q"][k], ps[f"f{F}_s{S}_Z3"][k]) for F, S in FS
            if f"f{F}_s{S}_Z0" in ps and f"f{F}_s{S}_Z3Q" in pq and f"f{F}_s{S}_Z3" in ps]
    if rows:
        a = np.array(rows)
        d = a[:, 2] - a[:, 1]
        print(f"| {name} | {a[:, 0].mean():.3f} | {a[:, 1].mean():.3f} | {a[:, 2].mean():.3f} | "
              f"{d.mean():+.3f}（{int((d > 0).sum())}/{len(d)}） |")
print("\n括号内为 12 个编码器中 L3 更高的个数。执行质量 AUROC 在 L3-noQ 中预期下降（质量头没有监督），用作开关是否生效的检查。")
