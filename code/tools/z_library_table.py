"""Table of the skill-library experiment (tools/z_library.sh): success v2 % per level and z source on the fold-1 test
clips (held-out layouts), nominal condition; references: own z (results_v6g10n_C0), arbitrary same-action clip
(z ablation "swap"), random library clip.  Plus layout distances of the chosen library clips (data/z_lib_f1/choice.json).
  python z_library_table.py [seed=1]"""
import json
import os
import sys

import numpy as np

H = "/root/gpufree-data/hnu_v2"
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 1
LV = ["Z0", "Z1", "Z2", "Z3", "Z4"]
VAR = [("自己的 z（上限）", f"results_v6g10n_C0_s{SEED}_dr0"),
       ("几何最近的库片段 + 踩桩时段对齐", f"results_v6g10n_zlib_geowarp_s{SEED}_dr0"),
       ("几何最近的库片段（按时长对齐）", f"results_v6g10n_zlib_geo_s{SEED}_dr0"),
       ("随机的库片段（同动作）", f"results_v6g10n_zlib_random_s{SEED}_dr0"),
       ("同动作另一片段（z 消融 swap）", f"results_v6g10n_zabl_swap_s{SEED}_dr0")]
TASKS = [("CrouchedTraverse", "独木桥"), ("ThreeStepPoleRun", "三步桩"), ("FiveStepPoleRun", "五步桩")]
choice = json.load(open(f"{H}/data/z_lib_f1/choice.json"))


def rows(d, P):
    f = f"{H}/runs/{d}/eval_student_{P}.json"
    return [c for c in json.load(open(f))["clips"] if c["split"] == "test"] if os.path.exists(f) else None


print(f"# 技能库 + 新布局（第 1 折种子 {SEED} 学生，4090，理想条件，新标准，16 个 test 片段 = 没见过的布局，每片段 32 次）\n")
print("库 = 第 1 折的 51 个训练片段；几何距离在各片段的起始坐标系中计算。动作时钟（时长、踩桩时刻）仍来自 test 试次，只有 z 来自库。\n")
print("## 成功率 %\n")
print("| z 的来源 | " + " | ".join(f"L{P[1]}" for P in LV) + " | 平均 |")
print("|---|" + "---|" * (len(LV) + 1))
for name, d in VAR:
    v = []
    for P in LV:
        r = rows(d, P)
        v.append(np.nan if r is None else 100 * np.mean([c["success_v2"] for c in r]))
    print(f"| {name} | " + " | ".join("–" if np.isnan(x) else f"{x:.1f}" for x in v) + f" | {np.nanmean(v):.1f} |")
print("\n## 分动作（L0–L4 平均，%）\n")
print("| z 的来源 | " + " | ".join(tn for _, tn in TASKS) + " |")
print("|---|---|---|---|")
for name, d in VAR:
    cells = []
    for t, _ in TASKS:
        v = [np.mean([c["success_v2"] for c in r if c["task"] == t]) for P in LV if (r := rows(d, P)) is not None]
        cells.append(f"{100 * np.mean(v):.1f}" if v else "–")
    print(f"| {name} | " + " | ".join(cells) + " |")
print("\n## 每个 test 片段：选中的库片段与几何距离；成功率（L0–L4 平均，%）\n")
print("| test 片段 | 几何最近的库片段 | 距离 m | 随机库片段 | 距离 m | 库内距离中位数 m | 自己的 z | 几何最近 + 对齐 | 几何最近 | 随机 |")
print("|---|---|---|---|---|---|---|---|---|---|")
mean_of = lambda d, n: np.mean([next(c["success_v2"] for c in r if c["clip"] == n) for P in LV if (r := rows(d, P))])  # noqa: E731
for n, c in sorted(choice.items()):
    if c["split"] != "test":
        continue
    print(f"| {n} | {c['geo']} | {c['geo_dist_m']:.2f} | {c['random']} | {c['random_dist_m']:.2f} | {c['median_lib_dist_m']:.2f} | "
          f"{100 * mean_of(VAR[0][1], n):.0f} | {100 * mean_of(VAR[1][1], n):.0f} | {100 * mean_of(VAR[2][1], n):.0f} | "
          f"{100 * mean_of(VAR[3][1], n):.0f} |")
