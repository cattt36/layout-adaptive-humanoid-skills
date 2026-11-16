"""Table of the layout-adaptation experiment (tools/z_library.sh + tools/z_plan.sh): success v2 % on the fold-1 test
clips (held-out layouts), nominal condition, L0-L3, per z source / clock.
  python z_plan_table.py [seed=1]"""
import json
import os
import sys

import numpy as np

H = "/root/gpufree-data/hnu_v2"
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 1
LV = ["Z0", "Z1", "Z2", "Z3"]
VAR = [("自己的 z + 人的时钟（上限）", f"results_v6g10n_C0_s{SEED}_dr0"),
       ("自己的 z + 规划时钟（对照）", f"results_v6g10n_zlib_ownclock_s{SEED}_dr0"),
       ("库 z + 对齐到人的踩桩时刻（geowarp，用了 test 时序）", f"results_v6g10n_zlib_geowarp_s{SEED}_dr0"),
       ("库 z + 按几何规划的时钟（plan，可部署）", f"results_v6g10n_zlib_plan_s{SEED}_dr0"),
       ("库 z + 规划时钟，起步等待 ≤ 0.5 s（plan2，可部署）", f"results_v6g10n_zlib_plan2_s{SEED}_dr0"),
       ("自己的 z + plan2 时钟（对照）", f"results_v6g10n_zlib_ownclock2_s{SEED}_dr0"),
       ("库 z + 规划时钟，首桩时刻按训练集回归（plan3，可部署）", f"results_v6g10n_zlib_plan3_s{SEED}_dr0"),
       ("自己的 z + plan3 时钟（对照）", f"results_v6g10n_zlib_ownclock3_s{SEED}_dr0"),
       ("库 z，按时长线性拉伸（geo）", f"results_v6g10n_zlib_geo_s{SEED}_dr0"),
       ("随机库片段", f"results_v6g10n_zlib_random_s{SEED}_dr0")]
TASKS = [("CrouchedTraverse", "独木桥"), ("ThreeStepPoleRun", "三步桩"), ("FiveStepPoleRun", "五步桩")]
timing = json.load(open(f"{H}/data/z_lib_f1/plan_timing.json"))
t3f = f"{H}/data/z_lib_f1/plan3_timing.json"
timing3 = json.load(open(t3f)) if os.path.exists(t3f) else {}


def rows(d, P):
    f = f"{H}/runs/{d}/eval_student_{P}.json"
    return [c for c in json.load(open(f))["clips"] if c["split"] == "test"] if os.path.exists(f) else None


def rate(r, t=None, n=None):
    v = [c["success_v2"] for c in r if (t is None or c["task"] == t) and (n is None or c["clip"] == n)]
    return 100 * np.mean(v) if v else np.nan


print(f"# 按布局调整技能（第 1 折种子 {SEED}，理想条件，16 个 test 片段 = 没见过的布局，每片段 32 次）\n")
print("| z 的来源 / 时钟 | L0 | L1 | L2 | L3 | 平均 | 独木桥 | 三步桩 | 五步桩 |")
print("|---|---|---|---|---|---|---|---|---|")
for name, d in VAR:
    R = {P: rows(d, P) for P in LV}
    lv = [rate(R[P]) if R[P] else np.nan for P in LV]
    tk = [np.nanmean([rate(R[P], t) for P in LV if R[P]]) if any(R.values()) else np.nan for t, _ in TASKS]
    f = lambda x: "–" if np.isnan(x) else f"{x:.1f}"  # noqa: E731
    print(f"| {name} | " + " | ".join(f(x) for x in lv) + f" | {f(np.nanmean(lv))} | " + " | ".join(f(x) for x in tk) + " |")
print("\n## 三步桩/五步桩 test 片段：规划时钟与人的踩桩时刻（s）及成功率（L0–L3 平均，%）\n")
print("| 片段 | 库片段 | 误差 s（plan / plan3） | 自己 z | geowarp | plan | plan3 | 自己 z + plan 时钟 | 自己 z + plan3 时钟 |")
print("|---|---|---|---|---|---|---|---|---|")
for n, v in sorted(timing.items()):
    if v["split"] != "test":
        continue
    cell = []
    for d in (VAR[0][1], VAR[2][1], VAR[3][1], VAR[6][1], VAR[1][1], VAR[7][1]):
        x = [rate(r, n=n) for P in LV if (r := rows(d, P))]
        cell.append(f"{np.mean(x):.0f}" if x else "–")
    e3 = f"{timing3[n]['mean_abs_err_s']:.2f}" if n in timing3 else "–"
    print(f"| {n} | {v['lib']} | {v['mean_abs_err_s']:.2f} / {e3} | " + " | ".join(cell) + " |")
