"""Cross-trial z pilot (tools/xz_l40.sh, xz_l40_plan4.sh): success v2 % of the new (v6g10x linear / v6g10xw stance-
aligned) and old (v6g10n) fold-1 seed-1 students on the split_v2 test clips, nominal, per condition, level and action;
balance beam also without contact; per-clip detail for the deployable conditions.
  python xz_table.py [results prefix=results_xz] [tags=n,x,xw]"""
import json
import os
import sys

import numpy as np

H = "/root/gpufree-data/hnu_v2/runs"
PRE = sys.argv[1] if len(sys.argv) > 1 else "results_xz"
WHO = {"n": "旧（v6g10n）", "x": "新：线性 50%（v6g10x）", "xw": "新：对齐 50%（v6g10xw，仅 L3）",
       "x25": "新：线性 25%（v6g10x25，L0/L3）", "xm": "新：混合 线性 25% + 对齐 25%（v6g10xm，L0/L3）"}
TAGS = (sys.argv[2] if len(sys.argv) > 2 else "n,x,xw").split(",")
SEED = sys.argv[3] if len(sys.argv) > 3 else "1"
COND = [("own", "自己的 z + 人的时钟（策略级）"), ("geo", "库 z 线性拉伸 + 人的时钟"),
        ("plan3", "库 z + 规划时钟（plan3，可部署）"), ("plan4", "库 z + 规划时钟 + 独木桥规划（plan4，可部署）"),
        ("geoplan3", "库 z 线性拉伸 + 规划时钟"), ("ownplan3", "自己的 z + 规划时钟")]
LV = ["Z0", "Z1", "Z2", "Z3"]
TK = [("CrouchedTraverse", "独木桥"), ("ThreeStepPoleRun", "三步桩"), ("FiveStepPoleRun", "五步桩")]


def rows(cond, tag, P):
    f = f"{H}/{PRE}_{cond}_s{SEED}_dr0/eval_student_{tag}_{P}.json"
    return [c for c in json.load(open(f))["clips"] if c["split"] == "test"] if os.path.exists(f) else None


def m(r, key="success_v2", task=None, clip=None):
    v = [c[key] for c in r if (task is None or c["task"] == task) and (clip is None or c["clip"] == clip)]
    return 100 * np.mean(v) if v else np.nan


f1 = lambda x: "–" if np.isnan(x) else f"{x:.1f}"  # noqa: E731
print(f"# 跨试次 z 增广试点（第 1 折种子 {SEED}，L40，理想条件，16 个 test 片段 = 没见过的布局，每片段 32 次）\n")
for cond, cn in COND:
    print(f"## {cn}\n")
    print("| student | L0 | L1 | L2 | L3 | 平均 | 独木桥 | 三步桩 | 五步桩 | 独木桥无碰撞成功 | 不摔且完成 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for tag in TAGS:
        R = {P: rows(cond, tag, P) for P in LV}
        if not any(R.values()):
            continue
        lv = [m(R[P]) if R[P] else np.nan for P in LV]
        have = [R[P] for P in LV if R[P]]
        tk = [np.mean([m(r, task=t) for r in have]) for t, _ in TK]
        nc = np.mean([m(r, "success_no_contact", "CrouchedTraverse") for r in have])
        sv = np.mean([m(r, "success") for r in have])
        print(f"| {WHO[tag]} | " + " | ".join(f1(x) for x in lv) + f" | {f1(np.nanmean(lv))} | " + " | ".join(f1(x) for x in tk)
              + f" | {f1(nc)} | {f1(sv)} |")
    print()
print("## 每个 test 片段（L3，%）：旧 / 新线性 / 新对齐\n")
cl = sorted({c["clip"] for c in rows("own", "n", "Z3")})
print("| 片段 | 自己的 z | plan4（可部署） | 库 z 线性 + 规划时钟 |")
print("|---|---|---|---|")
for c in cl:
    cells = []
    for cond in ("own", "plan4", "geoplan3"):
        cells.append(" / ".join(f"{m(r, clip=c):.0f}" if (r := rows(cond, t, "Z3")) else "–" for t in TAGS))
    print(f"| {c} | " + " | ".join(cells) + " |")
