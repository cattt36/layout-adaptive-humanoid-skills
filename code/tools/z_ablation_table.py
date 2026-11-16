"""Table of the z-usage ablation (tools/z_ablation.sh): success v2 % per level and z variant, fold-1 test clips (and
train clips), nominal condition; reference = original z (results_v6g10n_C0_s<seed>_dr0, same server and protocol).
  python z_ablation_table.py [seed=1]"""
import json
import os
import sys

import numpy as np

H = "/root/gpufree-data/hnu_v2"
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 1
LV = ["Z0", "Z1", "Z2", "Z3", "Z4"]
VAR = [("original", f"results_v6g10n_C0_s{SEED}_dr0"), ("mean", f"results_v6g10n_zabl_mean_s{SEED}_dr0"),
       ("freeze", f"results_v6g10n_zabl_freeze_s{SEED}_dr0"), ("swap", f"results_v6g10n_zabl_swap_s{SEED}_dr0")]
NAME = {"original": "原始 z", "mean": "均值 z（无片段信息）", "freeze": "冻结在第 1 帧（无时间信息）",
        "swap": "同动作另一片段的 z（时间对齐）"}
TASKS = [("CrouchedTraverse", "独木桥"), ("ThreeStepPoleRun", "三步桩"), ("FiveStepPoleRun", "五步桩")]


def load(d, P):
    f = f"{H}/runs/{d}/eval_student_{P}.json"
    return json.load(open(f))["clips"] if os.path.exists(f) else None


print(f"# z 消融：学生是否依赖 z_skill（第 1 折种子 {SEED} 学生，4090，理想条件，新标准，每片段 32 次）\n")
for sp, spn in (("test", "test 片段（16 个）"), ("train", "训练片段（51 个）")):
    print(f"## {spn}：成功率 %（括号内为相对原始 z 的变化）\n")
    print("| z 的处理 | " + " | ".join(f"L{P[1]}" for P in LV) + " |")
    print("|---|" + "---|" * len(LV))
    base = {}
    for v, d in VAR:
        cells = []
        for P in LV:
            r = load(d, P)
            if r is None:
                cells.append("–")
                continue
            s = 100 * np.mean([c["success_v2"] for c in r if c["split"] == sp])
            if v == "original":
                base[P] = s
                cells.append(f"{s:.1f}")
            else:
                cells.append(f"{s:.1f} ({s - base.get(P, np.nan):+.1f})")
        print(f"| {NAME[v]} | " + " | ".join(cells) + " |")
    print()
print("## test 片段分动作：成功率 %（L0 / L4）\n")
print("| z 的处理 | " + " | ".join(tn for _, tn in TASKS) + " |")
print("|---|---|---|---|")
for v, d in VAR:
    cells = []
    for t, _ in TASKS:
        vals = []
        for P in ("Z0", "Z4"):
            r = load(d, P)
            vals.append("–" if r is None else
                        f"{100 * np.mean([c['success_v2'] for c in r if c['split'] == 'test' and c['task'] == t]):.0f}")
        cells.append(" / ".join(vals))
    print(f"| {NAME[v]} | " + " | ".join(cells) + " |")
