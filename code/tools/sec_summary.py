"""Secondary experiments on the 4090 server, fold 1 (test clips of split_v2): C (mocap dropout) and B (semantic
consistency of successful rollouts).  Mean over test clips (each clip 32 rollouts), then over the given RL seeds.
python sec_summary.py [seeds=1]"""
import json
import os
import sys

import numpy as np

R = "/root/gpufree-data/hnu_v2/runs"
SEEDS = (sys.argv[1] if len(sys.argv) > 1 else "1").split(",")
LV = ["Z0", "Z1", "Z2", "Z3", "Z4"]
TASKS = ["CrouchedTraverse", "ThreeStepPoleRun", "FiveStepPoleRun"]
CN = {"CrouchedTraverse": "独木桥", "ThreeStepPoleRun": "三步桩", "FiveStepPoleRun": "五步桩"}


def load(d, P):
    f = f"{R}/{d}/eval_student_{P}.json"
    return [c for c in json.load(open(f))["clips"] if c["split"] == "test"] if os.path.exists(f) else None


print(f"## 实验 C：障碍物特征冻结（动捕短时丢失），第 1 折 test，理想环境，成功率 %（新标准），种子 {','.join(SEEDS)}\n")
DUR = ["0", "0.25", "0.5", "1.0"]
for T in TASKS + ["ALL"]:
    print(f"\n**{CN.get(T, '三个动作合计')}**\n")
    print("| 冻结时长 | " + " | ".join(f"L{P[1]}" for P in LV) + " |")
    print("|---" * 6 + "|")
    for d in DUR:
        cells = []
        for P in LV:
            v = [c["success_v2"] for s in SEEDS for c in (load(f"results_v6g10n_C{d}_s{s}_dr0", P) or []) if T == "ALL" or c["task"] == T]
            cells.append(f"{100 * np.mean(v):.1f}" if v else "-")
        print(f"| {d} s | " + " | ".join(cells) + " |")
    # drop from 0 to 1.0 s
    cells = []
    for P in LV:
        a = [c["success_v2"] for s in SEEDS for c in (load(f"results_v6g10n_C0_s{s}_dr0", P) or []) if T == "ALL" or c["task"] == T]
        b = [c["success_v2"] for s in SEEDS for c in (load(f"results_v6g10n_C1.0_s{s}_dr0", P) or []) if T == "ALL" or c["task"] == T]
        cells.append(f"{100 * (np.mean(b) - np.mean(a)):+.1f}" if a and b else "-")
    print("| 1.0 s 相对 0 s 的变化 | " + " | ".join(cells) + " |")

print(f"\n## 实验 B：执行的语义一致性（只统计成功的试跑），第 1 折 test，理想环境，种子 {','.join(SEEDS)}\n")
print("| 层级 | 成功试跑数 | 着地序列一致率 % | 节奏偏差（中位数，s） | 参考着地未出现 % | 落桩脚正确率 %（接触判定） | 落桩脚正确率 %（旧规则） |")
print("|---|---|---|---|---|---|---|")
for P in LV:
    rows = [c for s in SEEDS for c in (load(f"results_v6g10n_B_s{s}_dr0", P) or [])]
    if not rows:
        continue
    def w(k, scale=1.0):  # mean over clips weighted by the number of successful rollouts
        v = [(c.get(f"sem_{k}"), c.get("sem_n", 0)) for c in rows if c.get(f"sem_{k}") is not None and c.get("sem_n", 0) > 0]
        return f"{scale * np.average([a for a, _ in v], weights=[n for _, n in v]):.{1 if scale == 100 else 3}f}" if v else "-"
    n = sum(c.get("sem_n", 0) for c in rows)
    print(f"| L{P[1]} | {n} | {w('contact_agree', 100)} | {w('tempo_dev_s')} | {w('td_miss', 100)} | {w('stakes_foot_ok', 100)} | {w('stakes_ok', 100)} |")
