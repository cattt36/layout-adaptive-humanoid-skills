"""Table of the deployment tolerance sweeps (tools/deploy_sweeps.sh): success v2 % on the fold-1 TEST clips (all clips
in brackets) per perturbation, L3 vs L0, per action.
  python sweep_table.py [EXP=v6g10n_f1_s1]"""
import json
import os
import sys

import numpy as np

H = "/root/gpufree-data/hnu_v2"
EXP = sys.argv[1] if len(sys.argv) > 1 else "v6g10n_f1_s1"
# (tools/deploy_v2_4090.sh): SPLIT = split file of the students' fold; RES_C = stem of the experiment-C dirs
TEST = set(json.load(open(os.environ.get("SPLIT", f"{H}/data/clip_split_trial/split_v2.json")))["test"])
TASKS = [("CrouchedTraverse", "独木桥"), ("ThreeStepPoleRun", "三步桩"), ("FiveStepPoleRun", "五步桩")]
GROUPS = [
    ("基准（0–20 ms 延迟）", ["base"]),
    ("固定 20 ms 延迟", ["lat20"]),
    ("新增（保守范围）：桩距整体缩放（三步桩/五步桩，重新规划）", ["spacing0.90", "spacing0.95", "spacing1.05", "spacing1.10"]),
    ("新增（保守范围）：起点到第一根桩的距离 m（三步桩/五步桩，重新规划）", ["start-0.10", "start-0.05", "start+0.05", "start+0.10"]),
    ("新增（保守范围）：桥柱间距变化 m（独木桥）", ["post-0.05", "post-0.03", "post+0.03", "post+0.05"]),
    ("感知固定偏差", ["perc_bias0.02", "perc_bias0.05", "perc_bias0.08", "perc_bias0.12"]),
    ("感知逐帧噪声", ["perc_noise0.02", "perc_noise0.05", "perc_noise0.10"]),
    ("摆放：整体平移", ["shift0.02", "shift0.05", "shift0.08", "shift0.12"]),
    ("摆放：每个桩独立偏移（三步桩/五步桩）", ["stake_jitter0.02", "stake_jitter0.05", "stake_jitter0.08"]),
    ("搭建：横梁高度（独木桥）", ["bridge_dz-0.05", "bridge_dz-0.03", "bridge_dz0.03", "bridge_dz0.05"]),
    ("起始朝向误差（度）", ["yaw3", "yaw5", "yaw10"]),
    ("里程计漂移（无外部定位，补充）", ["drift0.01", "drift0.02", "drift0.05", "drift0.10"]),
]


def load(L, name):
    f = f"{H}/deploy_exports/{EXP}/{L}/sweeps/{name}.json"
    return json.load(open(f))["rows"] if os.path.exists(f) else None


def rate(rows, test=True, task=None):
    v = [r["success_v2"] for r in rows if (not test or r["clip"] in TEST) and (task is None or r["task"] == task)]
    return 100 * np.mean(v) if v else float("nan")


print(f"# 部署容差检查（{EXP}，CPU MuJoCo sim2sim，0–20 ms 延迟，成功标准 v2，每片段 8 次）\n")
print("每格：test 片段成功率 %（全部片段，括号内）。L3 / L0。\n")
for gname, names in GROUPS:
    print(f"## {gname}\n")
    print("| 设置 | L3 合计 | L0 合计 | " + " | ".join(f"L3 {tn} | L0 {tn}" for _, tn in TASKS) + " |")
    print("|---|---|---|" + "---|---|" * len(TASKS))
    for nm in names:
        r3, r0 = load("L3", nm), load("L0", nm)
        if r3 is None and r0 is None:
            continue
        cell = lambda r, t=None: "–" if r is None else f"{rate(r, True, t):.0f}（{rate(r, False, t):.0f}）"  # noqa: E731
        label = nm.replace("perc_bias", "").replace("perc_noise", "").replace("shift", "").replace("stake_jitter", "") \
            .replace("bridge_dz", "").replace("yaw", "").replace("drift", "").replace("base", "–").replace("lat20", "20 ms") \
            .replace("spacing", "×").replace("start", "").replace("post", "")
        print(f"| {label} | {cell(r3)} | {cell(r0)} | " + " | ".join(f"{cell(r3, t)} | {cell(r0, t)}" for t, _ in TASKS) + " |")
    print()
RC = os.environ.get("RES_C")
if RC:  # experiment C: mocap dropout (mjlab evaluation, 32 rollouts per clip, planner chain)
    print("## 动捕中断（实验 C；障碍物特征冻结的时长 s；mjlab 评估，每片段 32 次）\n")
    print("| 冻结时长 | L3 合计 | L0 合计 | " + " | ".join(f"L3 {tn} | L0 {tn}" for _, tn in TASKS) + " |")
    print("|---|---|---|" + "---|---|" * len(TASKS))
    for dur in ("0", "0.25", "0.5", "1.0"):
        rr = {}
        for L, P in (("L3", "Z3"), ("L0", "Z0")):
            f = f"{H}/runs/{RC}{dur}_s1_dr0/eval_student_{P}.json"
            rr[L] = [dict(clip=c["clip"], task=c["task"], success_v2=c["success_v2"]) for c in json.load(open(f))["clips"]] \
                if os.path.exists(f) else None
        cell = lambda r, t=None: "–" if r is None else f"{rate(r, True, t):.0f}（{rate(r, False, t):.0f}）"  # noqa: E731
        print(f"| {dur} | {cell(rr['L3'])} | {cell(rr['L0'])} | " + " | ".join(f"{cell(rr['L3'], t)} | {cell(rr['L0'], t)}" for t, _ in TASKS) + " |")
    print()
