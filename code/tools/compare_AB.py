"""Version A (selected deployment students) vs A3 (extra round, no localisation error) vs B (localisation-error
robust): mjlab evaluation and sim2sim sweeps, per action and level.  Output: markdown (run_v6g10sB.sh)."""
import json
import os

H = "/root/gpufree-data/hnu_v2"
A = "v6g10s"
TASKS = ["CrouchedTraverse", "ThreeStepPoleRun", "FiveStepPoleRun"]
CN = {"CrouchedTraverse": "独木桥", "ThreeStepPoleRun": "三步桩", "FiveStepPoleRun": "五步桩"}
sel = json.load(open(f"{H}/runs/results_{A}_selection.json"))


def ev(var, P, dr, x):
    s = sel[P]["selected_seed"]
    d = f"{H}/runs/results_{A}_s{s}_dr{dr}" if var == "A" else f"{H}/runs/results_{A}{var}_dr{dr}"
    f = f"{d}/eval_student_{P}{x}.json"
    return json.load(open(f))["summary"] if os.path.exists(f) else None


def exp_dir(var, P):
    s = sel[P]["selected_seed"]
    return f"{H}/deploy_exports/{A}/L{P[1]}_s{s}" if var == "A" else f"{H}/deploy_exports/{A}{var}/L{P[1]}"


def s2s(var, P, rel):
    f = os.path.join(exp_dir(var, P), rel)
    return json.load(open(f))["summary"] if os.path.exists(f) else None


def pct(summ, T, key="success"):
    return "-" if summ is None or T not in summ else f"{100 * summ[T][key]:.1f}"


print("# 版本 A / A3 / B 对比\n")
print("A = 当前部署学生（选定种子）；A3 = A 再训练 2000 步、不加定位误差（对照组）；"
      "B = A 再训练 2000 步、加入定位误差（初始偏差 ≤ 5 cm，漂移 ≤ 2%）。成功率 %。\n")
print("## 1. 训练仿真评估（mjlab；理想 / 理想+推扰 / 鲁棒 / 鲁棒+推扰）\n")
for T in TASKS:
    print(f"\n**{CN[T]}**\n\n| 层级 | 版本 | 理想 | 理想+推扰 | 鲁棒 | 鲁棒+推扰 |\n|---|---|---|---|---|---|")
    for P in ["Z0", "Z1", "Z2", "Z3", "Z4"]:
        for var in ("A", "A3", "B"):
            cells = []
            for dr, x in (("0", ""), ("0", "_push"), ("1", ""), ("1", "_push")):
                e = ev(var, P, dr, x)
                cells.append("-" if e is None else f"{100 * e[f'{T}/all']['success']:.1f}")
            print(f"| L{P[1]} | {var} | " + " | ".join(cells) + " |")
COLS = [("基准", "perception/bias0.json"), ("看偏2cm", "perception/bias0.02.json"), ("看偏5cm", "perception/bias0.05.json"),
        ("看偏8cm", "perception/bias0.08.json"), ("摆偏12cm", "placement/shift0.12.json"),
        ("单桩±5cm", "placement/jitter0.05.json"), ("单桩±8cm", "placement/jitter0.08.json"),
        ("漂移1%", "drift/r0.01.json"), ("漂移2%", "drift/r0.02.json"), ("起点5cm+漂移2%", "drift/b0.05_r0.02.json")]
print("\n## 2. sim2sim（CPU MuJoCo，0–20 ms 延迟，摩擦 / 质心 / 编码器随机化）\n")
print("看偏 = 机器人感知的障碍物位置有固定偏差；摆偏 = 障碍物实际位置偏离参考布局（感知准确）；"
      "漂移 = 不用外部定位、靠里程计推算自身位置。\n")
for T in TASKS:
    cols = [c for c in COLS if not (T == "CrouchedTraverse" and "单桩" in c[0])]
    print(f"\n**{CN[T]}**\n\n| 层级 | 版本 | " + " | ".join(c[0] for c in cols) + " |\n|---|---|" + "---|" * len(cols))
    for P in ["Z0", "Z1", "Z2", "Z3", "Z4"]:
        for var in ("A", "A3", "B"):
            print(f"| L{P[1]} | {var} | " + " | ".join(pct(s2s(var, P, rel), T) for _, rel in cols) + " |")
print("\n## 3. 各层级部署种子\n")
print(" / ".join(f"L{P[1]}: 种子 {sel[P]['selected_seed']}" for P in ["Z0", "Z1", "Z2", "Z3", "Z4"]))
