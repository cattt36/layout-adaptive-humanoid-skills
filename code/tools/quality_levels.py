"""Where does the biomechanical supervision matter in closed loop? (user request; exploratory)
Motion-quality metrics of the students per level, with the pre-registered estimator of tools/closed_loop_ci.py
(unit = (test trial, RL seed); per-trial mean over seeds, then mean over trials; paired level differences; 95 % CI by
bootstrap stratified by fold).  Metrics per clip from eval_student.py (means over ALL rollouts of the clip, failed ones
included until termination -- confounded with success; the paired design controls the clip):
  contact_mismatch       share of steps where the robot's foot contacts disagree with the human reference (gait events)
  body_err               body-position error to the human reference motion (m)
  peak_force_bw          peak vertical foot force (body weights)
  max_loading_rate_bw_s  maximal loading rate (body weights / s)
  torque_sat_frac        share of steps with a saturated joint torque
  foot_slip_m_s          foot slip speed during contact (m/s)
  success_v2             for reference
  python quality_levels.py <results prefix, e.g. results_v6g10n | results_v6g10xm | results_v6g10xmB> [seeds=1,2,3] [reps=4000]"""
import json
import os
import sys

import numpy as np

H = "/root/gpufree-data/hnu_v2/runs"
PRE = sys.argv[1] if len(sys.argv) > 1 else "results_v6g10n"
SEEDS = [int(s) for s in (sys.argv[2] if len(sys.argv) > 2 else "1,2,3").split(",")]
REPS = int(sys.argv[3]) if len(sys.argv) > 3 else 4000
LV = ["Z0", "Z1", "Z2", "Z3"]
MET = [("contact_mismatch", "接触时序不一致比例", 100, "%"), ("body_err", "身体位置误差", 100, "cm"),
       ("peak_force_bw", "峰值足底力", 1, "BW"), ("max_loading_rate_bw_s", "最大负载率", 1, "BW/s"),
       ("torque_sat_frac", "力矩饱和比例", 100, "%"), ("foot_slip_m_s", "足底滑移速度", 100, "cm/s"),
       ("success_v2", "成功率（参考）", 100, "%")]
TASKS = [(None, "合计"), ("CrouchedTraverse", "独木桥"), ("ThreeStepPoleRun", "三步桩"), ("FiveStepPoleRun", "五步桩")]
rng = np.random.default_rng()


def load(F, s, dr, P, x=""):
    p = f"{H}/{PRE}{'' if F == 1 else f'_f{F}'}_s{s}_dr{dr}/eval_student_{P}{x}.json"
    if not os.path.exists(p):
        return None
    return {c["clip"]: c for c in json.load(open(p))["clips"] if c["split"] == "test"}


def data(dr, task, key):
    out = {}
    for F in (1, 2, 3, 4):
        seeds = [s for s in SEEDS if all(load(F, s, dr, P) is not None for P in LV)]
        if not seeds:
            continue
        ref = load(F, seeds[0], dr, "Z0")
        trials = sorted(c for c, r in ref.items() if task is None or r["task"] == task)
        if not trials:
            continue
        A = np.array([[[load(F, s, dr, P)[c][key] for s in seeds] for c in trials] for P in LV], float)
        out[F] = (trials, seeds, A)
    return out


def estimate(d, idx=None):
    v = []
    for F, (t, s, A) in d.items():
        ti, si = (np.arange(len(t)), np.arange(len(s))) if idx is None else idx[F]
        v.append(np.nanmean(A[:, ti][:, :, si], 2))
    return np.nanmean(np.concatenate(v, 1), 1)


def boot(d):
    return np.array([estimate(d, {F: (rng.integers(0, len(t), len(t)), rng.integers(0, len(s), len(s)))
                                  for F, (t, s, _) in d.items()}) for _ in range(REPS)])


units = {F: s for F, (t, s, _) in data("0", None, "success_v2").items()}
print(f"# 闭环动作质量 × 层级（{PRE}，测试试次；预注册估计方法，按折分层 bootstrap {REPS} 次）\n")
print("已有的（折：种子）：" + "；".join(f"第 {F} 折：{units[F]}" for F in sorted(units)) + "\n")
print("指标为每个片段全部回合的平均（含失败回合，至终止为止），与成功率有混杂；配对差值控制了片段。"
      "数值越小越好（成功率除外）。★ = 95% CI 不含 0。\n")
for dr, cn in (("0", "理想"), ("1", "动力学随机化 + 延迟")):
    print(f"## {cn}\n")
    for t, tn in TASKS:
        print(f"### {tn}\n")
        print("| 指标 | L0 | L1 | L2 | L3 | L3 − L0 [95% CI] | L2 − L0 | L1 − L0 |")
        print("|---|---|---|---|---|---|---|---|")
        for key, name, sc, unit in MET:
            d = data(dr, t, key)
            if not d:
                continue
            e, B = estimate(d), boot(d)

            def diff(k):
                lo, hi = np.nanpercentile(B[:, k] - B[:, 0], [2.5, 97.5])
                star = " ★" if lo > 0 or hi < 0 else ""
                return f"{sc * (e[k] - e[0]):+.2f} [{sc * lo:+.2f}, {sc * hi:+.2f}]{star}"
            print(f"| {name}（{unit}） | " + " | ".join(f"{sc * x:.2f}" for x in e) + f" | {diff(3)} | {diff(2)} | {diff(1)} |")
        print()
