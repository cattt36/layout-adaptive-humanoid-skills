"""Q_exec as a runtime fall monitor (EXPLORATORY item 5; , user request): early-warning analysis of the
closed-loop Q_exec traces written by tools/qexec_heldout.sh (results_v6g10n_Q_s<seed>_dr*/qexec_<P>[_push].npz),
fold-1 TEST clips, Q_exec trained on the training clips only (mean of the 3 frac1.0 nets).
For an alarm threshold tau:
  falls caught with lead >= L : share of falling rollouts with Q_exec < tau at some step in [fall - 2 s, fall - L]
  false alarms                : share of rollouts WITHOUT a fall that ever have Q_exec < tau; alarms (downward
                                crossings) per minute of such rollouts
  python qexec_early_warning.py [seed=1] > runs/qexec_early_warning_s1.md"""
import os
import sys

import numpy as np

H = "/root/gpufree-data/hnu_v2"
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 1
LV = ["Z0", "Z1", "Z2", "Z3", "Z4"]
CONDS = [("理想", f"results_v6g10n_Q_s{SEED}_dr0", ""), ("鲁棒+推扰", f"results_v6g10n_Q_s{SEED}_dr1", "_push")]
TAUS = [0.3, 0.5, 0.7]
HZ = 50
import json  # noqa: E402

TEST = set(json.load(open(f"{H}/data/clip_split_trial/split_v2.json"))["test"])


def load(d, x, P):
    f = f"{H}/runs/{d}/qexec_{P}{x}.npz"
    if not os.path.exists(f):
        return None
    z = np.load(f)
    nets = [str(s) for s in z["nets"]]
    k = [i for i, p in enumerate(nets) if "quality_v6g10n_tr_" in p and "/frac1.0_" in p]
    names = [str(s) for s in z["names"]]
    test = np.array([names[c] in TEST for c in z["clip"]])
    q = z["q"][:, test][:, :, k].mean(2)  # [T, B_test]
    return q, z["fell"][test].astype(bool)


print(f"# Q_exec 作为执行中的摔倒监测：提前预警分析（第 1 折种子 {SEED} 学生，test 片段，只用训练片段训练的 Q_exec）\n")
print("- 预警：Q_exec 低于阈值 τ。")
print("- 提前 ≥ L 被发现：在摔倒前 2 s 到摔倒前 L 秒之间，至少有一步 Q_exec < τ。")
print("- 误报：没有摔倒的执行中，Q_exec 曾经低于 τ；每分钟误报次数按 Q_exec 向下穿过 τ 的次数计算。\n")
for c, d, x in CONDS:
    print(f"## {c}\n")
    print("| τ | 层级 | 摔倒次数 | 提前 ≥ 0.2 s 发现 % | 提前 ≥ 0.5 s % | 提前 ≥ 1.0 s % | 未摔倒执行数 | 有误报的执行 % | 误报次数 / 分钟 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for tau in TAUS:
        for P in LV:
            r = load(d, x, P)
            if r is None:
                continue
            q, fell = r
            alive = ~np.isnan(q)
            T = alive.sum(0)
            low = alive & (q < tau)
            caught = {L: [] for L in (0.2, 0.5, 1.0)}
            for b in np.where(fell)[0]:
                fs = T[b] - 1
                for L in caught:
                    lo, hi = max(fs - 2 * HZ, 0), fs - int(L * HZ)
                    caught[L].append(hi >= lo and bool(low[lo : hi + 1, b].any()))
            ok = np.where(~fell)[0]
            fa = [bool(low[:, b].any()) for b in ok]
            cross = sum(int(((~low[:-1, b]) & low[1:, b] & alive[1:, b]).sum() + low[0, b]) for b in ok)
            minutes = T[ok].sum() / HZ / 60
            pct = lambda v: f"{100 * np.mean(v):.0f}" if len(v) else "–"  # noqa: E731
            print(f"| {tau} | L{P[1]} | {int(fell.sum())} | {pct(caught[0.2])} | {pct(caught[0.5])} | {pct(caught[1.0])} | "
                  f"{len(ok)} | {pct(fa)} | {cross / max(minutes, 1e-9):.2f} |")
    print()
