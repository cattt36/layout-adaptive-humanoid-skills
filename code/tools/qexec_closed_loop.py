"""Q_exec vs closed-loop execution success (EXPLORATORY item 5; , user request), fold-1 students.

Reads runs/results_v6g10n_Q_s<seed>_dr{0,1}/qexec_<P>[_push].npz (eval_student.py with HNU_EVAL_QEXEC, written by
tools/qexec_heldout.sh) and prints a markdown report (Chinese, for the user):
  1. state level, TEST clips: Q_exec bin -> observed share of states without a fall in the next 1 s (same label and
     keep rule as quality.py collect), coverage (top k %), AUROC; train clips as a comparison line
  2. clip level, TEST clips: per clip a Q_exec score vs the clip's success rate over its 32 rollouts (Spearman;
     quartile table pooled over L0-L4)
       A  ex ante  : mean Q_exec over the first 1 s of every rollout
       B  full run : mean Q_exec over the run WITHOUT the last 1 s before a fall (those states are labelled failures)
       C  split half: mean Q_exec over all states of the even rollouts vs the success of the odd rollouts
          (reference: success of the even rollouts vs the odd ones)
  3. rollout level, TEST clips: AUROC of score A / B for the rollout's own outcome
Q_exec of a group = mean of its 3 nets (training seeds).  Q_exec predicts falls; success v2 additionally needs every
stake stepped on, so both success v2 and "no fall" are reported.
  python qexec_closed_loop.py [seed=1] > runs/qexec_closed_loop_s1.md
"""
import json
import os
import sys

import numpy as np

H = "/root/gpufree-data/hnu_v2"
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 1
HZ = 50  # quality.py HORIZON: 1 s at 50 Hz
LV = ["Z0", "Z1", "Z2", "Z3", "Z4"]
# (name, results dir stem, file suffix); "track" = the training environment's tracking terminations count as
# failures, i.e. the label Q_exec was trained on
CONDS = [("理想", "results_v6g10n_Q_s{}_dr0", ""), ("鲁棒+推扰", "results_v6g10n_Q_s{}_dr1", "_push"),
         ("理想·跟踪终止", "results_v6g10ntrk_Q_s{}_dr0", "")]
GROUPS = [("tr1.0", "Q_exec（只用训练片段，全量数据）"), ("tr0.05", "Q_exec（只用训练片段，5% 数据）"),
          ("all1.0", "旧 Q_exec（全部片段含 test，全量数据；仅作对照）")]
BINS = [(0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, .9), (.9, 1.01)]
TASKS = [("CrouchedTraverse", "独木桥"), ("ThreeStepPoleRun", "三步桩"), ("FiveStepPoleRun", "五步桩")]
split = json.load(open(os.environ.get("HNU_SPLIT_JSON") or f"{H}/data/clip_split_trial/split_v2.json"))
TEST, TRAIN = set(split["test"]), set(split["train"])


def group_of(path):
    return ("tr" if "quality_v6g10n_tr_" in path else "all") + ("1.0" if "/frac1.0_" in path else "0.05")


def rankdata(x):
    x = np.asarray(x, float)
    o = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    xs = x[o]
    i = 0
    while i < len(x):  # average rank for ties
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]:
            j += 1
        r[o[i : j + 1]] = (i + j) / 2
        i = j + 1
    return r


def spearman(a, b):  # pairs with an undefined score are dropped
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    ra, rb = rankdata(a[ok]), rankdata(b[ok])
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def auroc(score, pos):
    pos = np.asarray(pos, bool)
    n1, n0 = pos.sum(), (~pos).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[pos].sum() - n1 * (n1 - 1) / 2) / (n1 * n0))


def load(stem, x, P):
    f = f"{H}/runs/{stem.format(SEED)}/qexec_{P}{x}.npz"
    if not os.path.exists(f):
        return None
    d = np.load(f)
    nets = [str(s) for s in d["nets"]]
    q = d["q"]  # [T, B, K], NaN once the rollout has ended
    G = {g: q[:, :, [i for i, p in enumerate(nets) if group_of(p) == g]].mean(2)
         for g, _ in GROUPS if any(group_of(p) == g for p in nets)}
    alive = ~np.isnan(q[:, :, 0])
    T = alive.sum(0)  # active steps per rollout (contiguous from step 0)
    fell = d["fell"].astype(bool)
    t = np.arange(q.shape[0])[:, None]
    fail_soon = fell[None] & (T[None] - 1 - t < HZ)
    keep = alive & ((t + HZ < T[None]) | fell[None])
    names = [str(s) for s in d["names"]]
    cname = np.array([names[c] for c in d["clip"]])
    return dict(G=G, alive=alive, fail_soon=fail_soon, keep=keep, fell=fell, sv2=d["success_v2"].astype(bool),
                cname=cname, test=np.isin(cname, list(TEST)), train=np.isin(cname, list(TRAIN)))


def f1(v):
    return "–" if v is None or not np.isfinite(v) else f"{v:.1f}"


def f3(v):
    return "–" if v is None or not np.isfinite(v) else f"{v:.2f}"


D = {(c, P): load(stem, x, P) for c, stem, x in CONDS for P in LV}
missing = [f"{c} {P}" for (c, P), v in D.items() if v is None]
print(f"# Q_exec 与闭环执行成功率对照（第 1 折种子 {SEED} 学生，4090 服务器）\n")
print("- 学生在每个片段上执行 32 次，从第 1 帧开始，成功标准为新标准（v2）。主要结果只统计 16 个 test 片段。")
print("- Q_exec = 当前状态下预测\"接下来 1 s 内不摔倒\"的概率（经过等渗校准），取 3 个训练种子的平均值。")
print("- 新 Q_exec 只用训练片段（51 个）的 rollout 训练，test 片段完全没见过。旧 Q_exec 用全部片段训练（含 test），只作对照。")
print("- Q_exec 预测的是摔倒；新标准还要求三步桩/五步桩每个桩都踩上，所以片段级同时报告新标准成功率和\"不摔倒\"比例。")
print("- Q_exec 训练时的\"失败\"是训练环境的终止条件（跟踪误差过大也算），比真正摔倒宽。\"理想·跟踪终止\"一栏用的是和训练相同的终止条件，"
      "这一栏的\"不摔倒\"实际指\"没有触发跟踪终止\"，用来区分\"排序不准\"和\"标签不同\"。")
if missing:
    print(f"- **缺少结果：{', '.join(missing)}**")
print()

# ------------------------------------------------------------------ 1. state level
print("## 1. 状态级：Q_exec 区间 → 实际 1 s 内不摔倒的比例（test 片段）\n")
print("每格数据：平均预测值 / 实际 % （括号内为样本比例 %）。top k%：只保留 Q_exec 最高的 k% 状态时，实际不摔倒的比例。\n")
for g, gname in GROUPS:
    print(f"### {gname}\n")
    print("| 条件 | 层级 | " + " | ".join(f"{a:.1f}–{min(b, 1):.1f}" for a, b in BINS) +
          " | top 20% | top 50% | top 80% | 全部 | AUROC test | AUROC 训练片段 |")
    print("|---|---|" + "---|" * (len(BINS) + 6))
    for c, _, _ in CONDS:
        for P in LV:
            d = D[(c, P)]
            if d is None or g not in d["G"]:
                continue
            m = d["keep"] & d["test"][None]
            p, y = d["G"][g][m], ~d["fail_soon"][m]
            cells = []
            for a, b in BINS:
                k = (p >= a) & (p < b)
                cells.append(f"{100 * p[k].mean():.0f}/{100 * y[k].mean():.0f} ({100 * k.mean():.0f})" if k.any() else "–")
            o = np.argsort(-p, kind="stable")
            top = lambda f: 100 * y[o[: max(int(round(f * len(o))), 1)]].mean()  # noqa: E731
            mt = d["keep"] & d["train"][None]
            print(f"| {c} | L{P[1]} | " + " | ".join(cells) + f" | {top(.2):.1f} | {top(.5):.1f} | {top(.8):.1f} | "
                  f"{100 * y.mean():.1f} | {f3(auroc(p, y))} | {f3(auroc(d['G'][g][mt], ~d['fail_soon'][mt]))} |")
    print()

# ------------------------------------------------------------------ 2. clip level


def clip_scores(d, g):
    Q, al, fs = d["G"][g], d["alive"], d["fail_soon"]
    out = []
    for name in sorted(set(d["cname"][d["test"]])):
        idx = np.where(d["cname"] == name)[0]
        qa = np.nanmean(np.where(al[:HZ, idx], Q[:HZ, idx], np.nan))
        per_b = [np.nanmean(np.where(al[:, i] & ~fs[:, i], Q[:, i], np.nan)) if (al[:, i] & ~fs[:, i]).any() else np.nan
                 for i in idx]
        ev, od = idx[0::2], idx[1::2]
        qc = np.nanmean(np.where(al[:, ev], Q[:, ev], np.nan))
        out.append(dict(clip=name, task=name.split("_")[0], A=float(qa), B=float(np.nanmean(per_b)), C=float(qc),
                        sv2=float(d["sv2"][idx].mean()), nofall=float(1 - d["fell"][idx].mean()),
                        sv2_odd=float(d["sv2"][od].mean()), nofall_odd=float(1 - d["fell"][od].mean()),
                        sv2_even=float(d["sv2"][ev].mean()), nofall_even=float(1 - d["fell"][ev].mean()),
                        epA=[float(np.nanmean(np.where(al[:HZ, i], Q[:HZ, i], np.nan))) for i in idx], epB=per_b,
                        ep_sv2=d["sv2"][idx], ep_nofall=~d["fell"][idx]))
    return out


print("## 2. 片段级：每个 test 片段的 Q_exec 与该片段 32 次执行的成功率（新 Q_exec，全量数据）\n")
print("- A 事前：每次执行前 1 s 的平均 Q_exec（还没到障碍物，不含摔倒信息）")
print("- B 全程：整次执行的平均 Q_exec，去掉摔倒前最后 1 s")
print("- C 交叉：用偶数次执行的全程 Q_exec，预测奇数次执行的成功率；对照为\"偶数次的实际成功率\"预测奇数次\n")
print("### 2a. 秩相关（Spearman，16 个 test 片段；合并为 5 个层级 × 16 = 80 个点）\n")
print("| 条件 | 层级 | A–新标准 | A–不摔倒 | B–新标准 | B–不摔倒 | C–新标准 | C–不摔倒 | 对照：偶数次成功率–新标准 | 对照–不摔倒 |")
print("|---|---|---|---|---|---|---|---|---|---|")
CS = {}
for c, _, _ in CONDS:
    pool = []
    for P in LV:
        d = D[(c, P)]
        if d is None or "tr1.0" not in d["G"]:
            continue
        r = CS[(c, P)] = clip_scores(d, "tr1.0")
        pool += r
        col = lambda k: [x[k] for x in r]  # noqa: E731
        print(f"| {c} | L{P[1]} | {f3(spearman(col('A'), col('sv2')))} | {f3(spearman(col('A'), col('nofall')))} | "
              f"{f3(spearman(col('B'), col('sv2')))} | {f3(spearman(col('B'), col('nofall')))} | "
              f"{f3(spearman(col('C'), col('sv2_odd')))} | {f3(spearman(col('C'), col('nofall_odd')))} | "
              f"{f3(spearman(col('sv2_even'), col('sv2_odd')))} | {f3(spearman(col('nofall_even'), col('nofall_odd')))} |")
    if pool:
        col = lambda k: [x[k] for x in pool]  # noqa: E731
        print(f"| {c} | **合并** | {f3(spearman(col('A'), col('sv2')))} | {f3(spearman(col('A'), col('nofall')))} | "
              f"{f3(spearman(col('B'), col('sv2')))} | {f3(spearman(col('B'), col('nofall')))} | "
              f"{f3(spearman(col('C'), col('sv2_odd')))} | {f3(spearman(col('C'), col('nofall_odd')))} | "
              f"{f3(spearman(col('sv2_even'), col('sv2_odd')))} | {f3(spearman(col('nofall_even'), col('nofall_odd')))} |")
print()
print("### 2b. 分档：按片段 Q_exec（A 事前 / B 全程）从低到高分成 4 档（5 个层级合并，每档约 20 个片段×层级；"
      "B 没有定义的片段（全部执行在 1 s 内终止）不计入）\n")
for c, _, _ in CONDS:
    for sk, sname in (("A", "A 事前"), ("B", "B 全程")):
        pool = [x for P in LV for x in CS.get((c, P), []) if np.isfinite(x[sk])]
        if not pool:
            continue
        o = np.argsort([x[sk] for x in pool], kind="stable")
        print(f"**{c}，{sname}**\n")
        print("| 档 | Q_exec 范围 | 平均 Q_exec % | 新标准成功率 % | 不摔倒 % | 独木桥 / 三步桩 / 五步桩个数 |")
        print("|---|---|---|---|---|---|")
        for qi, part in enumerate(np.array_split(o, 4)):
            xs = [pool[i] for i in part]
            v = [x[sk] for x in xs]
            nt = " / ".join(str(sum(x["task"] == t for x in xs)) for t, _ in TASKS)
            print(f"| {qi + 1} | {100 * min(v):.0f}–{100 * max(v):.0f} | {100 * np.mean(v):.1f} | "
                  f"{100 * np.mean([x['sv2'] for x in xs]):.1f} | {100 * np.mean([x['nofall'] for x in xs]):.1f} | {nt} |")
        print()
print("### 2c. 按动作分开（5 个层级合并；排除动作之间难度不同带来的相关）\n")
print("| 条件 | 动作 | 点数 | A–新标准 | A–不摔倒 | B–新标准 | B–不摔倒 |")
print("|---|---|---|---|---|---|---|")
for c, _, _ in CONDS:
    pool = [x for P in LV for x in CS.get((c, P), [])]
    for t, tn in TASKS:
        xs = [x for x in pool if x["task"] == t]
        if not xs:
            continue
        col = lambda k: [x[k] for x in xs]  # noqa: E731
        print(f"| {c} | {tn} | {len(xs)} | {f3(spearman(col('A'), col('sv2')))} | {f3(spearman(col('A'), col('nofall')))} | "
              f"{f3(spearman(col('B'), col('sv2')))} | {f3(spearman(col('B'), col('nofall')))} |")
print()

# ------------------------------------------------------------------ 3. rollout level
print("## 3. 单次执行级：用这次执行自己的 Q_exec 预测这次是否成功（test 片段，AUROC）\n")
print("| 条件 | 层级 | 执行次数 | A 事前–新标准 | A 事前–不摔倒 | B 全程–新标准 | B 全程–不摔倒 | 片段内 A–不摔倒（片段平均） |")
print("|---|---|---|---|---|---|---|---|")
for c, _, _ in CONDS:
    for P in LV:
        r = CS.get((c, P))
        if not r:
            continue
        eA = np.concatenate([x["epA"] for x in r]); eB = np.concatenate([x["epB"] for x in r])
        s2 = np.concatenate([x["ep_sv2"] for x in r]); nf = np.concatenate([x["ep_nofall"] for x in r])
        okB = np.isfinite(eB)
        within = [auroc(np.array(x["epA"]), x["ep_nofall"]) for x in r]
        within = [w for w in within if np.isfinite(w)]
        print(f"| {c} | L{P[1]} | {len(eA)} | {f3(auroc(eA, s2))} | {f3(auroc(eA, nf))} | {f3(auroc(eB[okB], s2[okB]))} | "
              f"{f3(auroc(eB[okB], nf[okB]))} | {f3(np.mean(within)) if within else '–'} ({len(within)} 个片段) |")
print()

# ------------------------------------------------------------------ per clip (appendix)
print("## 附：每个 test 片段（理想条件，新 Q_exec 全量；L0 / L4）\n")
print("| 片段 | L0 Q_exec A / B % | L0 新标准 / 不摔倒 % | L4 Q_exec A / B % | L4 新标准 / 不摔倒 % |")
print("|---|---|---|---|---|")
r0 = {x["clip"]: x for x in CS.get(("理想", "Z0"), [])}
r4 = {x["clip"]: x for x in CS.get(("理想", "Z4"), [])}
for n in sorted(set(r0) | set(r4)):
    cell = lambda r: (f"{100 * r[n]['A']:.0f} / {100 * r[n]['B']:.0f} | {100 * r[n]['sv2']:.0f} / {100 * r[n]['nofall']:.0f}"  # noqa: E731
                      if n in r else "– | –")
    print(f"| {n} | {cell(r0)} | {cell(r4)} |")
