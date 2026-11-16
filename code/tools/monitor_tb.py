"""Live monitor of a student batch in TensorBoard (run dir runs/logs/rsl_rl/monitor_<V>), refreshed every 5 min:
  TEXT     progress table (round 1 / round 2 / evaluation per level and seed, ETA) and results table (success /
           body error per action, level and condition, mean over the finished seeds, teacher for reference)
  SCALARS  Progress/*            students done, overall percentage
           Success_<action>/<condition>   x-axis = level (0..4): mean over finished seeds
           BodyErr_<action>/<condition>   same, cm
  python tools/monitor_tb.py [V]   (stops after the batch log says "finished")
"""
import glob
import json
import os
import sys
import time

from torch.utils.tensorboard import SummaryWriter

V = sys.argv[1] if len(sys.argv) > 1 else "v6g10s"
# CV fold (v6g10n on the L40 server): python tools/monitor_tb.py v6g10n <fold>; fold 1 keeps the v6g10n names,
# folds 2-4 use the stem v6g10n_f<k> (students / results / logs), L0-L4 x seeds 1-3 only
FOLD = int(sys.argv[2]) if len(sys.argv) > 2 else 1
VS = V if FOLD == 1 else f"{V}_f{FOLD}"
H = "/root/gpufree-data/hnu_v2/runs"
N = f"{H}/night"
LV = ["Z0", "Z1", "Z2", "Z3", "Z4"]
# held-out batches (v6g10h ...): report the TEST-subject clips only, and the No-latent baseline (NL)
HELD = V.endswith(("h", "t", "j", "k", "m", "n"))
SPLIT = os.environ.get("HNU_MON_SPLIT", "test" if HELD else "all")
if HELD and FOLD == 1:
    LV = LV + ["NL", "REF"]
SPLIT_CN = {"test": ("未见过的受试者片段（test）" if V.endswith("h") else "未见过的试次（按试次留出的 test）"), "all": "全部片段"}.get(SPLIT, SPLIT)


# v6g10n: success criterion v2 (every stake stepped on) and the contact-based stake metric
SUCC = "success_v2" if V == "v6g10n" else "success"
STAKE_K = "stakes_all_stepped" if V == "v6g10n" else "stakes_all_hit"


def lab(P):
    return {"NL": "No-latent", "REF": "参考动作(上限)"}.get(P, f"L{P[1]}")
SEEDS = [1, 2, 3, 4, 5] if V in ("v6g10h", "v6g10j", "v6g10k", "v6g10m", "v6g10n") and FOLD == 1 else [1, 2, 3]


def seeds_of(P):  # v6g10j / v6g10k: L0-L4 x 5 seeds, the NL / REF baselines x 3
    return [1, 2, 3] if (V in ("v6g10j", "v6g10k", "v6g10m", "v6g10n") and P in ("NL", "REF")) else SEEDS
TASKS = ["CrouchedTraverse", "ThreeStepPoleRun", "FiveStepPoleRun"]
CN = {"CrouchedTraverse": "独木桥", "ThreeStepPoleRun": "三步桩", "FiveStepPoleRun": "五步桩"}
CONDS = [("nominal", "0", ""), ("nominal_push", "0", "_push"), ("randomised", "1", ""), ("randomised_push", "1", "_push")]
CONDN = {"nominal": "理想", "nominal_push": "理想+推扰", "randomised": "鲁棒", "randomised_push": "鲁棒+推扰"}
R1, R2 = 4000, 2000


def last_it(f):
    try:
        lines = [x for x in open(f).read().splitlines() if x.startswith("{")]
        return json.loads(lines[-1])["it"] + 10 if lines else 0
    except (OSError, ValueError, KeyError):
        return 0


def ev(s, dr, P, x):
    f = f"{H}/results_{VS}_s{s}_dr{dr}/eval_student_{P}{x}.json"
    return json.load(open(f))["summary"] if os.path.exists(f) else None


def status(s, P):
    n_ev = sum(ev(s, dr, P, x) is not None for _, dr, x in CONDS)
    if n_ev == 4:
        return "✅ 完成", 1.0
    if os.path.exists(f"{H}/students_{VS}_d2_s{s}/{P}/student_2000.pt"):
        return f"评估 {n_ev}/4", 0.97
    if os.path.exists(f"{H}/students_{VS}_s{s}/{P}/student_4000.pt"):
        i = min(last_it(f"{N}/{VS}_d2_s{s}_{P}.log"), R2)
        return f"第二轮 {i}/{R2}", (R1 + i) / (R1 + R2) * 0.97
    i = min(last_it(f"{N}/{VS}_s{s}_E_{P}.log"), R1)
    if i == 0:
        return "排队", 0.0
    return f"第一轮 {i}/{R1}", i / (R1 + R2) * 0.97


def main():
    os.makedirs(f"{H}/logs/rsl_rl/monitor_{VS}", exist_ok=True)
    tb = SummaryWriter(f"{H}/logs/rsl_rl/monitor_{VS}")  # main page: success + tracking error only
    det = SummaryWriter(f"{H}/logs/rsl_rl/monitor_{VS}_detail")  # progress, per-seed results, selection
    t_start = os.path.getmtime(f"{N}/{V}.log")
    step = 0
    written = {}
    while True:
        st = {(s, P): status(s, P) for P in LV for s in seeds_of(P)}
        frac = sum(v[1] for v in st.values()) / len(st)
        done = sum(v[1] >= 1.0 for v in st.values())
        el = (time.time() - t_start) / 3600
        eta = f"约 {el / frac * (1 - frac):.1f} 小时后" if frac > 0.02 else "-"
        rows = ["| 层级 | " + " | ".join(f"种子{s}" for s in SEEDS) + " |", "|---" * (len(SEEDS) + 1) + "|"]
        rows += [f"| {lab(P)} | " + " | ".join(st[(s, P)][0] if (s, P) in st else "—" for s in SEEDS) + " |" for P in LV]
        prog = (f"**{VS}：已完成 {done}/{len(st)} 个学生，总进度 {100 * frac:.0f}%，已用 {el:.1f} 小时，预计全部完成：{eta}**"
                f"（更新于 {time.strftime('%m-%d %H:%M')}）\n\n" + "\n".join(rows))
        det.add_text("1_训练进度", prog, step)
        det.add_scalar("Progress/students_done", done, step)
        det.add_scalar("Progress/overall_percent", 100 * frac, step)
        # main page: success and tracking error per action, L0-L4 (mean over the finished seeds) + teacher
        nseed = {P: sum(ev(s, "1", P, "") is not None for s in seeds_of(P)) for P in LV}
        note = (f"已完成种子的平均（已完成种子数：" + "，".join(f"{lab(P)} {nseed[P]}/{len(seeds_of(P))}" for P in LV) +
                f"）\n\n**训练进度：已完成 {done}/{len(st)} 个学生，总进度 {100 * frac:.0f}%，预计全部完成：{eta}；"
                f"更新于 {time.strftime('%m-%d %H:%M')}**\n")
        for key, title, unit, scale in ((SUCC, "成功率" + ("（新标准 v2：每个桩都要踩上）" if SUCC == "success_v2" else ""), "%", 100), ("body_err", "跟踪误差（身体位置误差）", "cm", 100)):
            res = [f"**{title}，单位 {unit}；评估片段：{SPLIT_CN}**；{note}"]
            for T in TASKS:
                res += [f"\n**{CN[T]}**\n", "| | " + " | ".join(CONDN[c] for c, _, _ in CONDS) + " |", "|---" * 5 + "|"]
                tline = []
                for c, dr, x in CONDS:
                    f = f"{H}/results_{V}_bank_teacher_dr{dr}/eval_teacher_teacher{x}.json"
                    tline.append(f"{scale * json.load(open(f))['summary'][f'{T}/{SPLIT}'][key]:.1f}" if os.path.exists(f) else "-")
                res.append("| 教师 | " + " | ".join(tline) + " |")
                for li, P in enumerate(LV):
                    line = []
                    for c, dr, x in CONDS:
                        vals = [v[f"{T}/{SPLIT}"][key] for v in (ev(s, dr, P, x) for s in seeds_of(P)) if v is not None]
                        if not vals:
                            line.append("-")
                            continue
                        val = scale * sum(vals) / len(vals)
                        line.append(f"{val:.1f}")
                        tag = f"{'Success' if key == SUCC else 'BodyErr'}_{T}/{c}"
                        if P not in ("NL", "REF") and written.get((tag, li)) != round(val, 3):  # one point per level, rewritten only on change
                            tb.add_scalar(tag, val, li)
                            written[(tag, li)] = round(val, 3)
                    res.append(f"| {lab(P)} | " + " | ".join(line) + " |")
            tb.add_text("1_成功率" if key == SUCC else "2_跟踪精度", "\n".join(res), step)
        # stakes (nominal, test): all stakes hit %, foot-to-stake-centre distance cm; mean over the finished seeds
        stk = [f"**踩桩（理想环境，{SPLIT_CN}）：全部踩中 % ／ 脚到桩中心距离 cm**\n",
               "| | 三步桩踩中 | 三步桩距离 | 五步桩踩中 | 五步桩距离 |", "|---" * 5 + "|"]
        tf = f"{H}/results_{V}_bank_teacher_dr0/eval_teacher_teacher.json"
        srcs = [("教师", [json.load(open(tf))["summary"]] if os.path.exists(tf) else [])]
        srcs += [(lab(P), [v for v in (ev(s, "0", P, "") for s in seeds_of(P)) if v]) for P in LV]
        for name, vs in srcs:
            cells = []
            for T in ("ThreeStepPoleRun", "FiveStepPoleRun"):
                for k in (STAKE_K, "stake_centre_dist_m"):
                    xs = [v[f"{T}/{SPLIT}"].get(k) for v in vs if v[f"{T}/{SPLIT}"].get(k) is not None]
                    cells.append(f"{100 * sum(xs) / len(xs):.1f}" if xs else "-")
            stk.append(f"| {name} | " + " | ".join(cells) + " |")
        tb.add_text("3_踩桩精度", "\n".join(stk), step)
        # per seed: success % of seeds 1 / 2 / 3 (- = not finished yet)
        per = []
        for T in TASKS:
            per += [f"\n**{CN[T]}**：成功率 %，每格依次为 " + " · ".join(f"种子{s}" for s in SEEDS) + "\n",
                    "| | " + " | ".join(CONDN[c] for c, _, _ in CONDS) + " |", "|---" * 5 + "|"]
            for li, P in enumerate(LV):
                cells = []
                for c, dr, x in CONDS:
                    vs = []
                    for s in seeds_of(P):
                        v = ev(s, dr, P, x)
                        vs.append(f"{100 * v[f'{T}/{SPLIT}'][SUCC]:.1f}" if v else "-")
                        if v:
                            val = 100 * v[f"{T}/{SPLIT}"][SUCC]
                            tag = f"SeedSuccess_{T}/{c}_s{s}"
                            if P not in ("NL", "REF") and written.get((tag, li)) != round(val, 3):
                                det.add_scalar(tag, val, li)
                                written[(tag, li)] = round(val, 3)
                    cells.append(" · ".join(vs))
                per.append(f"| {lab(P)} | " + " | ".join(cells) + " |")
        det.add_text("2_各种子成功率", "\n".join(per), step)
        sel = f"{H}/results_{VS}_selection.json"
        if os.path.exists(sel):
            j = json.load(open(sel))
            det.add_text("3_部署种子", " / ".join(f"{lab(P)}: 种子 {j[P]['selected_seed']}" for P in LV if P in j), step)
        tb.flush()
        det.flush()
        step += 1
        if os.path.exists(f"{N}/{V}.log") and "finished" in open(f"{N}/{V}.log").read() and (os.path.exists(sel) or FOLD > 1):
            break
        time.sleep(300)


if __name__ == "__main__":
    main()
