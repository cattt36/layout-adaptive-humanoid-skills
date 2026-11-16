"""Aggregate evaluation outputs into paper-ready tables (markdown + json).

Closed-loop metrics are computed per action and macro-averaged; 95 % CIs by cluster bootstrap over clips.
Columns:  success (clean env)  |  success under pushes + observation noise  |  dangerous failure (= fall rate under
pushes)  |  teacher retention (student / teacher success, clean).  Reported for unseen-subject test clips and all clips.
"""
import glob, json, os, sys
import numpy as np

R = sys.argv[1]  # runs/results
Q = sys.argv[2]  # runs/quality
OFF = sys.argv[3] if len(sys.argv) > 3 else None  # offline metrics csv dir
TASKS = ("ThreeStepPoleRun", "FiveStepPoleRun", "CrouchedTraverse")
rng = np.random.default_rng(0)


def load(tag, push):
    p = os.path.join(R, f"eval_{'teacher' if tag == 'teacher' else 'student'}_{tag}{'_push' if push else ''}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def macro(ev, split):
    """macro over actions of mean clip success; returns (mean, lo, hi, per-task dict)."""
    per = {}
    for t in TASKS:
        s = np.array([c["success"] for c in ev["clips"] if c["task"] == t and (split == "all" or c["split"] == split)])
        if len(s):
            per[t] = s
    means = np.array([v.mean() for v in per.values()])
    boots = []
    for _ in range(2000):
        boots.append(np.mean([rng.choice(v, len(v)).mean() for v in per.values()]))
    return float(means.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)), {k: float(v.mean()) for k, v in per.items()}


def fmt(m, lo, hi):
    return f"{100*m:.1f} [{100*lo:.1f}, {100*hi:.1f}]"


rows, out = [], {}
order = ["teacher", "ref", "nolatent", "Z0", "Z1", "Z2", "Z3", "Z4", "Z4_moe"]
names = {"teacher": "Expert policy bank (teacher, privileged)", "ref": "Student + explicit reference (upper bound)",
         "nolatent": "Student, no latent (task id + progress)", "Z4_moe": "L4 latent-gated MoE student"}
for split in ("test", "all"):
    t_clean = load("teacher", False)
    tm = macro(t_clean, split)[0] if t_clean else float("nan")
    lines = [f"\n### Closed-loop simulation — {'unseen-subject test clips' if split == 'test' else 'all clips'}\n",
             "| Model | Success (clean) | Success (push+noise) | Dangerous failure (push) | Teacher retention | ThreeStepPoleRun / FiveStepPoleRun / DuMu (clean) |",
             "|---|---|---|---|---|---|"]
    for tag in order:
        c, p = load(tag, False), load(tag, True)
        if not c:
            continue
        mc = macro(c, split)
        mp = macro(p, split) if p else (float("nan"),) * 3 + ({},)
        name = names.get(tag, f"L{tag[1]} student" if tag.startswith("Z") else tag)
        ret = mc[0] / tm if tm == tm and tm > 0 else float("nan")
        per = " / ".join(f"{100*mc[3].get(t, float('nan')):.0f}" for t in TASKS)
        lines.append(f"| {name} | {fmt(*mc[:3])} | {fmt(*mp[:3])} | {100*(1-mp[0]):.1f} | {100*ret:.1f}% | {per} |")
        out[f"{split}/{tag}"] = dict(clean=mc[:3], push=mp[:3], retention=ret, per_task_clean=mc[3], per_task_push=mp[3])
    rows += lines

# executability
rows += ["\n### Executability Q_exec (test rollouts, env-disjoint)\n",
         "| Profile | Score | AUROC | AUPRC(fail) | Brier | ECE | Risk@80% cov | Risk@50% cov | No-filter risk |", "|---|---|---|---|---|---|---|---|---|"]
for qdir in sorted(glob.glob(os.path.join(Q, "Z*"))):
    f = os.path.join(qdir, "quality_eval.json")
    if not os.path.exists(f):
        continue
    e = json.load(open(f))
    out[f"qexec/{os.path.basename(qdir)}"] = {k: v for k, v in e.items()}
    for key, label in (("heuristic", "offline heuristic"), ("uncalibrated", "learned, uncalibrated"),
                       ("temperature_scaled", "learned + temperature"), ("isotonic", "learned + isotonic")):
        m = e[key]
        rows.append(f"| {os.path.basename(qdir)} | {label} | {m['auroc']:.3f} | {m['auprc_failure']:.3f} | {m['brier']:.4f} | "
                    f"{m['ece']:.4f} | {100*m['risk@cov80']:.2f}% | {100*m['risk@cov50']:.2f}% | {100*e['no_filter_risk']:.2f}% |")

md = "\n".join(rows)
open(os.path.join(R, "paper_tables.md"), "w").write(md)
json.dump(out, open(os.path.join(R, "paper_tables.json"), "w"), indent=1, default=float)
print(md)
