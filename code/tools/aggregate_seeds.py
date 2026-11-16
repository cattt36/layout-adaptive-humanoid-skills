"""Closed-loop success over RL seeds: per seed macro success (test subjects / all clips, clean / push), mean ± std,
and paired differences vs L0 (same clips, same env seeds) with a clip-cluster bootstrap over seeds × clips."""
import json, os, sys
import numpy as np

dirs = sys.argv[1:]
TASKS = ("ThreeStepPoleRun", "FiveStepPoleRun", "CrouchedTraverse")
LV = ("Z0", "Z1", "Z2", "Z3", "Z4")
rng = np.random.default_rng(0)


def clip_table(d, P, push):
    f = os.path.join(d, f"eval_student_{P}{'_push' if push else ''}.json")
    if not os.path.exists(f):
        return None
    return {c["clip"]: (c["task"], c["split"], c["success"]) for c in json.load(open(f))["clips"]}


def macro(tab, split):
    per = [np.mean([s for (t, sp, s) in tab.values() if t == T and (split == "all" or sp == split)]) for T in TASKS]
    return float(np.mean(per))


rows = ["| Level | seeds | test clean | test push | all clean | all push | Δ vs L0 (all clean, paired) |", "|---|---|---|---|---|---|---|"]
base = {}
for P in LV:
    tabs = [(clip_table(d, P, False), clip_table(d, P, True)) for d in dirs]
    tabs = [t for t in tabs if t[0] and t[1]]
    if not tabs:
        continue
    m = {k: np.array([macro(t[i], sp) for t in tabs]) for k, (i, sp) in
         {"tc": (0, "test"), "tp": (1, "test"), "ac": (0, "all"), "ap": (1, "all")}.items()}
    if P == "Z0":
        base = tabs
        delta = "—"
    else:
        # paired: same seeds index and same clips; bootstrap over clips within each action
        n = min(len(tabs), len(base))
        clips = sorted(set(tabs[0][0]) & set(base[0][0]))
        diff = np.array([[tabs[k][0][c][2] - base[k][0][c][2] for c in clips] for k in range(n)]).mean(0)  # per clip
        task = np.array([tabs[0][0][c][0] for c in clips])
        boots = []
        for _ in range(4000):
            boots.append(np.mean([rng.choice(diff[task == T], (task == T).sum()).mean() for T in TASKS]))
        point = np.mean([diff[task == T].mean() for T in TASKS])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        delta = f"{100*point:+.1f} [{100*lo:+.1f}, {100*hi:+.1f}]"
    f = lambda a: f"{100*a.mean():.1f} ± {100*a.std():.1f}"  # noqa: E731
    rows.append(f"| L{P[1]} | {len(tabs)} | {f(m['tc'])} | {f(m['tp'])} | {f(m['ac'])} | {f(m['ap'])} | {delta} |")
print("\n".join(rows))
