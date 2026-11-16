"""Results 8.1 (offline latent quality) over the 4 CV folds: fold 1 = latent_v5t, folds 2-4 = latent_cv_f<k>; 3 latent
seeds each.  Per level: mean ± sd over the 12 encoders (fold x seed), per-fold means, and paired L4-L0 / L3-L0
differences within (fold, seed) with the number of encoders where the higher level is better.
python aggregate_81_cv.py <paper_root> -> prints markdown and writes <paper_root>/offline_81_cv_summary.md"""
import csv
import glob
import os
import sys

import numpy as np

root = sys.argv[1] if len(sys.argv) > 1 else "/root/gpufree-data/hnu_v2/paper_root"
FOLDS = {1: "latent_v5t", 2: "latent_cv_f2", 3: "latent_cv_f3", 4: "latent_cv_f4"}
LV = ["L0", "L1", "L2", "L3", "L4"]
data = {}  # (metric, level) -> {(fold, seed): value}
for F, V in FOLDS.items():
    for d in sorted(glob.glob(os.path.join(root, f"{V}_seed*"))):
        s = d.rsplit("seed", 1)[1]
        o = os.path.join(d, "Results", "8.1_Offline_Representation", "outputs")
        f = os.path.join(o, "offline_retrieval_macro_by_level.csv")
        if not os.path.exists(f):
            print("missing", f)
            continue
        for r in csv.DictReader(open(f, encoding="utf-8-sig")):
            for k, col in (("R@1", "task_macro_r_at_1"), ("R@5", "task_macro_r_at_5"), ("MRR", "task_macro_mrr")):
                data.setdefault((f"retrieval {r['protocol']} {k}", r["supervision_level"]), {})[(F, s)] = float(r[col])
        acc = {}
        for r in csv.DictReader(open(os.path.join(o, "offline_latent_metrics.csv"), encoding="utf-8-sig")):
            for k, col in (("phase BA", "phase_balanced_accuracy"), ("quality mAP", "quality_macro_average_precision"),
                           ("quality AUROC", "quality_macro_auroc"), ("deploy semantic cos", "deploy_semantic_cosine")):
                x = r.get(col)
                if x not in ("", None):
                    acc.setdefault((k, r["supervision_level"]), []).append(float(x))
        for key, v in acc.items():  # mean over actions
            data.setdefault(key, {})[(F, s)] = float(np.mean(v))


def norm_lv(l):
    return l if l.startswith("L") else f"L{l[-1]}"


levels = sorted({k[1] for k in data})
metrics = list(dict.fromkeys(k[0] for k in data))
out = [f"# Results 8.1 over 4 CV folds × 3 latent seeds (fold 1 = latent_v5t; {root})\n"]
for m in metrics:
    out.append(f"\n## {m}\n")
    out.append("| Level | 4 folds mean ± sd (n) | fold 1 | fold 2 | fold 3 | fold 4 |")
    out.append("|---|---|---|---|---|---|")
    for l in levels:
        v = data.get((m, l), {})
        a = np.array(list(v.values()))
        if not len(a):
            continue
        per = [np.mean([x for (F, _), x in v.items() if F == f]) if any(F == f for F, _ in v) else np.nan for f in (1, 2, 3, 4)]
        out.append(f"| {l} | {a.mean():.3f} ± {a.std(ddof=1) if len(a) > 1 else 0:.3f} ({len(a)}) | " + " | ".join(f"{p:.3f}" for p in per) + " |")
    lo = [l for l in levels if l.endswith("0")]
    for hi in [l for l in levels if l.endswith(("3", "4"))]:
        if not lo:
            continue
        a, b = data.get((m, hi), {}), data.get((m, lo[0]), {})
        common = sorted(set(a) & set(b))
        if common:
            dd = np.array([a[k] - b[k] for k in common])
            out.append(f"\n{hi} − {lo[0]} (paired within fold × seed): mean {dd.mean():+.3f}, higher in {int((dd > 0).sum())}/{len(dd)} encoders")
md = "\n".join(out)
print(md)
open(os.path.join(root, "offline_81_cv_summary.md"), "w").write(md)
