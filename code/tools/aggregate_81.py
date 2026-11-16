"""Mean ± std over seeds of the Results-8.1 metrics for latent_v2 / latent_v3 paper roots."""
import csv, glob, os, sys
import numpy as np

root = sys.argv[1]  # hnu_v2/paper_root
out = []
for V in ("latent_v2", "latent_v3", "latent_v4b", "latent_v4d", "latent_v5", "latent_v5t"):
    ret, lat = {}, {}
    for d in sorted(glob.glob(os.path.join(root, f"{V}_seed*"))):
        o = os.path.join(d, "Results", "8.1_Offline_Representation", "outputs")
        f = os.path.join(o, "offline_retrieval_macro_by_level.csv")
        if not os.path.exists(f):
            continue
        for r in csv.DictReader(open(f, encoding="utf-8-sig")):
            ret.setdefault((r["supervision_level"], r["protocol"]), []).append(
                (float(r["task_macro_r_at_1"]), float(r["task_macro_r_at_5"]), float(r["task_macro_mrr"])))
        for r in csv.DictReader(open(os.path.join(o, "offline_latent_metrics.csv"), encoding="utf-8-sig")):
            def fl(x):
                return float(x) if x not in ("", None) else np.nan
            lat.setdefault(r["supervision_level"], []).append((fl(r["phase_balanced_accuracy"]), fl(r["quality_macro_average_precision"]),
                                                              fl(r["quality_macro_auroc"]), fl(r["deploy_semantic_cosine"])))
    if not ret:
        continue
    n = len(glob.glob(os.path.join(root, f"{V}_seed*")))
    out.append(f"\n### {V} ({n} seeds) — skill retrieval (task-macro, unseen-subject test queries vs train gallery)\n")
    out.append("| Level | protocol | R@1 | R@5 | MRR |")
    out.append("|---|---|---|---|---|")
    for (lv, pr), v in sorted(ret.items()):
        a = np.array(v)
        out.append(f"| {lv} | {pr} | {a[:,0].mean():.3f} ± {a[:,0].std():.3f} | {a[:,1].mean():.3f} ± {a[:,1].std():.3f} | {a[:,2].mean():.3f} ± {a[:,2].std():.3f} |")
    out.append(f"\n### {V} — latent diagnostics (mean over actions and seeds)\n")
    out.append("| Level | gait-phase BA (z_skill, NC probe) | quality mAP | quality AUROC | deploy semantic cos |")
    out.append("|---|---|---|---|---|")
    for lv, v in sorted(lat.items()):
        a = np.array(v, dtype=float)
        out.append(f"| {lv} | {np.nanmean(a[:,0]):.3f} ± {np.nanstd(a[:,0]):.3f} | {np.nanmean(a[:,1]):.3f} | {np.nanmean(a[:,2]):.3f} | {np.nanmean(a[:,3]):.3f} |")
md = "\n".join(out)
print(md)
open(os.path.join(root, "offline_81_summary.md"), "w").write(md)
