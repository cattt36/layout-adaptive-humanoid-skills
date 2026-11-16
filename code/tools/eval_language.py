"""Language -> z_sem -> skill routing per supervision level (deployment chain of the paper).

For every latent checkpoint: each benchmark instruction (Results/8.4 router_ood_benchmark_v1, 72 cases) is encoded
with the task-only z_sem encoder; the nearest z_skill neighbours in the TRAIN gallery (all three actions) give the
routed action (top-1 and top-5 majority).  Rejection quality: AUROC of the max cosine similarity separating
resolvable from ambiguous / out-of-scope instructions (threshold-free).

  python eval_language.py <benchmark.json> <latent_root> [<latent_root> ...]   (latent_root contains seed*/Z*/)
"""
import glob, json, os, sys
from collections import Counter
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "stage4_latent"))
from retrieve_skill import build_model, encode_query  # noqa: E402


def auroc(pos, neg):
    s = np.r_[pos, neg]
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    o = np.argsort(-s)
    y = y[o]
    tp, fp = np.cumsum(y), np.cumsum(1 - y)
    return float(np.trapezoid(np.r_[0, tp / tp[-1]], np.r_[0, fp / fp[-1]]))


bench = json.load(open(sys.argv[1], encoding="utf-8"))["cases"]
rows = {}
for root in sys.argv[2:]:
    for zdir in sorted(glob.glob(os.path.join(root, "seed*", "Z*"))):
        prof, seed = os.path.basename(zdir), os.path.basename(os.path.dirname(zdir))
        if not os.path.exists(os.path.join(zdir, "train_latents.npz")):
            continue
        ck = torch.load(os.path.join(zdir, "best_motion_latent.pt"), map_location="cpu", weights_only=False)
        os.environ["HNU_TEXT_ENCODER"] = str((ck.get("args", {}) or {}).get("text_encoder", "hash"))
        model, schema, _ = build_model(ck)
        g = np.load(os.path.join(zdir, "train_latents.npz"), allow_pickle=True)
        gkey = os.environ.get("HNU_ROUTE_GALLERY", "z_skill")  # z_skill (motion space) or z_sem (semantic space)
        Zs = g[gkey] / np.linalg.norm(g[gkey], axis=1, keepdims=True)
        acts = np.array([str(t).rsplit("_", 1)[0] for t in g["trial_id"]])
        res = {"top1": [], "top5": [], "cond": [], "maxsim": [], "abstain": []}
        for c in bench:
            q = encode_query(model, schema, c["query"], query_mode="task")
            q = q / np.linalg.norm(q)
            sim = Zs @ q
            o = np.argsort(-sim)
            res["maxsim"].append(float(sim[o[0]]))
            res["abstain"].append(bool(c["should_abstain"]))
            res["cond"].append(c["condition"])
            res.setdefault("routed", []).append(dict(id=c.get("id"), query=c["query"], expected=c.get("expected_action"),
                                                     abstain=bool(c["should_abstain"]), routed=str(acts[o[0]]),
                                                     maxsim=float(sim[o[0]])))
            if not c["should_abstain"]:
                res["top1"].append(acts[o[0]] == c["expected_action"])
                res["top5"].append(Counter(acts[o[:5]]).most_common(1)[0][0] == c["expected_action"])
        ms, ab = np.array(res["maxsim"]), np.array(res["abstain"])
        conds = [c["condition"] for c in bench if not c["should_abstain"]]
        per_cond = {k: float(np.mean([t for t, cc in zip(res["top1"], conds) if cc == k])) for k in sorted(set(conds))}
        if os.environ.get("HNU_LANG_DUMP"):  # per-query routing decisions, for the end-to-end chain evaluation
            with open(os.path.join(zdir, "language_routing.json"), "w", encoding="utf-8") as f:
                json.dump(res["routed"], f, ensure_ascii=False, indent=1)
        key = (os.path.basename(os.path.normpath(root)), prof)
        rows.setdefault(key, []).append(dict(seed=seed, top1=float(np.mean(res["top1"])), top5=float(np.mean(res["top5"])),
                                             reject_auroc=auroc(ms[~ab], ms[ab]), per_cond=per_cond))
print("| latent | level | seeds | route acc top-1 | top-5 vote | reject AUROC | paraphrase | code-switch | ASR noise | implicit |")
print("|---|---|---|---|---|---|---|---|---|---|")
for (v, p), rs in sorted(rows.items()):
    f = lambda k: f"{np.mean([r[k] for r in rs]):.3f} ± {np.std([r[k] for r in rs]):.3f}"  # noqa: E731
    pc = {k: np.mean([r["per_cond"].get(k, np.nan) for r in rs]) for k in
          ("held_out_paraphrase", "bilingual_code_switch", "synthetic_asr_noise", "implicit_description")}
    print(f"| {v} | L{p[1]} | {len(rs)} | {f('top1')} | {f('top5')} | {f('reject_auroc')} | "
          + " | ".join(f"{pc[k]:.2f}" for k in pc) + " |")
