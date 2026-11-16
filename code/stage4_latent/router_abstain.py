"""Abstention rule for the router v2 (router_query_encoder_ft2.pt), identical for every level.

v2 rejected when the best cosine to the TRAIN z_skill gallery was below one threshold -- unstable (correct
rejection 24-53 %, large seed variance).  Here the per-action score is the best cosine among that action's TRAIN
windows; reject if  best score < tau_s  (out of scope)  OR  best - second-best action score < tau_m  (ambiguous).
(tau_s, tau_m) maximise the balanced accuracy {route correctly | reject} on the validation commands (val2:
resolvable + reject); the Results 8.4 benchmark is only scored.  Reported: coverage of resolvable commands,
accuracy of accepted, correct rejection, wrong-action activation (= accepted and wrong, over all commands).

  python stage4_latent/router_abstain.py <instructions_val_v2.json> <benchmark.json> <latent_root> [tag=ft2]
"""
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from retrieve_skill import build_model, hashed_text_vector  # noqa: E402

v2 = json.load(open(sys.argv[1], encoding="utf-8"))
bench = json.load(open(sys.argv[2], encoding="utf-8"))["cases"]
root = sys.argv[3]
tag = sys.argv[4] if len(sys.argv) > 4 else "ft2"
ACTS = ["CrouchedTraverse", "ThreeStepPoleRun", "FiveStepPoleRun"]
torch.set_num_threads(8)
summary = {}
for zdir in sorted(glob.glob(os.path.join(root, "seed*", "Z*"))):
    f = os.path.join(zdir, f"router_query_encoder_{tag}.pt")
    if not os.path.exists(f):
        continue
    prof = os.path.basename(zdir)
    ck = torch.load(os.path.join(zdir, "best_motion_latent.pt"), map_location="cpu", weights_only=False)
    os.environ["HNU_TEXT_ENCODER"] = str((ck.get("args", {}) or {}).get("text_encoder", "hash"))
    model, schema, _ = build_model(ck)
    q = model.task_semantic_encoder
    q.load_state_dict(torch.load(f, map_location="cpu", weights_only=False)["state"])
    q.eval()
    tdim = int(schema.get("text_dim", 128))
    g = np.load(os.path.join(zdir, "train_latents.npz"), allow_pickle=True)
    gacts = np.array([str(t).rsplit("_", 1)[0] for t in g["trial_id"]])
    K = F.normalize(torch.tensor(g["z_skill"], dtype=torch.float32), dim=-1)
    masks = [torch.tensor(gacts == a) for a in ACTS]

    def scores(texts):
        X = torch.tensor(np.stack([np.r_[hashed_text_vector(f"task_goal: {t}".strip(), tdim), 1.0] for t in texts]).astype(np.float32))
        with torch.no_grad():
            S = F.normalize(q(X), dim=-1) @ K.T
        return torch.stack([S[:, m].max(1).values for m in masks], 1).numpy()  # [n, 3] best cosine per action

    # gate in the query space itself: per-action best cosine to the in-scope TRAINING commands (same encoder)
    if os.environ.get("HNU_ABSTAIN_SPACE", "skill") == "query":
        ins = json.load(open(os.environ["HNU_INSTRUCTIONS"], encoding="utf-8"))
        goal = {a: sorted(set(str(x) for x, y in zip(g["task_goal"], gacts) if y == a)) for a in ACTS}
        ref = {a: goal[a] + ins["train"][a] + ins["val"][a] for a in ACTS}
        def enc(texts):
            X = torch.tensor(np.stack([np.r_[hashed_text_vector(f"task_goal: {t}".strip(), tdim), 1.0] for t in texts]).astype(np.float32))
            with torch.no_grad():
                return F.normalize(q(X), dim=-1)
        R = {a: enc(ref[a]) for a in ACTS}
        def gate_scores(texts):
            Qe = enc(texts)
            return np.stack([(Qe @ R[a].T).max(1).values.numpy() for a in ACTS], 1)
    else:
        gate_scores = None
    vt = [t for a in ACTS for t in v2["val"][a]]
    vy = np.array([i for i, a in enumerate(ACTS) for _ in v2["val"][a]] + [-1] * len(v2["val_reject"]))
    Sv = scores(vt + v2["val_reject"])
    Sb = scores([c["query"] for c in bench])
    Gv = gate_scores(vt + v2["val_reject"]) if gate_scores else Sv  # rejection uses the gate scores,
    Gb = gate_scores([c["query"] for c in bench]) if gate_scores else Sb  # the route stays the z_skill argmax
    yb = np.array([-1 if c["should_abstain"] else ACTS.index(c["expected_action"]) for c in bench])

    def decide(S, ts, tm, G=None):
        G = S if G is None else G
        top = np.sort(G, 1)
        pred = S.argmax(1)
        rej = (top[:, -1] < ts) | (top[:, -1] - top[:, -2] < tm)
        return pred, rej

    def bal(S, y, ts, tm):
        pred, rej = decide(S, ts, tm, Gv)
        res = y >= 0
        return 0.5 * (np.mean((pred == y) & ~rej & res) / res.mean() + np.mean(rej[~res]))

    grid_s = np.unique(np.r_[-1.0, np.quantile(Gv.max(1), np.linspace(0, 1, 41))])
    grid_m = np.r_[0.0, np.quantile(np.diff(np.sort(Gv, 1)[:, -2:], axis=1)[:, 0], np.linspace(0, 1, 41))]
    best = max(((bal(Sv, vy, ts, tm), -ts, -tm, ts, tm) for ts in grid_s for tm in grid_m))  # ties -> least rejection
    ts, tm = best[3], best[4]
    pred, rej = decide(Sb, ts, tm, Gb)
    res = yb >= 0
    rows = [dict(id=c.get("id"), query=c["query"], expected=c.get("expected_action"), abstain=bool(c["should_abstain"]),
                 routed=ACTS[p], rejected=bool(r)) for c, p, r in zip(bench, pred, rej)]
    json.dump(rows, open(os.path.join(zdir, f"language_routing_{tag}a{os.environ.get('HNU_ABSTAIN_SPACE', 'skill')[0]}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    s = summary.setdefault(prof, {k: [] for k in ("top1", "cov", "acc", "rej", "wrong", "safe")})
    s["top1"].append(np.mean(pred[res] == yb[res]))
    s["cov"].append(np.mean(~rej[res]))
    s["acc"].append(np.mean(pred[res & ~rej] == yb[res & ~rej]) if (res & ~rej).any() else np.nan)
    s["rej"].append(np.mean(rej[~res]))
    s["wrong"].append(np.mean(~rej & (pred != yb)))  # accepted and wrong action, incl. accepted rejection cases
    s["safe"].append(np.mean(np.where(res, (pred == yb) & ~rej, rej)))
print("| level | top-1 (no abstention) | coverage (resolvable) | accuracy of accepted | correct rejection | "
      "wrong-action activation | safe decision accuracy |")
print("|---|---|---|---|---|---|---|")
for prof, s in sorted(summary.items()):
    f = lambda k: f"{100 * np.nanmean(s[k]):.1f} ± {100 * np.nanstd(s[k]):.1f}"  # noqa: E731
    print(f"| L{prof[1]} | {f('top1')} | {f('cov')} | {f('acc')} | {f('rej')} | {f('wrong')} | {f('safe')} |")
