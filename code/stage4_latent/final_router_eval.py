"""FINAL, single scoring of the frozen router on the sealed independent test set (router_test_v2_sealed.json).

Frozen method (fixed before the test set was opened): router v2 query encoders (router_query_encoder_ft2.pt; one
shared training length chosen on val2), route = action of the nearest TRAIN z_skill window, abstention in the query
space (per-action best cosine to the in-scope training commands; reject if best < tau_s or best - second < tau_m;
(tau_s, tau_m) maximise balanced accuracy on val2).  Pre-registered exclusion: test commands with cosine > 0.92
(MiniLM sentence space) to any training / validation command are not scored (count reported).  Baselines: keyword
rules; MiniLM prototype router over the same instruction set (no latent).

  python stage4_latent/final_router_eval.py <test.json> <instructions_v1.json> <val_v2.json> <latent_root>
"""
import glob
import hashlib
import json
import os
import re
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from retrieve_skill import build_model, hashed_text_vector  # noqa: E402

test_path, ins_path, v2_path, root = sys.argv[1:5]
print("test set sha256:", hashlib.sha256(open(test_path, "rb").read()).hexdigest())
test = json.load(open(test_path, encoding="utf-8"))["cases"]
ins = json.load(open(ins_path, encoding="utf-8"))
v2 = json.load(open(v2_path, encoding="utf-8"))
ACTS = ["CrouchedTraverse", "ThreeStepPoleRun", "FiveStepPoleRun"]
torch.set_num_threads(8)
unit = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True)  # noqa: E731
os.environ["HNU_TEXT_ENCODER"] = "minilm"
TD = 384
sv = lambda ts: unit(np.stack([hashed_text_vector(t, TD) for t in ts]))  # noqa: E731
seen = [t for p in ("train", "val") for a in ACTS for t in ins[p][a]] + [t for a in ACTS for t in v2["val"][a]] \
    + ins["val_reject"] + v2["val_reject"]
near = (sv([c["query"] for c in test]) @ sv(seen).T).max(1) > 0.92
test = [c for c, n in zip(test, near) if not n]
print(f"near-duplicates of training / validation commands excluded: {int(near.sum())}; scored cases: {len(test)} "
      f"(resolvable {sum(not c['should_abstain'] for c in test)}, reject {sum(c['should_abstain'] for c in test)})")
yt = np.array([-1 if c["should_abstain"] else ACTS.index(c["expected_action"]) for c in test])
conds = np.array([c["condition"] for c in test])
res = yt >= 0


def report(name, pred, rej):
    acc_c = {k: np.mean(pred[(conds == k) & res] == yt[(conds == k) & res]) for k in sorted(set(conds[res]))}
    return dict(name=name, top1=np.mean(pred[res] == yt[res]), cov=np.mean(~rej[res]),
                acc=np.mean(pred[res & ~rej] == yt[res & ~rej]) if (res & ~rej).any() else np.nan,
                rej=np.mean(rej[~res]), wrong=np.mean(~rej & (pred != yt)), safe=np.mean(np.where(res, (pred == yt) & ~rej, rej)),
                cond=acc_c)


rows = []
# baseline 1: keyword rules
kw = []
for c in test:
    s = c["query"].lower()
    if re.search(r"bridge|beam|plank|crossbar|under|桥|梁|横|钻|下面|底下", s):
        kw.append(0)
    elif re.search(r"\b(3|three)\b|三", s):
        kw.append(1)
    elif re.search(r"\b(5|five)\b|五", s):
        kw.append(2)
    else:
        kw.append(-1)
kw = np.array(kw)
rows.append(report("keyword rules", np.where(kw < 0, 0, kw), kw < 0))
# baseline 2: MiniLM prototype over the same instruction set (no latent), max-cosine gate chosen on val2
P = np.stack([unit(np.mean(sv(ins["train"][a] + ins["val"][a]), 0)) for a in ACTS])
Sv = sv([t for a in ACTS for t in v2["val"][a]] + v2["val_reject"]) @ P.T
yv = np.array([i for i, a in enumerate(ACTS) for _ in v2["val"][a]] + [-1] * len(v2["val_reject"]))
def best_gate(S, y):
    gs = np.quantile(S.max(1), np.linspace(0, 1, 41))
    def bal(t):
        pr, rj = S.argmax(1), S.max(1) < t
        return 0.5 * (np.mean((pr == y) & ~rj & (y >= 0)) / np.mean(y >= 0) + np.mean(rj[y < 0]))
    return max(gs, key=bal)
tg = best_gate(Sv, yv)
St = sv([c["query"] for c in test]) @ P.T
rows.append(report("MiniLM prototype (no latent)", St.argmax(1), St.max(1) < tg))
# frozen latent routers per level and seed
per = {}
for zdir in sorted(glob.glob(os.path.join(root, "seed*", "Z*"))):
    f = os.path.join(zdir, "router_query_encoder_ft2.pt")
    if not os.path.exists(f):
        continue
    prof = os.path.basename(zdir)
    ck = torch.load(os.path.join(zdir, "best_motion_latent.pt"), map_location="cpu", weights_only=False)
    model, schema, _ = build_model(ck)
    q = model.task_semantic_encoder
    q.load_state_dict(torch.load(f, map_location="cpu", weights_only=False)["state"])
    q.eval()
    g = np.load(os.path.join(zdir, "train_latents.npz"), allow_pickle=True)
    gacts = np.array([str(t).rsplit("_", 1)[0] for t in g["trial_id"]])
    K = F.normalize(torch.tensor(g["z_skill"], dtype=torch.float32), dim=-1)
    masks = [torch.tensor(gacts == a) for a in ACTS]
    goal = {a: sorted(set(str(x) for x, y in zip(g["task_goal"], gacts) if y == a)) for a in ACTS}

    def enc(texts):
        X = torch.tensor(np.stack([np.r_[hashed_text_vector(f"task_goal: {t}".strip(), TD), 1.0] for t in texts]).astype(np.float32))
        with torch.no_grad():
            return F.normalize(q(X), dim=-1)

    Rf = {a: enc(goal[a] + ins["train"][a] + ins["val"][a]) for a in ACTS}
    route = lambda E: torch.stack([(E @ K.T)[:, m].max(1).values for m in masks], 1).numpy()  # noqa: E731
    gate = lambda E: np.stack([(E @ Rf[a].T).max(1).values.numpy() for a in ACTS], 1)  # noqa: E731
    Ev = enc([t for a in ACTS for t in v2["val"][a]] + v2["val_reject"])
    Sv_, Gv = route(Ev), gate(Ev)
    gs = np.unique(np.r_[-1.0, np.quantile(Gv.max(1), np.linspace(0, 1, 41))])
    gm = np.r_[0.0, np.quantile(np.diff(np.sort(Gv, 1)[:, -2:], axis=1)[:, 0], np.linspace(0, 1, 41))]
    def dec(S, G, ts, tm):
        top = np.sort(G, 1)
        return S.argmax(1), (top[:, -1] < ts) | (top[:, -1] - top[:, -2] < tm)
    def bal(ts, tm):
        pr, rj = dec(Sv_, Gv, ts, tm)
        return 0.5 * (np.mean((pr == yv) & ~rj & (yv >= 0)) / np.mean(yv >= 0) + np.mean(rj[yv < 0]))
    best = max(((bal(a_, b_), -a_, -b_, a_, b_) for a_ in gs for b_ in gm))
    Et = enc([c["query"] for c in test])
    pred, rej = dec(route(Et), gate(Et), best[3], best[4])
    per.setdefault(prof, []).append(report(prof, pred, rej))
    json.dump([dict(id=c["id"], condition=c["condition"], expected=c["expected_action"], abstain=c["should_abstain"],
                    routed=ACTS[p], rejected=bool(r)) for c, p, r in zip(test, pred, rej)],
              open(os.path.join(zdir, "language_routing_testv2.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
for prof, rs in sorted(per.items()):
    m = {k: np.mean([r[k] for r in rs]) for k in ("top1", "cov", "acc", "rej", "wrong", "safe")}
    s = {k: np.std([r[k] for r in rs]) for k in ("top1",)}
    m["cond"] = {k: np.mean([r["cond"][k] for r in rs]) for k in rs[0]["cond"]}
    rows.append(dict(name=f"L{prof[1]} (z_sem router, {len(rs)} latent seeds)", top1_std=s["top1"], **m))
cond_keys = sorted(rows[0]["cond"])
print("\n| router | top-1 (resolvable) | " + " | ".join(cond_keys) + " | coverage | acc. of accepted | correct rejection | "
      "wrong-action activation | safe decisions |")
print("|---|---|" + "---|" * len(cond_keys) + "---|---|---|---|---|")
for r in rows:
    t1 = f"{100 * r['top1']:.1f}" + (f" ± {100 * r['top1_std']:.1f}" if "top1_std" in r else "")
    print(f"| {r['name']} | {t1} | " + " | ".join(f"{100 * r['cond'][k]:.0f}" for k in cond_keys)
          + f" | {100 * r['cov']:.1f} | {100 * r['acc']:.1f} | {100 * r['rej']:.1f} | {100 * r['wrong']:.1f} | {100 * r['safe']:.1f} |")
