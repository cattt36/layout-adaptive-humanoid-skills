"""Router v3 (; deviation, instruction layer; identical for every level).  2 x 2 ablation on the sealed test:
  encoder : ft2 (frozen v2 query encoder)  |  ft3 (same procedure + implicit / ASR-noise training commands,
            router_finetune2.py --extra data_lang/instructions_aug_v3.json --tag ft3)
  gate    : old  (frozen: per-action best cosine to the in-scope training commands; reject if best < tau_s or
                  best - second < tau_m)
            new  (logistic gate in the same query space: similarity to the in-scope commands of each action AND to a
                  bank of out-of-scope / ambiguous commands; leave-one-out features for its own training texts)
Route is unchanged (action of the nearest TRAIN z_skill window).  Every threshold is chosen on val2 with the frozen
rule (max balanced accuracy, ties -> least rejection); the sealed test is only scored, with the frozen near-duplicate
exclusion.  A second, safety threshold rule (max val2 coverage s.t. val2 correct rejection >= 90 %, defined in
tools/router_operating_point.py) is reported for both gates.  Writes <zdir>/language_routing_test<code>.json (codes in
CODE; v3 = ft3 + new gate) and <zdir>/language_routing_testv3_ablation.json; prints per-level means.
  python stage4_latent/router_gate3.py <latent_root> [<latent_root> ...] > runs/router_v3.md"""
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ["HNU_TEXT_ENCODER"] = "minilm"
from retrieve_skill import build_model, hashed_text_vector  # noqa: E402

DL = os.path.join(HERE, "..", "data_lang")
ACTS = ["CrouchedTraverse", "ThreeStepPoleRun", "FiveStepPoleRun"]
LV = ["Z0", "Z1", "Z2", "Z3"]
TD = 384
torch.set_num_threads(8)
ins = json.load(open(f"{DL}/instructions_v1.json", encoding="utf-8"))
v2 = json.load(open(f"{DL}/instructions_val_v2.json", encoding="utf-8"))
aug = json.load(open(f"{DL}/instructions_aug_v3.json", encoding="utf-8"))
test = json.load(open(f"{DL}/router_test_v2_sealed.json", encoding="utf-8"))["cases"]
unit = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True)  # noqa: E731
sv = lambda ts: unit(np.stack([hashed_text_vector(t, TD) for t in ts]))  # noqa: E731
seen = [t for p in ("train", "val") for a in ACTS for t in ins[p][a]] + [t for a in ACTS for t in v2["val"][a]] \
    + ins["val_reject"] + v2["val_reject"]
near = (sv([c["query"] for c in test]) @ sv(seen).T).max(1) > 0.92  # frozen exclusion (aug texts were filtered)
test = [c for c, n in zip(test, near) if not n]
yt = np.array([-1 if c["should_abstain"] else ACTS.index(c["expected_action"]) for c in test])
conds = np.array([c["condition"] for c in test])
vt = [t for a in ACTS for t in v2["val"][a]] + v2["val_reject"]
yv = np.array([i for i, a in enumerate(ACTS) for _ in v2["val"][a]] + [-1] * len(v2["val_reject"]))


def pick_threshold(pred, y, score_grid, decide):
    """frozen rule: max balanced accuracy on val2, ties -> least rejection (grid ordered from least rejection)."""
    best, arg = -1.0, None
    for th in score_grid:
        rej = decide(th)
        b = 0.5 * (np.mean((pred == y) & ~rej & (y >= 0)) / np.mean(y >= 0) + np.mean(rej[y < 0]))
        if b > best + 1e-12:
            best, arg = b, th
    return arg


def pick_threshold90(pred, y, score_grid, decide, lim=0.90):
    """safety rule (defined in tools/router_operating_point.py): max val2 correct execution subject to val2
    correct rejection >= 90 %; falls back to the frozen rule if no threshold reaches it."""
    best, arg = None, None
    for th in score_grid:
        rej = decide(th)
        r = np.mean(rej[y < 0])
        if r < lim:
            continue
        key = (np.mean((pred == y) & ~rej & (y >= 0)) / np.mean(y >= 0), r)
        if best is None or key > best:
            best, arg = key, th
    return arg if arg is not None else pick_threshold(pred, y, score_grid, decide)


def run(zdir, tag):
    ck = torch.load(os.path.join(zdir, "best_motion_latent.pt"), map_location="cpu", weights_only=False)
    model, _, _ = build_model(ck)
    q = model.task_semantic_encoder
    q.load_state_dict(torch.load(os.path.join(zdir, f"router_query_encoder_{tag}.pt"), map_location="cpu", weights_only=False)["state"])
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

    route = lambda E: torch.stack([(E @ K.T)[:, m].max(1).values for m in masks], 1).numpy()  # noqa: E731
    Ev, Et = enc(vt), enc([c["query"] for c in test])
    Sv, St = route(Ev), route(Et)
    pv, pt = Sv.argmax(1), St.argmax(1)
    out = {}
    # ---- old gate (frozen)
    Rf = {a: enc(goal[a] + ins["train"][a] + ins["val"][a]) for a in ACTS}
    gate = lambda E: np.stack([(E @ Rf[a].T).max(1).values.numpy() for a in ACTS], 1)  # noqa: E731
    Gv, Gt = gate(Ev), gate(Et)
    gs = np.unique(np.r_[-1.0, np.quantile(Gv.max(1), np.linspace(0, 1, 41))])
    gm = np.r_[0.0, np.quantile(np.diff(np.sort(Gv, 1)[:, -2:], axis=1)[:, 0], np.linspace(0, 1, 41))]
    dec = lambda G, ts, tm: (np.sort(G, 1)[:, -1] < ts) | (np.sort(G, 1)[:, -1] - np.sort(G, 1)[:, -2] < tm)  # noqa: E731
    grid = sorted(((a_, b_) for a_ in gs for b_ in gm), key=lambda x: (x[0], x[1]))
    ts, tm = pick_threshold(pv, yv, grid, lambda th: dec(Gv, *th))
    out["old"] = dec(Gt, ts, tm)
    ts, tm = pick_threshold90(pv, yv, grid, lambda th: dec(Gv, *th))
    out["old90"] = dec(Gt, ts, tm)
    # ---- new gate: logistic classifier in the query space
    pos = {a: list(dict.fromkeys(goal[a] + ins["train"][a] + ins["val"][a] + aug["train"][a])) for a in ACTS}
    negs = list(dict.fromkeys(ins["val_reject"] + aug["reject"]))
    P_txt = [t for a in ACTS for t in pos[a]]
    P_act = np.array([i for i, a in enumerate(ACTS) for _ in pos[a]])
    EP, EN = enc(P_txt), enc(negs)

    def feats(E, self_pos=None, self_neg=None):
        SP = (E @ EP.T).numpy()
        SN = (E @ EN.T).numpy()
        if self_pos is not None:
            SP[np.arange(len(E)), self_pos] = -1.0
        if self_neg is not None:
            SN[np.arange(len(E)), self_neg] = -1.0
        per = [np.sort(SP[:, P_act == i], 1)[:, ::-1] for i in range(3)]  # per action, sorted desc
        top1 = np.stack([s[:, 0] for s in per], 1)
        top3 = np.stack([s[:, :3].mean(1) for s in per], 1)
        o = np.sort(top1, 1)
        k3 = top3[np.arange(len(E)), top1.argmax(1)]
        n_sorted = np.sort(SN, 1)[:, ::-1]
        R = route(E)
        r = np.sort(R, 1)
        return np.c_[o[:, -1], o[:, -1] - o[:, -2], k3, n_sorted[:, 0], n_sorted[:, :3].mean(1), o[:, -1] - n_sorted[:, 0],
                     r[:, -1], r[:, -1] - r[:, -2]]

    Xp = feats(EP, self_pos=np.arange(len(EP)))
    Xn = feats(EN, self_neg=np.arange(len(EN)))
    X = np.r_[Xp, Xn]
    y = np.r_[np.ones(len(Xp)), np.zeros(len(Xn))]
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000).fit(sc.transform(X), y)
    prob = lambda E: clf.predict_proba(sc.transform(feats(E)))[:, 1]  # noqa: E731
    qv, qt = prob(Ev), prob(Et)
    grid = np.unique(np.r_[0.0, np.quantile(qv, np.linspace(0, 1, 81))])
    th = pick_threshold(pv, yv, grid, lambda t_: qv < t_)
    out["new"] = qt < th
    th = pick_threshold90(pv, yv, grid, lambda t_: qv < t_)
    out["new90"] = qt < th
    return pt, out


def stats(pred, rej):
    res = yt >= 0
    return dict(cov=np.mean(~rej[res]), acc=np.mean(pred[res & ~rej] == yt[res & ~rej]) if (res & ~rej).any() else np.nan,
                correct=np.mean(((pred == yt) & ~rej)[res]), rej=np.mean(rej[~res]), wrong=np.mean(~rej & (pred != yt)),
                cond={k: np.mean(((pred == yt) & ~rej)[(conds == k) & res]) for k in sorted(set(conds[res]))})


VAR = [(t, g) for t in ("ft2", "ft3") for g in ("old", "old90", "new", "new90")]
# routing files for the E2E analysis (tools/expA_rank.py HNU_RTAG=<code>); v3 = ft3 + new gate, frozen threshold rule
CODE = {("ft3", "new"): "v3", ("ft3", "new90"): "v3n90", ("ft3", "old"): "v3o", ("ft3", "old90"): "v3o90",
        ("ft2", "old90"): "v2o90"}
R = {}
for root in sys.argv[1:]:
    for zdir in sorted(glob.glob(os.path.join(root, "seed*", "Z*"))):
        P = os.path.basename(zdir)
        if P not in LV or not os.path.exists(os.path.join(zdir, "router_query_encoder_ft3.pt")):
            continue
        abl = {}
        for tag in ("ft2", "ft3"):
            pred, rej = run(zdir, tag)
            for gname, r in rej.items():
                R.setdefault((P, tag, gname), []).append(stats(pred, r))
                abl[f"{tag}/{gname}"] = [dict(id=c["id"], condition=c["condition"], expected=c["expected_action"],
                                              abstain=c["should_abstain"], routed=ACTS[p], rejected=bool(x))
                                         for c, p, x in zip(test, pred, r)]
        for (tag, gname), code in CODE.items():
            json.dump(abl[f"{tag}/{gname}"], open(os.path.join(zdir, f"language_routing_test{code}.json"), "w", encoding="utf-8"),
                      ensure_ascii=False, indent=1)
        json.dump(abl, open(os.path.join(zdir, "language_routing_testv3_ablation.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(zdir, "done", file=sys.stderr)

ck = sorted(next(iter(R.values()))[0]["cond"])
n = len(next(iter(R.values())))
print(f"# 指令层 v3：编码器 × 门 × 阈值规则（封存测试集，应执行 {int((yt >= 0).sum())} 条、应拒识 {int((yt < 0).sum())} 条；"
      f"每格为 {n} 个编码器（折 × latent 种子）的平均）\n")
print("阈值都只在 val2 上选：old / new = 冻结规则（最大平衡准确率）；old90 / new90 = 安全规则（val2 拒识 ≥ 90% 下覆盖最大，"
      "tools/router_operating_point.py 已定义）。ft2 + old = 原冻结路由器。\n")
print("| 层级 | 编码器 | 门 | 覆盖率 | 正确执行 | 错误动作 | " + " | ".join(ck) + " | 越界/歧义正确拒识 |")
print("|---|---|---|---|---|---|" + "---|" * len(ck) + "---|")
for P in LV:
    for tag, gname in VAR:
        rs = R.get((P, tag, gname))
        if not rs:
            continue
        m = lambda k: 100 * np.nanmean([r[k] for r in rs])  # noqa: E731
        cc = " | ".join(f"{100 * np.mean([r['cond'][k] for r in rs]):.0f}" for k in ck)
        print(f"| L{P[1]} | {tag} | {gname} | {m('cov'):.1f} | {m('correct'):.1f} | {m('wrong'):.1f} | {cc} | {m('rej'):.1f} |")
