"""Router v4 gate (; deviation, instruction layer; identical for every level).  Encoder = ft3 (router v3).
v3 showed that the logistic gate raises coverage strongly but its threshold could not be calibrated on val2 (20 reject
commands): correct rejection fell to 73-83 % on the sealed test.  v4 adds an LLM-written validation set v3
(data_lang/instructions_val_v3.json, ~120 in-scope + ~150 out-of-scope / ambiguous) and more gate training negatives
(instructions_aug_v3b.json).  Rules fixed BEFORE scoring the sealed test:
  threshold : per encoder, max val3 correct execution subject to val3 correct rejection >= 90 % (fallback: max balanced
              accuracy)
  gate type : ONE type for all levels = max mean val3 correct execution (at its threshold) over levels Z0-Z3, folds and
              latent seeds; candidates
                old   : frozen gate (per-action best cosine to the in-scope commands; best < ts or margin < tm)
                lr8   : logistic on 8 similarity features (as router_gate3 "new"), leave-one-out for its training texts
                lremb : logistic on the 8 features + the query embedding itself
                knn   : mean cosine to the 5 nearest in-scope commands minus to the 5 nearest out-of-scope commands
Route unchanged (nearest TRAIN z_skill window).  Gate training: positives = goal + instructions_v1 train/val +
aug_v3 train; negatives = instructions_v1 val_reject + aug_v3 reject + aug_v3b reject.  Writes
<zdir>/language_routing_testv4.json (chosen gate) and <zdir>/language_routing_testv4_ablation.json.
  python stage4_latent/router_gate4.py <latent_root> [...] > runs/router_v4.md"""
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
GATES = ["old", "lr8", "lremb", "knn"]
TD = 384
torch.set_num_threads(8)
ins = json.load(open(f"{DL}/instructions_v1.json", encoding="utf-8"))
v2 = json.load(open(f"{DL}/instructions_val_v2.json", encoding="utf-8"))
v3 = json.load(open(f"{DL}/instructions_val_v3.json", encoding="utf-8"))
aug = json.load(open(f"{DL}/instructions_aug_v3.json", encoding="utf-8"))
augb = json.load(open(f"{DL}/instructions_aug_v3b.json", encoding="utf-8"))
test = json.load(open(f"{DL}/router_test_v2_sealed.json", encoding="utf-8"))["cases"]
unit = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True)  # noqa: E731
sv = lambda ts: unit(np.stack([hashed_text_vector(t, TD) for t in ts]))  # noqa: E731
seen = [t for p in ("train", "val") for a in ACTS for t in ins[p][a]] + [t for a in ACTS for t in v2["val"][a]] \
    + ins["val_reject"] + v2["val_reject"]
near = (sv([c["query"] for c in test]) @ sv(seen).T).max(1) > 0.92  # frozen exclusion (new texts were filtered)
test = [c for c, n in zip(test, near) if not n]
yt = np.array([-1 if c["should_abstain"] else ACTS.index(c["expected_action"]) for c in test])
conds = np.array([c["condition"] for c in test])
vt = [t for a in ACTS for t in v3["val"][a]] + v3["val_reject"]
yv = np.array([i for i, a in enumerate(ACTS) for _ in v3["val"][a]] + [-1] * len(v3["val_reject"]))
vcat = np.array(["in_scope"] * int((yv >= 0).sum()) + v3["val_reject_category"])


def val_obj(pred, rej, y):
    return np.mean((pred == y) & ~rej & (y >= 0)) / np.mean(y >= 0), np.mean(rej[y < 0])


def pick(pred, y, grid, decide, lim=0.90):
    best, arg = None, None
    for th in grid:
        ex, rj = val_obj(pred, decide(th), y)
        if rj >= lim and (best is None or (ex, rj) > best):
            best, arg = (ex, rj), th
    if arg is None:  # fallback: balanced accuracy, ties -> least rejection (grid ordered from least rejection)
        b0 = -1
        for th in grid:
            ex, rj = val_obj(pred, decide(th), y)
            if 0.5 * (ex + rj) > b0 + 1e-12:
                b0, arg = 0.5 * (ex + rj), th
    return arg


def run(zdir):
    ck = torch.load(os.path.join(zdir, "best_motion_latent.pt"), map_location="cpu", weights_only=False)
    model, _, _ = build_model(ck)
    q = model.task_semantic_encoder
    q.load_state_dict(torch.load(os.path.join(zdir, "router_query_encoder_ft3.pt"), map_location="cpu", weights_only=False)["state"])
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
    pv, pt = route(Ev).argmax(1), route(Et).argmax(1)
    out = {}
    # old gate
    Rf = {a: enc(goal[a] + ins["train"][a] + ins["val"][a]) for a in ACTS}
    gate = lambda E: np.stack([(E @ Rf[a].T).max(1).values.numpy() for a in ACTS], 1)  # noqa: E731
    Gv, Gt = gate(Ev), gate(Et)
    gs = np.unique(np.r_[-1.0, np.quantile(Gv.max(1), np.linspace(0, 1, 41))])
    gm = np.r_[0.0, np.quantile(np.diff(np.sort(Gv, 1)[:, -2:], axis=1)[:, 0], np.linspace(0, 1, 41))]
    dec = lambda G, ts, tm: (np.sort(G, 1)[:, -1] < ts) | (np.sort(G, 1)[:, -1] - np.sort(G, 1)[:, -2] < tm)  # noqa: E731
    th = pick(pv, yv, sorted((a_, b_) for a_ in gs for b_ in gm), lambda x: dec(Gv, *x))
    out["old"] = (dec(Gv, *th), dec(Gt, *th))
    # learned gates
    pos = {a: list(dict.fromkeys(goal[a] + ins["train"][a] + ins["val"][a] + aug["train"][a])) for a in ACTS}
    negs = list(dict.fromkeys(ins["val_reject"] + aug["reject"] + augb["reject"]))
    P_act = np.array([i for i, a in enumerate(ACTS) for _ in pos[a]])
    EP, EN = enc([t for a in ACTS for t in pos[a]]), enc(negs)

    def sims(E, self_pos=None, self_neg=None):
        SP, SN = (E @ EP.T).numpy(), (E @ EN.T).numpy()
        if self_pos is not None:
            SP[np.arange(len(E)), self_pos] = -1.0
        if self_neg is not None:
            SN[np.arange(len(E)), self_neg] = -1.0
        return SP, SN

    def feats(E, SP, SN):
        per = [np.sort(SP[:, P_act == i], 1)[:, ::-1] for i in range(3)]
        top1 = np.stack([s[:, 0] for s in per], 1)
        top3 = np.stack([s[:, :3].mean(1) for s in per], 1)
        o = np.sort(top1, 1)
        k3 = top3[np.arange(len(E)), top1.argmax(1)]
        ns = np.sort(SN, 1)[:, ::-1]
        r = np.sort(route(E), 1)
        return np.c_[o[:, -1], o[:, -1] - o[:, -2], k3, ns[:, 0], ns[:, :3].mean(1), o[:, -1] - ns[:, 0], r[:, -1], r[:, -1] - r[:, -2]]

    def knn(SP, SN, k=5):
        return np.sort(SP, 1)[:, -k:].mean(1) - np.sort(SN, 1)[:, -k:].mean(1)

    trP, trN = sims(EP, self_pos=np.arange(len(EP))), sims(EN, self_neg=np.arange(len(EN)))
    X8 = np.r_[feats(EP, *trP), feats(EN, *trN)]
    Xe = np.r_[EP.numpy(), EN.numpy()]
    y = np.r_[np.ones(len(EP)), np.zeros(len(EN))]
    sv_, st_ = sims(Ev), sims(Et)
    F8v, F8t = feats(Ev, *sv_), feats(Et, *st_)
    scores = {}
    for name, (Xtr, Xv, Xt) in {"lr8": (X8, F8v, F8t), "lremb": (np.c_[X8, Xe], np.c_[F8v, Ev.numpy()], np.c_[F8t, Et.numpy()])}.items():
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000).fit(sc.transform(Xtr), y)
        scores[name] = (clf.predict_proba(sc.transform(Xv))[:, 1], clf.predict_proba(sc.transform(Xt))[:, 1])
    scores["knn"] = (knn(*sv_), knn(*st_))
    for name, (sv1, st1) in scores.items():
        grid = np.unique(np.r_[sv1.min() - 1, np.quantile(sv1, np.linspace(0, 1, 101))])
        t_ = pick(pv, yv, grid, lambda x: sv1 < x)
        out[name] = (sv1 < t_, st1 < t_)
    return pv, pt, out


def tstats(pred, rej):
    res = yt >= 0
    return dict(cov=np.mean(~rej[res]), correct=np.mean(((pred == yt) & ~rej)[res]), rej=np.mean(rej[~res]),
                wrong=np.mean(~rej & (pred != yt)), cond={k: np.mean(((pred == yt) & ~rej)[(conds == k) & res]) for k in sorted(set(conds[res]))})


def vstats(pred, rej):
    ex, rj = val_obj(pred, rej, yv)
    return dict(ex=ex, rej=rj, cat={c: np.mean(rej[vcat == c]) for c in sorted(set(vcat)) if c != "in_scope"})


RES, DEC = {}, {}
for root in sys.argv[1:]:
    for zdir in sorted(glob.glob(os.path.join(root, "seed*", "Z*"))):
        P = os.path.basename(zdir)
        if P not in LV or not os.path.exists(os.path.join(zdir, "router_query_encoder_ft3.pt")):
            continue
        pv, pt, out = run(zdir)
        for gname, (rv, rt) in out.items():
            RES.setdefault((P, gname), []).append((vstats(pv, rv), tstats(pt, rt)))
        DEC[zdir] = (pt, {gname: rt for gname, (_, rt) in out.items()})
        print(zdir, "done", file=sys.stderr)

choice = max(GATES, key=lambda gn: (np.mean([v["ex"] for P in LV for v, _ in RES[(P, gn)]]), -GATES.index(gn)))
for zdir, (pt, rts) in DEC.items():
    rows = lambda rt: [dict(id=c["id"], condition=c["condition"], expected=c["expected_action"], abstain=c["should_abstain"],  # noqa: E731
                            routed=ACTS[p], rejected=bool(x)) for c, p, x in zip(test, pt, rt)]
    json.dump(rows(rts[choice]), open(os.path.join(zdir, "language_routing_testv4.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump({gn: rows(rt) for gn, rt in rts.items()}, open(os.path.join(zdir, "language_routing_testv4_ablation.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

n = len(RES[("Z0", "old")])
print(f"# 指令层 v4：在 LLM 编写的验证集 val3 上选门和阈值（编码器 ft3；每格 {n} 个编码器 = 折 × latent 种子的平均）\n")
print(f"val3：应执行 {int((yv >= 0).sum())} 条，应拒识 {int((yv < 0).sum())} 条；封存测试集：应执行 {int((yt >= 0).sum())} 条，"
      f"应拒识 {int((yt < 0).sum())} 条。阈值规则：val3 拒识 ≥ 90% 下正确执行最大。门的种类按 val3 正确执行（各层级平均）"
      f"预先规定只选一个：**{choice}**。\n")
print("## 1. val3（选择用）\n")
cats = sorted(next(iter(RES.values()))[0][0]["cat"])
print("| 层级 | 门 | 正确执行 | 拒识（合计） | " + " | ".join(cats) + " |")
print("|---|---|---|---|" + "---|" * len(cats))
for P in LV:
    for gn in GATES:
        vs = [v for v, _ in RES[(P, gn)]]
        print(f"| L{P[1]} | {gn} | {100 * np.mean([v['ex'] for v in vs]):.1f} | {100 * np.mean([v['rej'] for v in vs]):.1f} | "
              + " | ".join(f"{100 * np.mean([v['cat'][c] for v in vs]):.0f}" for c in cats) + " |")
print("\n## 2. 封存测试集（只打分）\n")
ck = sorted(next(iter(RES.values()))[0][1]["cond"])
print("| 层级 | 门 | 覆盖率 | 正确执行 | 错误动作 | " + " | ".join(ck) + " | 越界/歧义正确拒识 |")
print("|---|---|---|---|---|" + "---|" * len(ck) + "---|")
for P in LV:
    for gn in GATES:
        ts = [t for _, t in RES[(P, gn)]]
        m = lambda k: 100 * np.mean([t[k] for t in ts])  # noqa: E731
        mark = " ★" if gn == choice else ""
        print(f"| L{P[1]} | {gn}{mark} | {m('cov'):.1f} | {m('correct'):.1f} | {m('wrong'):.1f} | "
              + " | ".join(f"{100 * np.mean([t['cond'][k] for t in ts]):.0f}" for k in ck) + f" | {m('rej'):.1f} |")
print("\n参考（runs/router_v3.md）：原冻结路由器 ft2 + old（val2 规则）覆盖率 79–84%、拒识 89–91%；ft3 + old 覆盖率 82–84%、拒识 91–93%。")
