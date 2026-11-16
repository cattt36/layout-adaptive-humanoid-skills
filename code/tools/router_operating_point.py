"""Router operating point (diagnosis of the low E2E success; exploratory, CPU only).  Same frozen router as
stage4_latent/final_router_eval.py (query encoders, routing to the nearest TRAIN z_skill window, gate on the per-action
best cosine to the in-scope training commands with thresholds (tau_s, tau_m)), same sealed test set and near-duplicate
exclusion.  Only the rule choosing (tau_s, tau_m) on val2 changes -- thresholds are never chosen on the test set:
  frozen   : max balanced accuracy on val2 (the pre-registered rule)
  rej>=95  : max val2 coverage subject to val2 correct rejection >= 95 %
  rej>=90  : same with >= 90 %
  no gate  : never reject (coverage 100 %, no refusals)
Reports on the sealed test: coverage of in-scope commands, wrong-action activations, correct refusal of out-of-scope /
ambiguous commands; mean over latent seeds 42-44 per level.
  PYTHONPATH=data/data_code python tools/router_operating_point.py > runs/router_operating_point.md"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

H = "/root/gpufree-data/hnu_v2"
sys.path.insert(0, f"{H}/code/stage4_latent")
os.environ["HNU_TEXT_ENCODER"] = "minilm"
from retrieve_skill import build_model, hashed_text_vector  # noqa: E402

D = f"{H}/code/data_lang"
test = json.load(open(f"{D}/router_test_v2_sealed.json", encoding="utf-8"))["cases"]
ins = json.load(open(f"{D}/instructions_v1.json", encoding="utf-8"))
v2 = json.load(open(f"{D}/instructions_val_v2.json", encoding="utf-8"))
ACTS = ["CrouchedTraverse", "ThreeStepPoleRun", "FiveStepPoleRun"]
TD = 384
torch.set_num_threads(8)
unit = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True)  # noqa: E731
sv = lambda ts: unit(np.stack([hashed_text_vector(t, TD) for t in ts]))  # noqa: E731
seen = [t for p in ("train", "val") for a in ACTS for t in ins[p][a]] + [t for a in ACTS for t in v2["val"][a]] \
    + ins["val_reject"] + v2["val_reject"]
near = (sv([c["query"] for c in test]) @ sv(seen).T).max(1) > 0.92
test = [c for c, n in zip(test, near) if not n]
yt = np.array([-1 if c["should_abstain"] else ACTS.index(c["expected_action"]) for c in test])
conds = np.array([c["condition"] for c in test])
yv = np.array([i for i, a in enumerate(ACTS) for _ in v2["val"][a]] + [-1] * len(v2["val_reject"]))
RULES = ["frozen", "rej>=95", "rej>=90", "no gate"]
res = {}
for seed in (42, 43, 44):
    for P in ["Z0", "Z1", "Z2", "Z3"]:
        zdir = f"{H}/runs/latent_v5t/seed{seed}/{P}"
        ck = torch.load(f"{zdir}/best_motion_latent.pt", map_location="cpu", weights_only=False)
        model, _, _ = build_model(ck)
        q = model.task_semantic_encoder
        q.load_state_dict(torch.load(f"{zdir}/router_query_encoder_ft2.pt", map_location="cpu", weights_only=False)["state"])
        q.eval()
        g = np.load(f"{zdir}/train_latents.npz", allow_pickle=True)
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
        Sv, Gv = route(Ev), gate(Ev)
        Et = enc([c["query"] for c in test])
        St, Gt = route(Et), gate(Et)

        def dec(S, G, ts, tm):
            top = np.sort(G, 1)
            return S.argmax(1), (top[:, -1] < ts) | (top[:, -1] - top[:, -2] < tm)

        def val_stats(ts, tm):
            pr, rj = dec(Sv, Gv, ts, tm)
            cov = np.mean(~rj[yv >= 0])
            ok = np.mean((pr == yv) & ~rj & (yv >= 0)) / np.mean(yv >= 0)
            return cov, ok, np.mean(rj[yv < 0])

        gs = np.unique(np.r_[-1.0, np.quantile(Gv.max(1), np.linspace(0, 1, 41))])
        gm = np.r_[0.0, np.quantile(np.diff(np.sort(Gv, 1)[:, -2:], axis=1)[:, 0], np.linspace(0, 1, 41))]
        grid = [(a, b, *val_stats(a, b)) for a in gs for b in gm]
        pick = {"frozen": max(grid, key=lambda r: (0.5 * (r[3] + r[4]), -r[0], -r[1]))}
        for name, lim in (("rej>=95", 0.95), ("rej>=90", 0.90)):
            ok = [r for r in grid if r[4] >= lim]
            pick[name] = max(ok, key=lambda r: (r[3], r[4])) if ok else pick["frozen"]
        pick["no gate"] = (-9.0, -9.0)
        for name in RULES:
            pr, rj = dec(St, Gt, pick[name][0], pick[name][1])
            r = dict(cov=np.mean(~rj[yt >= 0]), correct=np.mean((pr == yt)[yt >= 0] & ~rj[yt >= 0]),
                     wrong=np.mean(~rj[yt >= 0] & (pr != yt)[yt >= 0]), refuse=np.mean(rj[yt < 0]),
                     cond={k: np.mean(~rj[(conds == k) & (yt >= 0)] & (pr == yt)[(conds == k) & (yt >= 0)])
                           for k in sorted(set(conds[yt >= 0]))})
            res.setdefault((P, name), []).append(r)
        print(seed, P, "done", file=sys.stderr)

ck = sorted(res[("Z0", "frozen")][0]["cond"])
print(f"# 路由工作点（封存测试集；应执行 {int((yt >= 0).sum())} 条，应拒识 {int((yt < 0).sum())} 条；阈值只在 val2 上选；3 个 latent 种子平均）\n")
print("| 层级 | 阈值规则 | 应执行指令：正确执行 % | 其中误拒 % | 错误动作 % | " + " | ".join(ck) + " | 越界/歧义指令正确拒识 % |")
print("|---|---|---|---|---|" + "---|" * len(ck) + "---|")
for P in ["Z0", "Z1", "Z2", "Z3"]:
    for name in RULES:
        rs = res[(P, name)]
        m = lambda k: 100 * np.mean([r[k] for r in rs])  # noqa: E731
        cc = " | ".join(f"{100 * np.mean([r['cond'][k] for r in rs]):.0f}" for k in ck)
        print(f"| L{P[1]} | {name} | {m('correct'):.1f} | {100 - m('cov'):.1f} | {m('wrong'):.1f} | {cc} | {m('refuse'):.1f} |")
