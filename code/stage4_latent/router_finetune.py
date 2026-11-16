"""Deployment query encoder (router) with instruction augmentation -- identical procedure for every level L0-L4.

Diagnosis: the latents were trained with ONE task_goal command per action, so the task-only query path
(task_semantic_encoder) collapsed onto three strings and does not generalise to new phrasings (benchmark routing
44-60 %, raw MiniLM 79 %).  Fix without touching the motion latent: a copy of task_semantic_encoder is fine-tuned
on an instruction set (data_lang/instructions_v1.json: many commands per action, zh / en / code-switch / ASR-like /
implicit) with the original routing objective (supervised contrastive, query vs frozen TRAIN z_skill windows of the
same action, temperature 0.1).  z_skill, the segment z_sem and the students are unchanged.  Model selection (best
step) and the abstention threshold use the validation commands only; the Results 8.4 benchmark is scored once.
Routing rule as before: action of the nearest TRAIN z_skill window (max cosine).

  python stage4_latent/router_finetune.py <instructions.json> <benchmark.json> <latent_root> [--steps 600]
Writes <latent_root>/seed*/Z*/router_query_encoder.pt and language_routing_ft.json, and prints the summary.
"""
import argparse
import copy
import glob
import json
import os
import re
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from retrieve_skill import build_model, encode_query, hashed_text_vector  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("instructions")
ap.add_argument("benchmark")
ap.add_argument("latent_root")
ap.add_argument("--steps", type=int, default=600)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--dedup", type=float, default=0.92, help="drop train / val commands this close to a benchmark query")
a = ap.parse_args()
torch.set_num_threads(8)
unit = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True)  # noqa: E731
ins = json.load(open(a.instructions, encoding="utf-8"))
bench = json.load(open(a.benchmark, encoding="utf-8"))["cases"]
ACTS = ["CrouchedTraverse", "ThreeStepPoleRun", "FiveStepPoleRun"]


def keyword_route(q):
    """Rule baseline: bridge / beam words -> CrouchedTraverse; 3 vs 5 count words -> ThreeStepPoleRun / 2; else reject."""
    s = q.lower()
    if re.search(r"bridge|beam|plank|crossbar|under|桥|梁|横|钻|下面|底下", s):
        return "CrouchedTraverse"
    if re.search(r"\b(3|three|thre|tree)\b|三", s):
        return "ThreeStepPoleRun"
    if re.search(r"\b(5|five|fiv)\b|五根|五个|5根|5个", s):
        return "FiveStepPoleRun"
    return None


summary = {}
first = True
for zdir in sorted(glob.glob(os.path.join(a.latent_root, "seed*", "Z*"))):
    if not os.path.exists(os.path.join(zdir, "train_latents.npz")):
        continue
    prof, seed = os.path.basename(zdir), os.path.basename(os.path.dirname(zdir))
    ck = torch.load(os.path.join(zdir, "best_motion_latent.pt"), map_location="cpu", weights_only=False)
    os.environ["HNU_TEXT_ENCODER"] = str((ck.get("args", {}) or {}).get("text_encoder", "hash"))
    model, schema, _ = build_model(ck)
    model.eval()
    tdim = int(schema.get("text_dim", 128))
    branch = lambda t: np.r_[hashed_text_vector(f"task_goal: {t}".strip(), tdim), 1.0].astype(np.float32)  # noqa: E731
    g = np.load(os.path.join(zdir, "train_latents.npz"), allow_pickle=True)
    gacts = np.array([str(t).rsplit("_", 1)[0] for t in g["trial_id"]])
    K = torch.tensor(unit(g["z_skill"]), dtype=torch.float32)
    kact = torch.tensor([ACTS.index(x) for x in gacts])
    if first:  # dedup against the benchmark (once; MiniLM sentence space)
        sv = lambda ts: unit(np.stack([hashed_text_vector(t, tdim) for t in ts]))  # noqa: E731
        B = sv([c["query"] for c in bench])
        drop = 0
        for part in ("train", "val"):
            for act in ACTS:
                keep = [t for t in ins[part][act] if (sv([t]) @ B.T).max() < a.dedup]
                drop += len(ins[part][act]) - len(keep)
                ins[part][act] = keep
        rej = [t for t in ins["val_reject"] if (sv([t]) @ B.T).max() < a.dedup]
        drop += len(ins["val_reject"]) - len(rej)
        ins["val_reject"] = rej
        goals = {act: sorted(set(str(x) for x, y in zip(g["task_goal"], gacts) if y == act)) for act in ACTS}
        print(f"instruction set after dedup (removed {drop}): train "
              + ", ".join(f"{k} {len(v)}" for k, v in ins["train"].items()) + "; val "
              + ", ".join(f"{k} {len(v)}" for k, v in ins["val"].items()) + f"; val_reject {len(ins['val_reject'])}")
        # baselines (level-independent)
        res_kw = [keyword_route(c["query"]) for c in bench]
        ok = [r == c["expected_action"] for r, c in zip(res_kw, bench) if not c["should_abstain"]]
        rejok = [r is None for r, c in zip(res_kw, bench) if c["should_abstain"]]
        print(f"baseline keyword router: resolvable acc {100 * np.mean(ok):.1f}  (reject rate on resolvable "
              f"{100 * np.mean([r is None for r, c in zip(res_kw, bench) if not c['should_abstain']]):.1f}), "
              f"correct rejection {100 * np.mean(rejok):.1f}")
        P = np.stack([unit(np.mean(sv(goals[act] + ins["train"][act]), 0)) for act in ACTS])
        Qb = sv([c["query"] for c in bench])
        ok = [ACTS[i] == c["expected_action"] for i, c in zip(np.argmax(Qb @ P.T, 1), bench) if not c["should_abstain"]]
        print(f"baseline MiniLM prototype router (same instruction set): resolvable acc {100 * np.mean(ok):.1f}")
        first = False
    tr_t, tr_y = [], []
    for i, act in enumerate(ACTS):
        for t in goals[act] + ins["train"][act]:
            tr_t.append(branch(t)); tr_y.append(i)
    va_t = [branch(t) for act in ACTS for t in ins["val"][act]]
    va_y = [i for i, act in enumerate(ACTS) for _ in ins["val"][act]]
    rj_t = [branch(t) for t in ins["val_reject"]]
    Xtr, Ytr = torch.tensor(np.stack(tr_t)), torch.tensor(tr_y)
    Xva, Yva, Xrj = torch.tensor(np.stack(va_t)), torch.tensor(va_y), torch.tensor(np.stack(rj_t))
    qenc = copy.deepcopy(model.task_semantic_encoder)
    qenc.train()
    torch.manual_seed(0)
    opt = torch.optim.AdamW(qenc.parameters(), lr=a.lr, weight_decay=1e-4)

    def route(X):
        with torch.no_grad():
            s = F.normalize(qenc(X), dim=-1) @ K.T
            m, j = s.max(1)
            return kact[j], m

    def val_acc():
        qenc.eval()
        p, _ = route(Xva)
        qenc.train()
        return float((p == Yva).float().mean())

    best, best_state, best_step = val_acc(), copy.deepcopy(qenc.state_dict()), 0
    v0 = best
    for step in range(1, a.steps + 1):
        idx = torch.randint(0, len(K), (2048,))
        q = F.normalize(qenc(Xtr + 0.01 * torch.randn_like(Xtr)), dim=-1)
        logits = q @ K[idx].T / 0.1
        pos = (Ytr[:, None] == kact[idx][None, :]).float()
        logp = logits - torch.logsumexp(logits, 1, keepdim=True)
        loss = -((logp * pos).sum(1) / pos.sum(1).clamp_min(1)).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 20 == 0:
            v = val_acc()
            if v > best:
                best, best_state, best_step = v, copy.deepcopy(qenc.state_dict()), step
    qenc.load_state_dict(best_state)
    qenc.eval()
    # abstention threshold on validation: maximise balanced accuracy of {route correctly | reject}
    pv, mv = route(Xva)
    _, mr = route(Xrj)
    cands = np.unique(np.r_[mv.numpy(), mr.numpy(), -1.0])
    def bal(th):
        acc_res = float(((pv == Yva) & (mv >= th)).float().mean())
        acc_rej = float((mr < th).float().mean())
        return 0.5 * (acc_res + acc_rej)
    th = float(max(cands, key=bal))
    torch.save(dict(state=best_state, threshold=th, best_step=best_step, val_acc=best, instructions=a.instructions),
               os.path.join(zdir, "router_query_encoder.pt"))
    # final, once: benchmark
    Xb = torch.tensor(np.stack([branch(c["query"]) for c in bench]))
    pb, mb = route(Xb)
    rows, by = [], {}
    for c, p, m in zip(bench, pb.tolist(), mb.tolist()):
        rows.append(dict(id=c.get("id"), query=c["query"], expected=c.get("expected_action"),
                         abstain=bool(c["should_abstain"]), routed=ACTS[p], maxsim=float(m), rejected=bool(m < th)))
        by.setdefault(c["condition"], []).append((ACTS[p] == c.get("expected_action")) if not c["should_abstain"] else (m < th))
    json.dump(rows, open(os.path.join(zdir, "language_routing_ft.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    res = [r for r in rows if not r["abstain"]]
    s = summary.setdefault(prof, {"val0": [], "val": [], "bench": [], "cov": [], "acc_acc": [], "rej": [], "cond": []})
    s["val0"].append(v0); s["val"].append(best)
    s["bench"].append(np.mean([r["routed"] == r["expected"] for r in res]))
    s["cov"].append(np.mean([not r["rejected"] for r in res]))
    acc = [r["routed"] == r["expected"] for r in res if not r["rejected"]]
    s["acc_acc"].append(np.mean(acc) if acc else np.nan)
    s["rej"].append(np.mean([r["rejected"] for r in rows if r["abstain"]]))
    s["cond"].append({k: float(np.mean(v)) for k, v in by.items()})
    print(f"{seed}/{prof}: val {100 * v0:.1f} -> {100 * best:.1f} (step {best_step}), threshold {th:.3f}, "
          f"benchmark resolvable {100 * s['bench'][-1]:.1f}")
print("\n| level | val before | val after | benchmark resolvable top-1 | coverage (val threshold) | accuracy of accepted | "
      "correct rejection (ambiguous + OOS) | paraphrase | code-switch | ASR | implicit |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for prof, s in sorted(summary.items()):
    f = lambda k: f"{100 * np.nanmean(s[k]):.1f} ± {100 * np.nanstd(s[k]):.1f}"  # noqa: E731
    pc = lambda k: f"{100 * np.mean([c.get(k, np.nan) for c in s['cond']]):.0f}"  # noqa: E731
    print(f"| L{prof[1]} | {f('val0')} | {f('val')} | {f('bench')} | {f('cov')} | {f('acc_acc')} | {f('rej')} | "
          f"{pc('held_out_paraphrase')} | {pc('bilingual_code_switch')} | {pc('synthetic_asr_noise')} | {pc('implicit_description')} |")
