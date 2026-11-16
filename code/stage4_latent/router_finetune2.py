"""Router v2 -- same procedure for every level, two protocol corrections over router_finetune.py (v1):

1  Training length.  v1 selected the best step on a validation set that saturated at 100 % within ~20 steps, so the
   tie rule ("earliest best step") stopped levels with a good start (L4) almost untrained.  v2 trains every model for
   the same grid of steps and picks ONE shared step count: the maximum of the mean validation accuracy over all
   levels and seeds on a harder, independently written validation set (data_lang/instructions_val_v2.json).
2  Level-owned language.  Each level additionally trains on the texts its own supervision contains -- exactly the
   query augmentation of the latent training (motion_latent_z.py, v4): L2 robot skill atoms, L4 human captions,
   human control hints and human skill atoms (L0 / L1 / L3 own none).  All levels share the common instruction set
   (instructions_v1 train + val).

Unchanged: z_skill / segment z_sem / students frozen; copy of task_semantic_encoder trained with the routing
objective (supervised contrastive vs TRAIN z_skill of the same action, T = 0.1); routing = action of the nearest
TRAIN z_skill; abstention threshold from validation; the Results 8.4 benchmark is scored once.

  python stage4_latent/router_finetune2.py <instructions_v1.json> <val_v2.json> <benchmark.json> <latent_root>
"""
import argparse
import copy
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from retrieve_skill import build_model, hashed_text_vector  # noqa: E402
import motion_latent_z as mlz  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("instructions")
ap.add_argument("val2")
ap.add_argument("benchmark")
ap.add_argument("latent_root")
ap.add_argument("--steps", type=int, default=600)
ap.add_argument("--every", type=int, default=20)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--dedup", type=float, default=0.92)
ap.add_argument("--tag", default="ft2")
ap.add_argument("--extra", default="", help="v3: extra TRAINING commands, data_lang/instructions_aug_v3.json")
a = ap.parse_args()
torch.set_num_threads(8)
unit = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True)  # noqa: E731
ACTS = ["CrouchedTraverse", "ThreeStepPoleRun", "FiveStepPoleRun"]
ins = json.load(open(a.instructions, encoding="utf-8"))
v2 = json.load(open(a.val2, encoding="utf-8"))
bench = json.load(open(a.benchmark, encoding="utf-8"))["cases"]


def own_texts(ck, trials, profile, task_goal_of):
    """Level-owned query augmentation texts of the TRAIN trials (motion_latent_z.py, v4 query augmentation)."""
    cfg = mlz.ablation_config(profile)
    if not (cfg.get("robot_semantics") or cfg.get("human_semantics")):
        return {act: [] for act in ACTS}
    args = ck["args"]
    docs = mlz.load_dataset_documents(args["datasets"], Path(args["json_root"]),
                                      Path(args["retargeted_json_root"]) if args.get("retargeted_json_root") else None)
    out = {act: set() for act in ACTS}
    for tid in trials:
        doc = docs.get(tid)
        if doc is None:
            continue
        act = tid.rsplit("_", 1)[0]
        task_text = task_goal_of[act]
        for seg in doc.get("segments", []):
            if cfg.get("robot_semantics"):
                rsem = mlz.nested_get(seg, ("Retargeted_Robot_Metrics", "robot_semantics"), {}) or {}
                ratoms = [str(x).replace("_", " ") for x in mlz.text_items(rsem.get("skill_atoms"))]
                if ratoms:
                    out[act].add(f"{task_text} . {', '.join(ratoms)}")
            if cfg.get("human_semantics"):
                h = mlz.nested_get(seg, ("Human_Metrics", "human_semantics"), {}) or {}
                if isinstance(h, dict):
                    for item in list(mlz.text_items(h.get("short_caption_en")))[:1] + \
                            list(mlz.text_items(h.get("human_derived_control_hints")))[:2]:
                        if item and not item.lower().startswith(("flight phase with no", "prepared phase with no",
                                                                 "unknown phase with no")):
                            out[act].add(item)
                    hatoms = [str(x).replace("_", " ") for x in mlz.text_items(h.get("skill_atoms"))]
                    if hatoms:
                        out[act].add(f"{task_text} . {', '.join(hatoms)}")
    return {act: sorted(v) for act, v in out.items()}


models, first = [], True
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
    goal_of = {act: sorted(set(str(x) for x, y in zip(g["task_goal"], gacts) if y == act))[0] for act in ACTS}
    if first:
        sv = lambda ts: unit(np.stack([hashed_text_vector(t, tdim) for t in ts]))  # noqa: E731
        B = sv([c["query"] for c in bench])
        V1 = sv([t for p in ("train", "val") for act in ACTS for t in ins[p][act]])
        keep = lambda ts, *refs: [t for t in ts if all((sv([t]) @ R.T).max() < a.dedup for R in refs)]  # noqa: E731
        common = {act: keep(ins["train"][act] + ins["val"][act], B) for act in ACTS}
        if a.extra:  # already filtered against val2 / benchmark / sealed test by tools/make_instructions_aug_v3.py
            ext = json.load(open(a.extra, encoding="utf-8"))["train"]
            common = {act: common[act] + ext[act] for act in ACTS}
        val = {act: keep(v2["val"][act], B, V1) for act in ACTS}
        rej = keep(v2["val_reject"], B, V1)
        print("common train " + ", ".join(f"{k} {len(v)}" for k, v in common.items()) + "; val2 "
              + ", ".join(f"{k} {len(v)}" for k, v in val.items()) + f"; val2 reject {len(rej)}")
        first = False
    own = own_texts(ck, sorted(set(str(t) for t in g["trial_id"])), prof, goal_of)
    tr_t, tr_y = [], []
    for i, act in enumerate(ACTS):
        for t in [goal_of[act]] + common[act] + own[act]:
            tr_t.append(branch(t)); tr_y.append(i)
    Xtr, Ytr = torch.tensor(np.stack(tr_t)), torch.tensor(tr_y)
    Xva = torch.tensor(np.stack([branch(t) for act in ACTS for t in val[act]]))
    Yva = torch.tensor([i for i, act in enumerate(ACTS) for _ in val[act]])
    Xrj = torch.tensor(np.stack([branch(t) for t in rej]))
    Xb = torch.tensor(np.stack([branch(c["query"]) for c in bench]))
    K = torch.tensor(unit(g["z_skill"]), dtype=torch.float32)
    kact = torch.tensor([ACTS.index(x) for x in gacts])
    qenc = copy.deepcopy(model.task_semantic_encoder)
    torch.manual_seed(0)
    opt = torch.optim.AdamW(qenc.parameters(), lr=a.lr, weight_decay=1e-4)

    def route(X):
        qenc.eval()
        with torch.no_grad():
            m, j = (F.normalize(qenc(X), dim=-1) @ K.T).max(1)
        qenc.train()
        return kact[j], m

    curve, states = {}, {}
    for step in range(0, a.steps + 1):
        if step % a.every == 0:
            p, _ = route(Xva)
            curve[step] = float((p == Yva).float().mean())
            states[step] = copy.deepcopy(qenc.state_dict())
        if step == a.steps:
            break
        idx = torch.randint(0, len(K), (2048,))
        q = F.normalize(qenc(Xtr + 0.01 * torch.randn_like(Xtr)), dim=-1)
        logits = q @ K[idx].T / 0.1
        pos = (Ytr[:, None] == kact[idx][None, :]).float()
        logp = logits - torch.logsumexp(logits, 1, keepdim=True)
        loss = -((logp * pos).sum(1) / pos.sum(1).clamp_min(1)).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    models.append(dict(prof=prof, seed=seed, zdir=zdir, curve=curve, states=states, qenc=qenc, K=K, kact=kact,
                       Xva=Xva, Yva=Yva, Xrj=Xrj, Xb=Xb, n_own=sum(len(v) for v in own.values())))
    print(f"{seed}/{prof}: own texts {models[-1]['n_own']}, val2 acc at 0/100/300/600 steps: "
          + " / ".join(f"{100 * curve.get(s, np.nan):.0f}" for s in (0, 100, 300, a.steps)))

# ONE shared training length for every level: argmax of the mean validation curve (ties -> longer training)
steps = sorted(models[0]["curve"])
mean_curve = {s: np.mean([m["curve"][s] for m in models]) for s in steps}
best_val = max(mean_curve.values())
S = max(s for s in steps if mean_curve[s] >= best_val - 1e-9)
print(f"\nshared training length: {S} steps (mean val2 acc {100 * mean_curve[S]:.1f}; "
      + ", ".join(f"{s}: {100 * mean_curve[s]:.0f}" for s in steps[:: max(1, len(steps) // 8)]) + ")")
summary = {}
for m in models:
    q = m["qenc"]
    q.load_state_dict(m["states"][S])
    q.eval()
    with torch.no_grad():
        def route(X):
            mm, j = (F.normalize(q(X), dim=-1) @ m["K"].T).max(1)
            return m["kact"][j], mm
        pv, mv = route(m["Xva"])
        _, mr = route(m["Xrj"])
        pb, mb = route(m["Xb"])
    cands = np.unique(np.r_[mv.numpy(), mr.numpy(), -1.0])
    bal = lambda th: 0.5 * (float(((pv == m["Yva"]) & (mv >= th)).float().mean()) + float((mr < th).float().mean()))  # noqa: E731
    th = float(max(cands, key=bal))
    torch.save(dict(state=m["states"][S], threshold=th, steps=S, protocol="router_finetune2"),
               os.path.join(m["zdir"], f"router_query_encoder_{a.tag}.pt"))
    rows, by = [], {}
    for c, p, s in zip(bench, pb.tolist(), mb.tolist()):
        rows.append(dict(id=c.get("id"), query=c["query"], expected=c.get("expected_action"), abstain=bool(c["should_abstain"]),
                         routed=ACTS[p], maxsim=float(s), rejected=bool(s < th)))
        by.setdefault(c["condition"], []).append((ACTS[p] == c.get("expected_action")) if not c["should_abstain"] else (s < th))
    json.dump(rows, open(os.path.join(m["zdir"], f"language_routing_{a.tag}.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    res = [r for r in rows if not r["abstain"]]
    s_ = summary.setdefault(m["prof"], {k: [] for k in ("val", "bench", "cov", "acc", "rej", "cond", "own")})
    s_["val"].append(m["curve"][S]); s_["own"].append(m["n_own"])
    s_["bench"].append(np.mean([r["routed"] == r["expected"] for r in res]))
    s_["cov"].append(np.mean([not r["rejected"] for r in res]))
    acc = [r["routed"] == r["expected"] for r in res if not r["rejected"]]
    s_["acc"].append(np.mean(acc) if acc else np.nan)
    s_["rej"].append(np.mean([r["rejected"] for r in rows if r["abstain"]]))
    s_["cond"].append({k: float(np.mean(v)) for k, v in by.items()})
print("\n| level | own texts | val2 acc | benchmark resolvable top-1 | coverage | accuracy of accepted | correct rejection | "
      "paraphrase | code-switch | ASR | implicit |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for prof, s in sorted(summary.items()):
    f = lambda k: f"{100 * np.nanmean(s[k]):.1f} ± {100 * np.nanstd(s[k]):.1f}"  # noqa: E731
    pc = lambda k: f"{100 * np.mean([c.get(k, np.nan) for c in s['cond']]):.0f}"  # noqa: E731
    print(f"| L{prof[1]} | {int(np.mean(s['own']))} | {f('val')} | {f('bench')} | {f('cov')} | {f('acc')} | {f('rej')} | "
          f"{pc('held_out_paraphrase')} | {pc('bilingual_code_switch')} | {pc('synthetic_asr_noise')} | {pc('implicit_description')} |")
