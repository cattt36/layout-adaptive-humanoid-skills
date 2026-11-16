"""Routing diagnosis on the language OOD benchmark (resolvable cases: top-1 action accuracy), per level and latent seed.
  a  current      : task-only query z_sem -> nearest TRAIN z_skill (cross space)
  b  zsem-gallery : query z_sem -> nearest TRAIN segment z_sem
  c  prototype    : query z_sem -> nearest per-action centroid of the TRAIN task_goal commands encoded by the same
                    task-only query encoder (same space as the query)
  d  MiniLM       : raw multilingual MiniLM embedding -> nearest per-action centroid of the TRAIN task_goal commands
                    (no latent; "generic text-embedding router" baseline)
Also prints the distinct TRAIN task_goal commands per action.
  python tools/route_diag.py <benchmark.json> <latent_root>
"""
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "stage4_latent"))
from retrieve_skill import build_model, encode_query  # noqa: E402

bench = [c for c in json.load(open(sys.argv[1], encoding="utf-8"))["cases"] if not c["should_abstain"]]
root = sys.argv[2]
unit = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True)  # noqa: E731
shown = False
res = defaultdict(list)
st = None
for zdir in sorted(glob.glob(os.path.join(root, "seed*", "Z*"))):
    if not os.path.exists(os.path.join(zdir, "train_latents.npz")):
        continue
    prof = os.path.basename(zdir)
    ck = torch.load(os.path.join(zdir, "best_motion_latent.pt"), map_location="cpu", weights_only=False)
    os.environ["HNU_TEXT_ENCODER"] = str((ck.get("args", {}) or {}).get("text_encoder", "hash"))
    model, schema, _ = build_model(ck)
    g = np.load(os.path.join(zdir, "train_latents.npz"), allow_pickle=True)
    acts = np.array([str(t).rsplit("_", 1)[0] for t in g["trial_id"]])
    goals = np.array([str(x) for x in g["task_goal"]]) if "task_goal" in g.files else None
    if not shown:
        print("train_latents keys:", g.files)
        if goals is not None:
            for a in sorted(set(acts)):
                u = sorted(set(goals[acts == a]))
                print(f"  {a}: {len(u)} distinct task_goal commands, e.g. {u[:3]}")
        shown = True
    Zs, Zm = unit(g["z_skill"]), unit(g["z_sem"]) if "z_sem" in g.files else None
    Q = unit(np.stack([encode_query(model, schema, c["query"], query_mode="task") for c in bench]))
    exp = np.array([c["expected_action"] for c in bench])
    res[(prof, "a current")].append(np.mean(acts[np.argmax(Q @ Zs.T, 1)] == exp))
    if Zm is not None:
        res[(prof, "b zsem-gallery")].append(np.mean(acts[np.argmax(Q @ Zm.T, 1)] == exp))
    if goals is not None:
        names = sorted(set(acts))
        cent = []
        for a in names:
            u = sorted(set(goals[acts == a]))
            cent.append(unit(np.mean(unit(np.stack([encode_query(model, schema, t, query_mode="task") for t in u])), 0)))
        res[(prof, "c prototype")].append(np.mean(np.array(names)[np.argmax(Q @ np.stack(cent).T, 1)] == exp))
        if st is None and os.environ["HNU_TEXT_ENCODER"] == "minilm":
            from retrieve_skill import hashed_text_vector  # MiniLM sentence vector (same front-end as the latents)
            st = True
            enc = lambda ts: unit(np.stack([hashed_text_vector(t, int(schema.get("text_dim", 128))) for t in ts]))  # noqa: E731
            Qm = enc([c["query"] for c in bench])
            cm = np.stack([unit(np.mean(enc(sorted(set(goals[acts == a]))), 0)) for a in names])
            res[("--", "d MiniLM")].append(np.mean(np.array(names)[np.argmax(Qm @ cm.T, 1)] == exp))
print("\nresolvable top-1 routing accuracy (mean over latent seeds):")
for (p, m), v in sorted(res.items()):
    print(f"  {p:3s} {m:15s} {100 * np.mean(v):5.1f}  (n={len(v)})")
