"""Experiment A (full chain, runs/preregistration_expA_impl.md), step "retrieve z_skill": for every in-scope sealed
test instruction and every level / latent seed, rank the LIBRARY clips (fold-1 training clips) of the routed action by
the maximum cosine between the instruction's query embedding (router v2 query encoder, as stage4_latent/
final_router_eval.py) and the clip's z_skill windows (train_latents.npz).  Routing decisions are taken unchanged from
the frozen router output (language_routing_testv2.json).  CPU only.
  python expA_rank.py -> runs/expA_rank.json
(E2E on all folds): python expA_rank.py <fold 2-4> -> runs/expA_rank_f<k>.json, fold encoders
runs/latent_cv_f<k> (routers made by tools/e2e_router_folds.sh), library = the fold's training clips with geometry
(data/z_cand_plan_f<k>/pairs.json), levels L0-L3."""
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

TD = 384
FOLD = int(sys.argv[1]) if len(sys.argv) > 1 else 1
ROOT = f"{H}/runs/latent_v5t" if FOLD == 1 else f"{H}/runs/latent_cv_f{FOLD}"
# HNU_RTAG=v3 -> router v3 (stage4_latent/router_gate3.py: ft3 encoder + new gate), output *_v3.json
RTAG = os.environ.get("HNU_RTAG", "")
ENC = "ft3" if RTAG.startswith(("v3", "v4")) else "ft2"  # codes: router_gate3.py CODE; v4 = router_gate4.py
ROUTES = f"language_routing_test{RTAG or 'v2'}.json"
OUT = (f"{H}/runs/expA_rank.json" if FOLD == 1 else f"{H}/runs/expA_rank_f{FOLD}.json").replace(".json", f"_{RTAG}.json" if RTAG else ".json")
LEVELS = ["Z0", "Z1", "Z2", "Z3", "Z4"] if FOLD == 1 else ["Z0", "Z1", "Z2", "Z3"]
cases = {c["id"]: c for c in json.load(open(f"{H}/code/data_lang/router_test_v2_sealed.json", encoding="utf-8"))["cases"]}
lib = json.load(open(f"{H}/data/z_cand_f1/cands.json" if FOLD == 1 else f"{H}/data/z_cand_plan_f{FOLD}/pairs.json"))["library"]
torch.set_num_threads(8)
out = {}
for seed in (42, 43, 44):
    for P in LEVELS:
        zdir = f"{ROOT}/seed{seed}/{P}"
        if not os.path.exists(f"{zdir}/{ROUTES}"):
            continue
        ck = torch.load(f"{zdir}/best_motion_latent.pt", map_location="cpu", weights_only=False)
        model, _, _ = build_model(ck)
        q = model.task_semantic_encoder
        q.load_state_dict(torch.load(f"{zdir}/router_query_encoder_{ENC}.pt", map_location="cpu", weights_only=False)["state"])
        q.eval()
        g = np.load(f"{zdir}/train_latents.npz", allow_pickle=True)
        trial = np.array([str(t) for t in g["trial_id"]])
        Kz = F.normalize(torch.tensor(g["z_skill"], dtype=torch.float32), dim=-1)
        routes = json.load(open(f"{zdir}/{ROUTES}", encoding="utf-8"))
        ins = [r for r in routes if not r["abstain"]]
        X = torch.tensor(np.stack([np.r_[hashed_text_vector(f"task_goal: {cases[r['id']]['query']}".strip(), TD), 1.0]
                                   for r in ins]).astype(np.float32))
        with torch.no_grad():
            E = F.normalize(q(X), dim=-1)
        S = (E @ Kz.T).numpy()
        rows = []
        for i, r in enumerate(ins):
            a = r["routed"]
            sc = {c: float(S[i, trial == c].max()) for c in lib[a] if (trial == c).any()}
            rows.append(dict(id=r["id"], condition=r["condition"], expected=r["expected"], routed=a,
                             rejected=bool(r.get("rejected", False)), ranking=sorted(sc, key=sc.get, reverse=True),
                             scores=sc))
        out[f"{seed}/{P}"] = rows
        print(seed, P, len(rows), "in-scope; library clips scored:", {a: sum(1 for c in lib[a] if (trial == c).any()) for a in lib})
json.dump(out, open(OUT, "w"), indent=1)
