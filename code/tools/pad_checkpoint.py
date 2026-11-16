"""Warm-start an expert on a larger observation (perception terms appended at the end of the actor / critic groups):
zero-pad the first layer's input columns, extend the observation normalisers (mean 0, var 1) and the Adam state.
The padded policy is identical to the original until it learns to use the new inputs.

  python pad_checkpoint.py <in.pt> <out.pt> <n_new_actor> <n_new_critic>
"""
import sys

import torch

src, dst, na, nc = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
ck = torch.load(src, map_location="cpu", weights_only=False)
shapes = {}
for part, n in (("actor_state_dict", na), ("critic_state_dict", nc)):
    sd = ck[part]
    w = sd["mlp.0.weight"]
    shapes[tuple(w.shape)] = n
    sd["mlp.0.weight"] = torch.cat([w, torch.zeros(w.shape[0], n)], 1)
    for k, fill in (("obs_normalizer._mean", 0.0), ("obs_normalizer._var", 1.0), ("obs_normalizer._std", 1.0)):
        v = sd[k]
        sd[k] = torch.cat([v, torch.full((v.shape[0], n), fill)], 1)
opt = ck.get("optimizer_state_dict")
if opt:
    for st in opt["state"].values():
        for key in ("exp_avg", "exp_avg_sq"):
            t = st.get(key)
            if t is not None and t.dim() == 2 and tuple(t.shape) in shapes:
                st[key] = torch.cat([t, torch.zeros(t.shape[0], shapes[tuple(t.shape)])], 1)
torch.save(ck, dst)
print("padded", src, "->", dst, {k: v for k, v in shapes.items()})
