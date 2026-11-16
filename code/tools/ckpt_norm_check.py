"""Does the latent checkpoint store its normalizers, and do they differ from the ones export_z_tables.py re-fits
(default split mode 'subject')?  -> consistency of the z tables with the trained encoder."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/root/gpufree-data/hnu_v2/code/stage4_latent")
import motion_latent_z as M  # noqa: E402

ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print("checkpoint keys:", sorted(ck.keys()))
for k in ck:
    if "norm" in k.lower():
        v = ck[k]
        print(" ", k, type(v).__name__, (sorted(v.keys())[:8] if isinstance(v, dict) else ""))
print("args split_mode:", (ck.get("args") or {}).get("split_mode"))
