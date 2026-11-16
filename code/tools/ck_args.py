"""Print the data / profile arguments stored in a latent checkpoint."""
import sys

import torch

ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
a = ck.get("args", {}) or {}
for k, v in sorted(a.items()):
    if any(s in k for s in ("root", "json", "dir", "profile", "dataset", "split", "text")):
        print(k, "=", v)
