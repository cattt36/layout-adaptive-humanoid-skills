"""Print the structure of an rsl_rl checkpoint (keys, tensor shapes)."""
import sys

import torch

ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)


def show(d, pre=""):
    for k, v in d.items():
        if isinstance(v, dict):
            show(v, f"{pre}{k}.")
        elif hasattr(v, "shape"):
            print(f"{pre}{k}: {tuple(v.shape)}")
        else:
            print(f"{pre}{k}: {type(v).__name__}")


show(ck)
