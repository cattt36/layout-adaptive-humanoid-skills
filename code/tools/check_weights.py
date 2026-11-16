"""Sampling mass per action (clip weight x clip length, as in the bin-based sampler) for the v6g10k training set
vs the v6g10j set (no copies), using hnu_tasks.multi_motion.clip_weights."""
import os
import types

import numpy as np
import torch

from hnu_tasks.multi_motion import TASK_NAMES, clip_weights, task_of_clip

H = "/root/gpufree-data/hnu_v2/data"
dirs = {"ThreeStepPoleRun": f"{H}/aug_ts/ref_stakes_s93_h10/ThreeStepPoleRun", "FiveStepPoleRun": f"{H}/aug_ts/ref_stakes_s83_h10/FiveStepPoleRun",
        "CrouchedTraverse": f"{H}/ref_bridge_video_lift/CrouchedTraverse"}
os.environ["HNU_TASK_WEIGHT"] = "CrouchedTraverse:1.5"
for name, f, share in (("v6g10j", "clips_train.txt", "0"), ("v6g10k", "clips_train_aug.txt", "0.5")):
    os.environ["HNU_TS_SHARE"] = share
    names = [ln.strip() for ln in open(f"{H}/clip_split_trial/{f}") if ln.strip()]
    m = types.SimpleNamespace(names=names, clip_task=torch.tensor([task_of_clip(n) for n in names]))
    w = clip_weights(m, "cpu").numpy()
    L = np.array([np.load(f"{dirs[n.split('_')[0]]}/{n}_motion.npz")["joint_pos"].shape[0] for n in names])
    mass = w * L
    tot = mass.sum()
    cells = []
    for t, T in enumerate(TASK_NAMES):
        sel = m.clip_task.numpy() == t
        ts = np.array(["_ts" in n for n in names]) & sel
        cells.append(f"{T} {100 * mass[sel].sum() / tot:.1f}% (copies {100 * mass[ts].sum() / max(mass[sel].sum(), 1e-9):.0f}%)")
    print(f"{name}: " + " | ".join(cells))
