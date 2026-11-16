"""Write latent-ready NPZ files (old key set + floor_contact_count) from the stage-1 references."""
import glob
import os
import sys

import numpy as np

src_root, dst_root = sys.argv[1], sys.argv[2]
for task in ("ThreeStepPoleRun", "FiveStepPoleRun", "CrouchedTraverse"):
    os.makedirs(os.path.join(dst_root, task), exist_ok=True)
    for f in sorted(glob.glob(os.path.join(src_root, task, "*_Dynamics.npz"))):
        d = dict(np.load(f))
        contact = d["foot_contact"]
        d["floor_contact_count"] = contact.sum(1).astype(np.int64)
        d["self_collision_count"] = np.zeros(len(contact), np.int64)
        d["max_floor_penetration_m"] = np.maximum(-d["sole_min_height_m"].min(1), 0.0)
        d["max_self_penetration_m"] = np.zeros(len(contact))
        d["scaled_robot_grf"] = d["foot_force_world"].sum(1)
        d["joint_limit_types"] = np.array(["hinge"] * d["joint_limit_excess"].shape[1])
        d["joint_limit_tolerance"] = np.array(1e-3)
        np.savez_compressed(os.path.join(dst_root, task, os.path.basename(f)), **d)
    print(task, len(glob.glob(os.path.join(dst_root, task, "*.npz"))))
