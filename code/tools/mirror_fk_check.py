"""Consistency of the mirrored references: forward kinematics of (mirrored root pose + mirrored joint angles) must
reproduce the mirrored body positions; also for the original as a reference.  Reports the max body position error
and joint-limit violations."""
import glob
import os

import mujoco
import numpy as np

from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_spec

m = get_spec().compile()
d = mujoco.MjData(m)
lo, hi = m.jnt_range[1:, 0], m.jnt_range[1:, 1]


def fk_err(z):
    e, viol = 0.0, 0.0
    for t in range(0, len(z["joint_pos"]), 10):
        d.qpos[:3] = z["body_pos_w"][t, 0]
        d.qpos[3:7] = z["body_quat_w"][t, 0]
        d.qpos[7:] = z["joint_pos"][t]
        mujoco.mj_kinematics(m, d)
        e = max(e, np.abs(d.xpos[1:] - z["body_pos_w"][t]).max())
        viol = max(viol, np.maximum(lo - z["joint_pos"][t], z["joint_pos"][t] - hi).max())
    return e, viol


H = "/root/gpufree-data/hnu_v2/data"
for orig in sorted(glob.glob(f"{H}/ref_stakes_s93_h10/ThreeStepPoleRun/*_motion.npz"))[:1] + \
        sorted(glob.glob(f"{H}/ref_bridge_video_lift/CrouchedTraverse/*_motion.npz"))[:1]:
    name = os.path.basename(orig).replace("_motion.npz", "")
    mir = glob.glob(f"{H}/augchk/mirror/*/*/{name}_mirror_motion.npz")
    a = np.load(os.path.realpath(orig))
    print(name, "original: max FK error %.4f m, max joint-limit excess %.3f rad" % fk_err(a))
    if mir:
        b = np.load(mir[0])
        print(name, "mirrored: max FK error %.4f m, max joint-limit excess %.3f rad" % fk_err(b))
