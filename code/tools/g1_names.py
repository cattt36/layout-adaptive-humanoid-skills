"""Print the G1 joint / body order of the mjlab model (motion npz order) for the mirror mapping."""
import mujoco
import numpy as np

from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_spec

m = get_spec().compile()
print("nbody", m.nbody, "njnt", m.njnt)
print("bodies:", [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody)])
print("joints:", [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)])
d = np.load("/root/gpufree-data/hnu_v2/data/ref_stakes_s83_h10/FiveStepPoleRun/FiveStepPoleRun_001_motion.npz")
print("npz joint_pos", d["joint_pos"].shape, "body_pos_w", d["body_pos_w"].shape)
