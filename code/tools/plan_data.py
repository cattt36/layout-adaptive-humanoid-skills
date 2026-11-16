"""DEMO: data for the experiment-plan figures (start-frame pelvis paths of bridge_013 REP runs, frame-0 feet)."""
import glob, json, os, sys
import mujoco
import numpy as np
sys.path.insert(0, "/root/gpufree-data/hnu_v2/code/tools")
B = "/root/gpufree-data/robot_sim_check/bmJ"; H = "/root/gpufree-data/hnu_v2"
def yaw(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
def frame(ref):
    a0 = yaw(ref["body_quat_w"][0, 0]); o0 = ref["body_pos_w"][0, 0, :2]
    R = np.array([[np.cos(-a0), -np.sin(-a0)], [np.sin(-a0), np.cos(-a0)]])
    return lambda p: (np.asarray(p, float)[..., :2] - o0) @ R.T
out = {}
ex = f"{H}/deploy_exports/rtJbx/rtJbxf_013"
M = mujoco.MjModel.from_binary_path(f"{ex}/scene.mjb"); D = mujoco.MjData(M)
meta = json.load(open(f"{ex}/policy_meta.json"))
pid = mujoco.mj_name2id(M, mujoco.mjtObj.mjOBJ_BODY, "robot/" + meta["body_names"][0])
ref = np.load(f"{B}/flat/CrouchedTraverse_013/CrouchedTraverse_013_motion.npz"); to_s = frame(ref)
Hh = int(round(12 / meta["sim"]["control_dt"]))
for i, f in enumerate(sorted(glob.glob(f"{B}/spread_bx/trace_rtJbxf_013_REP/*.npz"))[:32]):
    xy = []
    for q in np.load(f)["qpos"][Hh::4]:
        D.qpos[:] = q[:M.nq]; mujoco.mj_kinematics(M, D); xy.append(D.xpos[pid, :2].copy())
    out[f"path013_{i}"] = to_s(np.array(xy))
# frame-0 feet of 3step_012_stakes (sole hull, start frame)
ref = np.load(f"{B}/calib/on_ThreeStepPoleRun_012/ThreeStepPoleRun_012_motion.npz"); to_s = frame(ref)
bp, bq = ref["body_pos_w"][0], ref["body_quat_w"][0]
for side, idx in (("L", 6), ("R", 12)):
    b = mujoco.mj_name2id(M, mujoco.mjtObj.mjOBJ_BODY, f"robot/{'left' if side == 'L' else 'right'}_ankle_roll_link")
    g = [k for k in range(M.ngeom) if M.geom_bodyid[k] == b and M.geom_type[k] == mujoco.mjtGeom.mjGEOM_MESH][0]
    mid = M.geom_dataid[g]
    v = M.mesh_vert[M.mesh_vertadr[mid]:M.mesh_vertadr[mid] + M.mesh_vertnum[mid]].astype(float)
    Rg = np.zeros(9); mujoco.mju_quat2Mat(Rg, M.geom_quat[g]); v = v @ Rg.reshape(3, 3).T + M.geom_pos[g]
    Rb = np.zeros(9); mujoco.mju_quat2Mat(Rb, bq[idx]); w = v @ Rb.reshape(3, 3).T + bp[idx]
    out[f"foot{side}"] = to_s(w[v[:, 2] < v[:, 2].min() + 0.01])
os.makedirs(f"{B}/plan_doc", exist_ok=True)
np.savez(f"{B}/plan_doc/data.npz", **out)
print(len(out), "arrays")
