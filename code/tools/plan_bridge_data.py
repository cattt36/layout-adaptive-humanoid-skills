"""DEMO: data for plan_doc.py fig 4 (reference-pose bridge posts in the start frame, robot / human pelvis paths) -> bmJ/plan_doc/bridge_ref.npz"""
import json, glob, numpy as np, mujoco
B = "/root/gpufree-data/robot_sim_check/bmJ"; H = "/root/gpufree-data/hnu_v2"
def yaw(q): return np.arctan2(2*(q[0]*q[3]+q[1]*q[2]), 1-2*(q[2]**2+q[3]**2))
out = {}
for c, run in (("013", "013w10"), ("005", "005j2")):
    L = f"{B}/bz_a/dir_{c}"
    ref = np.load(f"{L}/CrouchedTraverse_{c}_motion.npz"); J = json.load(open(f"{L}/CrouchedTraverse_{c}_bridge.json"))
    a0 = yaw(ref["body_quat_w"][0, 0]); o0 = ref["body_pos_w"][0, 0, :2]
    R = np.array([[np.cos(-a0), -np.sin(-a0)], [np.sin(-a0), np.cos(-a0)]]); s = lambda p: (np.asarray(p, float)[..., :2] - o0) @ R.T
    P = J["parts"]
    names = sorted((k for k in P if k.startswith("post")), key=lambda k: -(P[k]["center"][2] + P[k]["half_height"]))
    pl = P["plank"]
    print(f"== {c}: plank yaw(start frame) {np.degrees(pl['yaw'] - a0):.1f} deg, pitch {np.degrees(pl.get('pitch', 0)):.2f}, half {pl.get('half')}")
    xy = []
    for i, k in enumerate(names):
        p = s(P[k]["center"]); top = P[k]["center"][2] + P[k]["half_height"]; xy.append(p)
        print(f"  post{i+1} ({k}) x {p[0]:.3f} y {p[1]:.3f}  toO {np.hypot(*p):.3f} toA {np.hypot(p[0]-3, p[1]):.3f} top {top:.3f} r {P[k]['radius']:.4f}")
    xy = np.array(xy); print("  spacing", np.round(np.linalg.norm(np.diff(xy, axis=0), axis=1), 3), "1-4", round(float(np.linalg.norm(xy[0]-xy[3])), 3))
    out[f"posts{c}"] = xy
    ex = f"{H}/deploy_exports/rtJbz/rtJbz_{run}"
    M = mujoco.MjModel.from_binary_path(f"{ex}/scene.mjb"); D = mujoco.MjData(M)
    meta = json.load(open(f"{ex}/policy_meta.json")); pid = mujoco.mj_name2id(M, mujoco.mjtObj.mjOBJ_BODY, "robot/" + meta["body_names"][0])
    Hh = int(round(12 / meta["sim"]["control_dt"])); allp = []
    for i, f in enumerate(sorted(glob.glob(f"{B}/bz_a/trace_{run}_REP/*.npz"))[:32]):
        q = np.load(f)["qpos"]; xy2 = []
        for qq in q[Hh::4]:
            D.qpos[:] = qq[:M.nq]; mujoco.mj_kinematics(M, D); xy2.append(D.xpos[pid, :2].copy())
        out[f"path{c}_{i}"] = s(np.array(xy2)); allp.append(out[f"path{c}_{i}"])
    allp = np.concatenate(allp); print("  pelvis range x", np.round(allp[:, 0].min(), 2), np.round(allp[:, 0].max(), 2), "y", np.round(allp[:, 1].min(), 2), np.round(allp[:, 1].max(), 2))
    out[f"ref{c}"] = s(ref["body_pos_w"][::4, 0])
np.savez(f"{B}/plan_doc/bridge_ref.npz", **out)
