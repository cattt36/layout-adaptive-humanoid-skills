"""Which bridge part does the CrouchedTraverse_001 reference intersect, where, and with which robot body?"""
import os, sys
import mujoco
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import bridge_check as B  # noqa: E402

r = B.load_robot()
feet = B.read_csv_footholds(sys.argv[2])
name = sys.argv[3] if len(sys.argv) > 3 else "CrouchedTraverse_001"
dd = np.load(os.path.join(sys.argv[1], f"{name}_Dynamics.npz"))
Q, crop0 = dd["qpos"], int(dd["source_crop"][0])
A, Bp = [], []
for foot, frame, X, Z in feet[name]:
    i = frame - crop0
    r.d.qpos[:] = Q[i]; mujoco.mj_kinematics(r.m, r.d)
    fid = r.foot_ids[0 if foot == "L" else 1]
    A.append([X, Z]); Bp.append((r.d.xpos[fid] + r.d.xmat[fid].reshape(3, 3) @ B.TOE_LOCAL)[:2])
s, R, t = B.fit_similarity(np.array(A), np.array(Bp))
m, d, obs, rob, names = B.build_model(s, R, t)
torso = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
head_g = max(rob, key=lambda g: m.geom_xpos[g][2] if False else 0)
ft = np.zeros(6)
Rinv = R.T
print("footholds lab (x,z):", [(round(a[0], 2), round(a[1], 2)) for a in A])
print("posts lab z:", [round(p[0], 2) for p in B.POSTS], "plank z:", round(B.PLANK["z0"], 2), round(B.PLANK["z1"], 2))
for i in range(0, len(Q), 12):
    d.qpos[:] = Q[i]; mujoco.mj_kinematics(m, d)
    best = (1.0, None, None)
    for k, o in enumerate(obs):
        for g in rob:
            dist = mujoco.mj_geomDistance(m, d, g, o, 0.5, ft)
            if dist < best[0]:
                best = (dist, names[k], mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g))
    p = d.xpos[torso]
    lab = (Rinv @ (p[:2] - t)) / s
    top = max(d.geom_xpos[g][2] + m.geom_rbound[g] for g in rob) / s
    print(f"t={i/120:5.2f}s torso lab x={lab[0]:+.2f} z={lab[1]:+.2f} body-top(lab)={top:.2f}m  closest={best[1]}:{best[2]} d={best[0]*1:+.3f}")
