"""Where is the actor relative to the bridge line?  Lab frame from the CSV marks: plank runs from
terminal-up (-0.157, 1.219, -2.617) to front-up (0.002, 1.256, 2.227) (x lateral, y up, z forward, metres)."""
import glob, os, sys
import mujoco
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from bridge_check import fit_similarity, read_csv_footholds  # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage1_reference"))
from build_reference import TOE_LOCAL, load_robot  # noqa: E402

A_UP, B_UP = np.array([-0.157, 1.219, -2.617]), np.array([0.002, 1.256, 2.227])
ref_dir, csv_path = sys.argv[1], sys.argv[2]
r = load_robot()
head = mujoco.mj_name2id(r.m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
feet = read_csv_footholds(csv_path)
for f in sorted(glob.glob(os.path.join(ref_dir, "CrouchedTraverse_*_Dynamics.npz")))[:6]:
    name = os.path.basename(f).replace("_Dynamics.npz", "")
    d = np.load(f)
    Q, crop0 = d["qpos"], int(d["source_crop"][0])
    A, B = [], []
    for foot, frame, X, Z in feet[name]:
        i = frame - crop0
        if 0 <= i < len(Q):
            r.d.qpos[:] = Q[i]
            mujoco.mj_kinematics(r.m, r.d)
            fid = r.foot_ids[0 if foot == "L" else 1]
            A.append([X, Z]); B.append((r.d.xpos[fid] + r.d.xmat[fid].reshape(3, 3) @ TOE_LOCAL)[:2])
    s, R, t = fit_similarity(np.array(A), np.array(B))
    rows = []
    for q in Q[::6]:
        r.d.qpos[:] = q
        mujoco.mj_kinematics(r.m, r.d)
        p = r.d.xpos[head]
        top = p[2] + 0.45 * s / s  # torso_link origin -> approx head top (+0.45 m robot)
        lab = (np.linalg.inv(R) @ (p[:2] - t)) / s
        u = (lab[1] - A_UP[2]) / (B_UP[2] - A_UP[2])
        if 0 <= u <= 1:
            line_x = A_UP[0] + u * (B_UP[0] - A_UP[0])
            plank_y = A_UP[1] + u * (B_UP[1] - A_UP[1])
            rows.append((lab[0] - line_x, (p[2] + 0.45) / s, plank_y))
    rows = np.array(rows)
    print(f"{name}: scale {s:.3f}; torso lateral offset from plank line (lab m): min|dx| {np.abs(rows[:,0]).min():.2f}, "
          f"median|dx| {np.median(np.abs(rows[:,0])):.2f}; head-top height lab m: min {rows[:,1].min():.2f} median {np.median(rows[:,1]):.2f}; plank mark height {rows[:,2].mean():.2f}")
