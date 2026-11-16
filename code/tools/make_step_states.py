"""start states "handed over from a locomotion controller stepping in place" (real G1 runs the
deployed student is switched on while the walking policy steps in place, or after a rope hold).

No locomotion policy is run here: the states are drawn from a parametric stepping-in-place family around the G1
walking stance (hip pitch -0.1, knee 0.3, ankle pitch -0.2 rad +- variation, feet flat), one swing leg lifted at a
random gait phase (single support 70 %, else double support), swing-leg joint velocities of a lift / set-down, lateral
sway and small body rates, arms near the G1 home pose.  The pelvis height is solved so that the lowest sole sphere
touches the floor (CPU MuJoCo forward kinematics of the exported scene).  Replace by recorded robot states when
available (same npz layout).

Out npz: joint_names [29], q / dq [N, 29] (policy joint order), z [N] pelvis height, rp_quat [N, 4] roll / pitch only
(wxyz, heading 0), v_yaw [N, 3] pelvis velocity in the heading frame, w_body [N, 3] body rates, support [N] (0 double,
1 left, 2 right).
  python tools/make_step_states.py <export_dir> <out.npz> [--n 8192] [--seed 0]
"""
import argparse
import json
import os
import re

import mujoco
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("export_dir")
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    meta = json.load(open(os.path.join(a.export_dir, "policy_meta.json")))
    names = meta["joint_names"]
    m = mujoco.MjModel.from_binary_path(os.path.join(a.export_dir, "scene.mjb"))
    d = mujoco.MjData(m)
    jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "robot/" + j) for j in names]
    qadr = np.array([m.jnt_qposadr[j] for j in jid])
    root = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "robot/floating_base_joint")] \
        if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "robot/floating_base_joint") >= 0 else 0
    soles = {s: [i for i in range(m.ngeom) if re.fullmatch(rf"robot/{s}_foot[1-7]_collision",
                                                             mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or "")]
             for s in ("left", "right")}
    J = {n: i for i, n in enumerate(names)}
    home = dict(left_shoulder_pitch_joint=0.2, right_shoulder_pitch_joint=0.2, left_shoulder_roll_joint=0.2,
                right_shoulder_roll_joint=-0.2, left_elbow_joint=0.6, right_elbow_joint=0.6)
    out = {k: [] for k in ("q", "dq", "z", "rp_quat", "v_yaw", "w_body", "support")}
    for _ in range(a.n):
        q = np.zeros(len(names))
        dq = rng.uniform(-0.2, 0.2, len(names))
        for k, v in home.items():
            q[J[k]] = v + rng.uniform(-0.2, 0.2)
        for k in names:
            if k.startswith("waist"):
                q[J[k]] = rng.uniform(-0.05, 0.05)
        hp0, kn0 = -0.1 + rng.uniform(-0.15, 0.15), max(0.3 + rng.uniform(-0.15, 0.2), 0.05)
        sup = 0 if rng.random() < 0.3 else int(rng.integers(1, 3))
        for side in ("left", "right"):
            hp, kn = hp0 + rng.uniform(-0.03, 0.03), kn0 + rng.uniform(-0.03, 0.03)
            hr = rng.uniform(-0.06, 0.06)
            swing = (sup == 1 and side == "right") or (sup == 2 and side == "left")
            if swing:  # lifted leg: hip flexion + knee flexion, foot roughly flat; lifting or setting down
                lift = rng.uniform(0.0, 0.4)
                hp, kn = hp - 0.8 * lift, kn + 1.6 * lift
                rate = rng.uniform(0.0, 1.5) * rng.choice([-1.0, 1.0])
                dq[J[f"{side}_hip_pitch_joint"]] = -0.8 * rate + rng.uniform(-0.3, 0.3)
                dq[J[f"{side}_knee_joint"]] = 1.6 * rate + rng.uniform(-0.3, 0.3)
                dq[J[f"{side}_ankle_pitch_joint"]] = -0.8 * rate + rng.uniform(-0.3, 0.3)
            q[J[f"{side}_hip_pitch_joint"]] = hp
            q[J[f"{side}_knee_joint"]] = kn
            q[J[f"{side}_ankle_pitch_joint"]] = -(hp + kn) + rng.uniform(-0.06, 0.06)
            q[J[f"{side}_hip_roll_joint"]] = hr
            q[J[f"{side}_ankle_roll_joint"]] = -hr + rng.uniform(-0.03, 0.03)
            q[J[f"{side}_hip_yaw_joint"]] = rng.uniform(-0.06, 0.06)
        roll, pitch = np.radians(rng.uniform(-2.5, 2.5)), np.radians(rng.uniform(-3.0, 3.0))
        rp = np.zeros(4)
        mujoco.mju_euler2Quat(rp, np.array([roll, pitch, 0.0]), "xyz")
        d.qpos[:] = m.qpos0
        d.qpos[root:root + 3] = [0.0, 0.0, 1.0]
        d.qpos[root + 3:root + 7] = rp
        d.qpos[qadr] = q
        mujoco.mj_kinematics(m, d)
        low = {s: min(d.geom_xpos[g][2] - m.geom_size[g][0] for g in soles[s]) for s in soles}
        z = 1.0 - min(low.values())
        out["q"].append(q)
        out["dq"].append(dq)
        out["z"].append(z)
        out["rp_quat"].append(rp)
        # stepping in place: lateral sway, little forward drift, body rates of the weight shift
        out["v_yaw"].append([rng.uniform(-0.1, 0.1), rng.uniform(-0.2, 0.2), rng.uniform(-0.05, 0.05)])
        out["w_body"].append([rng.uniform(-0.3, 0.3), rng.uniform(-0.2, 0.2), rng.uniform(-0.2, 0.2)])
        out["support"].append(sup)
    res = {k: np.asarray(v, np.float32 if k != "support" else np.int8) for k, v in out.items()}
    np.savez(a.out, joint_names=np.array(names), **res)
    print(a.out, {k: v.shape for k, v in res.items()}, "z %.3f-%.3f" % (res["z"].min(), res["z"].max()))


if __name__ == "__main__":
    main()
