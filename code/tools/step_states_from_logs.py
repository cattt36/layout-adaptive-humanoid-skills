"""stepping-in-place start states from the REAL robot logs (results/data_F_*.csv of the user's deploy
software, rows with mode "preloco" = the walking controller stepping in place just before the switch to the student).
Same npz layout as tools/make_step_states.py (which drew synthetic states around a knee angle of 0.3 rad; the real
walking stance is knee 0.63-0.88, hip pitch -0.30..-0.45, ankle -0.30..-0.48).

Joint angles / velocities from the encoders, roll / pitch from the IMU quaternion, body rates from the gyro, pelvis
velocity from the mocap (vx, vy, vz; world frame) rotated into the heading frame with yaw_deg; the pelvis height is
solved by forward kinematics (lowest sole sphere on the floor; the mocap height had a +4-5 cm bias).
  python tools/step_states_from_logs.py <export_dir> <out.npz> <csv> [<csv> ...] [--stride 2]
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
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--stride", type=int, default=2)
    a = ap.parse_args()
    meta = json.load(open(os.path.join(a.export_dir, "policy_meta.json")))
    names = meta["joint_names"]
    m = mujoco.MjModel.from_binary_path(os.path.join(a.export_dir, "scene.mjb"))
    d = mujoco.MjData(m)
    qadr = np.array([m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "robot/" + j)] for j in names])
    soles = [i for i in range(m.ngeom) if re.fullmatch(r"robot/(left|right)_foot[1-7]_collision",
                                                       mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or "")]
    out = {k: [] for k in ("q", "dq", "z", "rp_quat", "v_yaw", "w_body", "support")}
    for f in a.csv:
        h = open(f).readline().strip().split(",")
        ix = {c: i for i, c in enumerate(h)}
        rows = [ln.strip().split(",") for ln in open(f).readlines()[1:]]
        rows = [r for r in rows if len(r) == len(h) and r[ix["mode"]] == "preloco"][:: a.stride]
        for r in rows:
            try:
                g = lambda c: float(r[ix[c]])  # noqa: E731
                q = np.array([g("q." + j) for j in names])
                dq = np.array([g("dq." + j) for j in names])
                qw, qx, qy, qz = (g(c) for c in ("imu_qw", "imu_qx", "imu_qy", "imu_qz"))
                gyro = np.array([g("gyro_x"), g("gyro_y"), g("gyro_z")])
                vw = np.array([g("vx"), g("vy"), g("vz")])
                yaw = np.radians(g("yaw_deg"))
            except (ValueError, KeyError):
                continue
            if not (np.all(np.isfinite(q)) and np.all(np.isfinite(dq)) and np.isfinite(qw)):
                continue
            # roll / pitch only: remove the heading of the IMU quaternion
            iy = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
            qy_inv = np.array([np.cos(iy / 2), 0.0, 0.0, -np.sin(iy / 2)])
            rp = np.zeros(4)
            mujoco.mju_mulQuat(rp, qy_inv, np.array([qw, qx, qy, qz]))
            d.qpos[:] = m.qpos0
            d.qpos[0:3] = [0.0, 0.0, 1.0]
            d.qpos[3:7] = rp
            d.qpos[qadr] = q
            mujoco.mj_kinematics(m, d)
            z = 1.0 - min(d.geom_xpos[i][2] - m.geom_size[i][0] for i in soles)
            c, s = np.cos(yaw), np.sin(yaw)
            v = np.array([c * vw[0] + s * vw[1], -s * vw[0] + c * vw[1], vw[2]]) if np.all(np.isfinite(vw)) else np.zeros(3)
            out["q"].append(q)
            out["dq"].append(np.clip(dq, -8, 8))
            out["z"].append(z)
            out["rp_quat"].append(rp)
            out["v_yaw"].append(np.clip(v, -0.6, 0.6))
            out["w_body"].append(np.clip(gyro, -2, 2))
            out["support"].append(-1)
        print(f, len(rows))
    res = {k: np.asarray(v, np.float32 if k != "support" else np.int8) for k, v in out.items()}
    np.savez(a.out, joint_names=np.array(names), **res)
    J = {n: i for i, n in enumerate(names)}
    print(a.out, len(res["z"]), "z %.3f-%.3f" % (res["z"].min(), res["z"].max()),
          "knee L %.2f-%.2f" % tuple(np.percentile(res["q"][:, J["left_knee_joint"]], [5, 95])),
          "|v| p95 %.2f" % np.percentile(np.linalg.norm(res["v_yaw"][:, :2], axis=1), 95),
          "|w| p95 %.2f" % np.percentile(np.linalg.norm(res["w_body"], axis=1), 95))


if __name__ == "__main__":
    main()
