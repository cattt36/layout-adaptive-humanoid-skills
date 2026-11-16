"""Parity of the deployment runtime with the training env (deploy venv: mujoco CPU + onnxruntime, no torch).

For the recorded training-env rollout (trace.npz from export_student.py): rebuild every observation from the raw
MuJoCo state with hnu_deploy.StudentPolicy and compare term by term; run policy.onnx on the recorded observations
and compare with the recorded torch actions.

  python deploy/parity_check.py <export_dir>
"""
import json
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hnu_deploy import ClipContext, RobotState, StudentPolicy  # noqa: E402


def clip_files(meta, clip):
    for d in meta["motion_dirs"]:
        if os.path.exists(os.path.join(d, f"{clip}_motion.npz")):
            return os.path.join(d, f"{clip}_stakes.json"), os.path.join(d, f"{clip}_bridge.json")
    raise FileNotFoundError(clip)


def sensor(m, d, name):
    i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, name)
    return d.sensordata[m.sensor_adr[i]: m.sensor_adr[i] + m.sensor_dim[i]].copy()


def state_from(m, d, meta):
    qadr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "robot/" + j)] for j in meta["joint_names"]]
    vadr = [m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "robot/" + j)] for j in meta["joint_names"]]
    return RobotState(ang_vel=sensor(m, d, meta["sensors"]["ang_vel"]), lin_vel=sensor(m, d, meta["sensors"]["lin_vel"]),
                      quat=d.qpos[3:7].copy(), pos=d.qpos[0:3].copy(), q=d.qpos[qadr].copy(), dq=d.qvel[vadr].copy())


def main():
    ex = sys.argv[1]
    meta = json.load(open(os.path.join(ex, "policy_meta.json")))
    tr = np.load(os.path.join(ex, "trace.npz"))
    m = mujoco.MjModel.from_binary_path(os.path.join(ex, "scene.mjb"))
    d = mujoco.MjData(m)
    pol = StudentPolicy(ex)
    bounds = np.cumsum([0] + [t["total"] for t in meta["obs_layout"]])
    worst = {t["name"]: 0.0 for t in meta["obs_layout"]}
    act_err, n = 0.0, 0
    for e, clip in enumerate(tr["clips"]):
        clip = str(clip)
        sj, bj = clip_files(meta, clip)
        ctx = ClipContext(meta, np.load(os.path.join(meta["latent_dir"], f"{clip}_z.npy")), sj, bj)
        for t in range(tr["obs"].shape[0]):
            d.qpos[:], d.qvel[:] = tr["qpos"][t, e], tr["qvel"][t, e]
            d.mocap_pos[:], d.mocap_quat[:] = tr["mocap_pos"][t, e], tr["mocap_quat"][t, e]
            mujoco.mj_forward(m, d)
            s = state_from(m, d, meta)
            s.pos = s.pos - tr["env_origins"][e]  # training envs sit on a grid; obstacles are in motion coordinates
            frame = int(tr["frame"][t, e])
            if t == 0:
                pol.reset(ctx, s, frame)
            pol.last_action = tr["action"][t - 1, e].astype(np.float64) if t > 0 else np.zeros(meta["act_dim"])
            o = pol.observation(s, frame)
            ref = tr["obs"][t, e]
            for k, tm in enumerate(meta["obs_layout"]):
                worst[tm["name"]] = max(worst[tm["name"]], float(np.abs(o[bounds[k]:bounds[k + 1]] -
                                                                         ref[bounds[k]:bounds[k + 1]]).max()))
            a = pol.sess.run(None, {"obs": ref[None].astype(np.float32)})[0][0]
            act_err = max(act_err, float(np.abs(a - tr["action"][t, e]).max()))
            n += 1
            if tr["done"][t, e]:
                break
        print(f"{clip}: {t + 1} steps compared")
    print("max |obs_deploy - obs_train| per term:")
    for k, v in worst.items():
        print(f"  {k:18s} {v:.2e}")
    print(f"max |onnx - torch| action: {act_err:.2e}  ({n} steps)")
    ok = max(worst.values()) < 1e-3 and act_err < 1e-3
    print("PARITY", "OK" if ok else "MISMATCH")


if __name__ == "__main__":
    main()
