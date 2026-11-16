"""Candidate augmentations for the teacher check (TRAIN trials only):
  A  local time warps (speed s < 1 = slower; 0.3 s linear ramps in / out of the warped segment):
       appr085 / appr115   run-up / approach segment at 0.85 / 1.15x (ThreeStepPoleRun / FiveStepPoleRun: until 0.2 s before the first stake
                           stance; CrouchedTraverse: until 0.2 s before the first frame under the beam)
       dwell2              ThreeStepPoleRun / FiveStepPoleRun: the stance on stake 2 lasts twice as long (s = 0.5 inside its stance window)
  B  mirror                left / right reflection about the motion x-z plane: y -> -y, left / right joints and
                           bodies swapped, roll / yaw joints negated, quaternion (w, x, y, z) -> (w, -x, y, -z),
                           angular velocity (-wx, wy, -wz), foot contacts swapped; stakes (y, foot) and bridge parts
                           (y, yaw) mirrored accordingly.
Writes data/augchk/<variant>/{ref_stakes_s93_h10/ThreeStepPoleRun, ref_stakes_s83_h10/FiveStepPoleRun,
ref_bridge_video_lift/CrouchedTraverse} and data/augchk/<variant>/z/Z0..Z4 (1.0x z resampled on the same time map / for
mirror the original z: the teacher check only needs matching lengths).
Velocities of warped motions are scaled by the local speed."""
import glob
import json
import os
import shutil

import numpy as np

H = "/root/gpufree-data/hnu_v2/data"
SRC = {"ref_stakes_s93_h10/ThreeStepPoleRun": "W", "ref_stakes_s83_h10/FiveStepPoleRun": "W", "ref_bridge_video_lift/CrouchedTraverse": "B"}
LV = ["Z0", "Z1", "Z2", "Z3", "Z4"]
train = {ln.strip() for ln in open(f"{H}/clip_split_trial/clips_train.txt") if ln.strip()}

JOINTS = ["left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee", "left_ankle_pitch", "left_ankle_roll",
          "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee", "right_ankle_pitch", "right_ankle_roll",
          "waist_yaw", "waist_roll", "waist_pitch",
          "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow", "left_wrist_roll",
          "left_wrist_pitch", "left_wrist_yaw",
          "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow", "right_wrist_roll",
          "right_wrist_pitch", "right_wrist_yaw"]
BODIES = ["pelvis", "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee", "left_ankle_pitch", "left_ankle_roll",
          "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee", "right_ankle_pitch", "right_ankle_roll",
          "waist_yaw", "waist_roll", "torso",
          "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow", "left_wrist_roll",
          "left_wrist_pitch", "left_wrist_yaw",
          "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow", "right_wrist_roll",
          "right_wrist_pitch", "right_wrist_yaw"]


def swap_lr(names):
    sw = lambda n: n.replace("left_", "@").replace("right_", "left_").replace("@", "right_")  # noqa: E731
    return np.array([names.index(sw(n)) for n in names])


J_PERM, B_PERM = swap_lr(JOINTS), swap_lr(BODIES)
J_SIGN = np.array([-1.0 if ("roll" in n or "yaw" in n) else 1.0 for n in JOINTS])


# ---------------------------------------------------------------- B: mirror
def mirror_motion(d):
    out = {"fps": d["fps"]}
    out["joint_pos"] = (d["joint_pos"][:, J_PERM] * J_SIGN).astype(np.float32)
    out["joint_vel"] = (d["joint_vel"][:, J_PERM] * J_SIGN).astype(np.float32)
    p = d["body_pos_w"][:, B_PERM].copy()
    p[..., 1] *= -1
    q = d["body_quat_w"][:, B_PERM].copy()
    q[..., 1] *= -1
    q[..., 3] *= -1
    v = d["body_lin_vel_w"][:, B_PERM].copy()
    v[..., 1] *= -1
    w = d["body_ang_vel_w"][:, B_PERM].copy()
    w[..., 0] *= -1
    w[..., 2] *= -1
    out.update(body_pos_w=p, body_quat_w=q, body_lin_vel_w=v, body_ang_vel_w=w, foot_contact=d["foot_contact"][:, ::-1].copy())
    return out


def mirror_stakes(j):
    j = json.loads(json.dumps(j))
    for s in j["stakes"]:
        s["y"] = -s["y"]
        s["foot"] = {"L": "R", "R": "L"}[s["foot"]]
    if "stake_xyz_50hz" in j:
        j["stake_xyz_50hz"] = [[p[0], -p[1], p[2]] for p in j["stake_xyz_50hz"]]
    j["mirrored"] = True
    return j


def mirror_bridge(j):
    j = json.loads(json.dumps(j))
    for part in j["parts"].values():
        part["center"][1] = -part["center"][1]
        part["yaw"] = -part["yaw"]
    j["mirrored"] = True
    return j


# ---------------------------------------------------------------- A: local time warp
def time_map(T, speed_fn):
    """speed_fn(t_orig) -> local speed; returns original (fractional) frame of every new frame and its speed."""
    t, ts, ss = 0.0, [], []
    while t <= T - 1:
        s = speed_fn(t)
        ts.append(t)
        ss.append(s)
        t += s
    return np.array(ts), np.array(ss)


def profile(a, b, v, ramp=15):
    """speed v on [a, b) (original 50 Hz frames), linear ramps of `ramp` frames outside, 1 elsewhere."""
    def f(t):
        if a <= t < b:
            return v
        if a - ramp <= t < a:
            return 1 + (v - 1) * (t - (a - ramp)) / ramp
        if b <= t < b + ramp:
            return v + (1 - v) * (t - b) / ramp
        return 1.0
    return f


def warp_array(x, ts, ss=None, vel=False, nearest=False, quat=False):
    T = len(x)
    i0 = np.floor(ts).astype(int)
    i1 = np.minimum(i0 + 1, T - 1)
    w = (ts - i0).reshape((-1,) + (1,) * (x.ndim - 1))
    if nearest:
        return x[np.where(w.reshape(-1) < 0.5, i0, i1)]
    if quat:
        q0, q1 = x[i0], x[i1].copy()
        sg = np.sign((q0 * q1).sum(-1, keepdims=True))
        sg[sg == 0] = 1
        q = (1 - w) * q0 + w * q1 * sg
        return (q / np.linalg.norm(q, axis=-1, keepdims=True)).astype(x.dtype)
    y = (1 - w) * x[i0] + w * x[i1]
    if vel:
        y = y * ss.reshape((-1,) + (1,) * (x.ndim - 1))
    return y.astype(x.dtype)


def warp_motion(d, ts, ss):
    out = {"fps": d["fps"]}
    for k in d.files:
        if k == "fps":
            continue
        out[k] = warp_array(d[k], ts, ss, vel="vel" in k, nearest=k == "foot_contact", quat="quat" in k)
    return out


def new_frame(ts, f_orig):
    """first new frame whose original time reaches f_orig."""
    return int(np.searchsorted(ts, f_orig))


def warp_stakes(j, ts):
    j = json.loads(json.dumps(j))
    r = 50.0 / j["fps"]
    conv = lambda f: int(round(new_frame(ts, f * r) / r))  # noqa: E731
    for s in j["stakes"]:
        s["stance"] = [conv(s["stance"][0]), conv(s["stance"][1])]
    if "ramp" in j:
        j["ramp"] = [conv(x) for x in j["ramp"]]
    return j


def first_under_beam(m, bj):
    p = bj["parts"]["plank"]
    c, yaw, half = np.array(p["center"]), p["yaw"], p["half"]
    d, n = np.array([np.cos(yaw), np.sin(yaw)]), np.array([-np.sin(yaw), np.cos(yaw)])
    rel = m["body_pos_w"][:, :, :2] - c[:2]
    inside = (np.abs(rel @ n) < 0.25) & (np.abs(rel @ d) < half[0])
    return int(np.where(inside.any(1))[0][0])


def main():
    variants = ["mirror", "appr085", "appr115", "dwell2"]
    for var in variants:
        root = f"{H}/augchk/{var}"
        made = []
        for rel, kind in SRC.items():
            dst = f"{root}/{rel}"
            os.makedirs(dst, exist_ok=True)
            for f in sorted(glob.glob(f"{H}/{rel}/*_motion.npz")):
                name = os.path.basename(f).replace("_motion.npz", "")
                if name not in train:
                    continue
                d = np.load(os.path.realpath(f))
                T = d["joint_pos"].shape[0]
                sj, bjf = f.replace("_motion.npz", "_stakes.json"), f.replace("_motion.npz", "_bridge.json")
                sjd = json.load(open(sj)) if os.path.exists(sj) else None
                bjd = json.load(open(os.path.realpath(bjf))) if os.path.exists(bjf) else None
                if var == "mirror":
                    out, ts = mirror_motion(d), np.arange(T, dtype=float)
                    sjd = mirror_stakes(sjd) if sjd else None
                    bjd = mirror_bridge(bjd) if bjd else None
                else:
                    if kind == "W":
                        r = 50.0 / sjd["fps"]
                        first = sjd["stakes"][0]["stance"][0] * r
                        if var == "dwell2":
                            a, b = sjd["stakes"][1]["stance"][0] * r, sjd["stakes"][1]["stance"][1] * r
                            fn = profile(a, b, 0.5, ramp=3)
                        else:
                            fn = profile(0, first - 10 - 15, 0.85 if var == "appr085" else 1.15)
                    elif var == "dwell2":  # no stakes: identity copy (the geom scene needs bridge clips; sanity check)
                        fn = lambda t: 1.0  # noqa: E731
                    else:
                        entry = first_under_beam(d, bjd)
                        fn = profile(0, entry - 10 - 15, 0.85 if var == "appr085" else 1.15)
                    ts, ss = time_map(T, fn)
                    out = warp_motion(d, ts, ss)
                    sjd = warp_stakes(sjd, ts) if sjd else None
                np.savez(f"{dst}/{name}_{var}_motion.npz", **out)
                if sjd:
                    json.dump(sjd, open(f"{dst}/{name}_{var}_stakes.json", "w"), indent=1)
                if bjd:
                    json.dump(bjd, open(f"{dst}/{name}_{var}_bridge.json", "w"), indent=1)
                for P in LV:
                    zd = f"{root}/z/{P}"
                    os.makedirs(zd, exist_ok=True)
                    z = np.load(f"{H}/z_tables_v5t/{P}/{name}_z.npy")
                    np.save(f"{zd}/{name}_{var}_z.npy", z if var == "mirror" else warp_array(z, ts))
                made.append((name, T, len(out["joint_pos"])))
        dur = np.mean([b / a for _, a, b in made])
        print(f"{var}: {len(made)} train clips, mean length ratio {dur:.3f}")


if __name__ == "__main__":
    main()
