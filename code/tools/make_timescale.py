"""Time-scaled copies of the reference motions (tempo check / augmentation).  Playback speed s (s < 1 = slower):
new frame k samples the original at time k * s.  Positions: linear interpolation; quaternions: normalised lerp
with sign alignment; linear / angular / joint velocities: interpolated and multiplied by s; foot contact: nearest
frame.  The stake stance windows (in the json fps) are divided by s; stake / bridge geometry is unchanged.
z tables are resampled the same way (linear interpolation per dimension), lengths match the new clips.

  python make_timescale.py 0.85 0.9 1.1 1.15
writes data/ts/ts<100s>/{ref_stakes_s93_h10/ThreeStepPoleRun, ref_stakes_s83_h10/FiveStepPoleRun,
                         ref_bridge_video_lift/CrouchedTraverse, z_tables_v5t/Z0..Z4}"""
import glob
import json
import os
import shutil
import sys

import numpy as np

H = "/root/gpufree-data/hnu_v2/data"
SRC = {"ref_stakes_s93_h10/ThreeStepPoleRun": "W", "ref_stakes_s83_h10/FiveStepPoleRun": "W", "ref_bridge_video_lift/CrouchedTraverse": "B"}
LV = ["Z0", "Z1", "Z2", "Z3", "Z4"]


def grid(T, s):
    Tn = int(round((T - 1) / s)) + 1
    t = np.minimum(np.arange(Tn) * s, T - 1)
    i0 = np.floor(t).astype(int)
    i1 = np.minimum(i0 + 1, T - 1)
    return Tn, i0, i1, (t - i0)


def lerp(x, i0, i1, w):
    w = w.reshape((-1,) + (1,) * (x.ndim - 1))
    return (1 - w) * x[i0] + w * x[i1]


def scale_motion(d, s):
    T = d["joint_pos"].shape[0]
    Tn, i0, i1, w = grid(T, s)
    out = {"fps": d["fps"]}
    for k in d.files:
        if k == "fps":
            continue
        x = d[k]
        if k == "foot_contact":
            out[k] = x[np.where(w < 0.5, i0, i1)]
        elif "quat" in k:
            q0, q1 = x[i0], x[i1].copy()
            sign = np.sign((q0 * q1).sum(-1, keepdims=True))
            sign[sign == 0] = 1
            q = (1 - w[:, None, None]) * q0 + w[:, None, None] * q1 * sign
            out[k] = (q / np.linalg.norm(q, axis=-1, keepdims=True)).astype(x.dtype)
        else:
            y = lerp(x, i0, i1, w)
            out[k] = (y * s if "vel" in k else y).astype(x.dtype)
    return out, Tn


def main():
    for s in map(float, sys.argv[1:]):
        root = f"{H}/ts/ts{int(round(100 * s)):03d}"
        lens = {}
        for rel, kind in SRC.items():
            dst = f"{root}/{rel}"
            os.makedirs(dst, exist_ok=True)
            for f in sorted(glob.glob(f"{H}/{rel}/*_motion.npz")):
                name = os.path.basename(f).replace("_motion.npz", "")
                d = np.load(os.path.realpath(f))
                out, Tn = scale_motion(d, s)
                np.savez(f"{dst}/{name}_motion.npz", **out)
                lens[name] = (d["joint_pos"].shape[0], Tn)
                sj = f.replace("_motion.npz", "_stakes.json")
                if os.path.exists(sj):
                    j = json.load(open(sj))
                    for st in j["stakes"]:
                        st["stance"] = [int(round(st["stance"][0] / s)), int(round(st["stance"][1] / s))]
                    if "ramp" in j:
                        j["ramp"] = [int(round(r / s)) for r in j["ramp"]]
                    j["time_scale"] = s
                    json.dump(j, open(f"{dst}/{name}_stakes.json", "w"), indent=1)
                bj = f.replace("_motion.npz", "_bridge.json")
                if os.path.exists(bj):
                    shutil.copy(os.path.realpath(bj), f"{dst}/{name}_bridge.json")
        for P in LV:
            dst = f"{root}/z_tables_v5t/{P}"
            os.makedirs(dst, exist_ok=True)
            for f in sorted(glob.glob(f"{H}/z_tables_v5t/{P}/*_z.npy")):
                name = os.path.basename(f).replace("_z.npy", "")
                z = np.load(f)
                if name not in lens:
                    continue
                T, Tn = lens[name]
                assert z.shape[0] == T, (name, z.shape, T)
                _, i0, i1, w = grid(T, s)
                np.save(f"{dst}/{name}_z.npy", lerp(z, i0, i1, w).astype(z.dtype))
            for extra in glob.glob(f"{H}/z_tables_v5t/{P}/*.json"):
                shutil.copy(extra, dst)
        print(f"speed {s}: {len(lens)} clips -> {root}")


if __name__ == "__main__":
    main()
