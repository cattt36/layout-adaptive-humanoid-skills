"""Post-aware CrouchedTraverse references: the retargeted reference passes 0.10-0.20 m through the DuMuQiao.usd support
posts (the actor weaves between them; the robot, with its own body proportions, does not).  Kinematic re-planning:
for every frame the whole body is shifted horizontally away from the nearest post (perpendicular to the plank line,
on the side the body already is) by the amount needed to clear it by MARGIN; the shift is smoothed in time
(Gaussian, sigma 0.2 s) and re-checked with exact geom distances (iterated).

  python make_detour_ref.py <ref_dir> <bridge_json_dir> <out_motion_dir>
Writes <clip>_motion.npz (mjlab, 50 Hz) + a copy of <clip>_bridge.json with the new reference clearances.
"""
import glob
import json
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from make_bridge_usd import model  # noqa: E402
from make_bridge import min_dist  # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage1_reference"))
from build_reference import OUT_FPS, export_mjlab_motion, load_robot, resample  # noqa: E402

MARGIN = 0.04


def per_frame_post_dist(m, d, obs, rob, Q):
    ft = np.zeros(6)
    out = np.full((len(Q), len(obs)), 9.0)
    for t, q in enumerate(Q):
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        for j, o in enumerate(obs):
            for g in rob:
                out[t, j] = min(out[t, j], mujoco.mj_geomDistance(m, d, g, o, 0.6, ft))
    return out


def gauss_smooth(x, sigma):
    if sigma <= 0:
        return x
    k = np.arange(-int(3 * sigma), int(3 * sigma) + 1)
    w = np.exp(-0.5 * (k / sigma) ** 2)
    w /= w.sum()
    pad = np.pad(x, ((len(k) // 2, len(k) // 2), (0, 0)), mode="edge")
    return np.stack([np.convolve(pad[:, i], w, mode="valid") for i in range(x.shape[1])], 1)


def main():
    ref_dir, jdir, out = sys.argv[1:4]
    os.makedirs(out, exist_ok=True)
    r = load_robot()
    for j in sorted(glob.glob(os.path.join(jdir, "*_bridge.json"))):
        meta = json.load(open(j))
        if meta["fit_err_m"] > 0.1:
            continue
        name = meta["trial"]
        dd = np.load(os.path.join(ref_dir, f"{name}_Dynamics.npz"))
        Q, fps = dd["qpos"].copy(), float(dd["fps"])
        parts = meta["parts"]
        posts = {k: v for k, v in parts.items() if k.startswith("post")}
        plank = parts["plank"]
        ax = np.array([np.cos(plank["yaw"]), np.sin(plank["yaw"])])  # plank direction (robot xy)
        nrm = np.array([-ax[1], ax[0]])
        m, d, obs, rob = model(posts)
        names = list(posts)
        oid = [obs[k] for k in names]
        shift = np.zeros((len(Q), 2))
        for it in range(6):
            Qs = Q.copy()
            Qs[:, :2] += shift
            dist = per_frame_post_dist(m, d, oid, rob, Qs[::2])
            dist = np.repeat(dist, 2, axis=0)[: len(Q)]
            need = np.zeros((len(Q), 2))
            for jj, k in enumerate(names):
                c = np.array(posts[k]["center"][:2])
                lack = np.clip(MARGIN - dist[:, jj], 0, None)
                side = np.sign((Qs[:, :2] - c) @ nrm)
                side[side == 0] = 1.0
                need += (lack * side)[:, None] * nrm[None]
            if np.abs(need).max() < 1e-3:
                break
            shift += gauss_smooth(need * 1.5, 0.2 * fps)
        Qs = Q.copy()
        Qs[:, :2] += shift
        dist = per_frame_post_dist(m, d, oid, rob, Qs[::2]).min(0)
        Q50 = resample(r, Qs, fps, OUT_FPS)
        motion = export_mjlab_motion(r, Q50, OUT_FPS)
        src = np.load(os.path.join(ref_dir, f"{name}_motion.npz"))
        n = len(Q50)
        fc = src["foot_contact"]
        motion["foot_contact"] = fc[:n] if len(fc) >= n else np.pad(fc, ((0, n - len(fc)), (0, 0)), mode="edge")
        np.savez_compressed(os.path.join(out, f"{name}_motion.npz"), **motion)
        meta["detour"] = dict(max_shift_m=float(np.abs(shift @ nrm).max()), post_min_dist_after_m=dict(zip(names, map(float, dist))))
        json.dump(meta, open(os.path.join(out, f"{name}_bridge.json"), "w"), indent=1)
        print(f"{name}: max lateral detour {np.abs(shift @ nrm).max():.3f} m; post clearance after "
              + " ".join(f"{k} {v:+.3f}" for k, v in zip(names, dist)), flush=True)


if __name__ == "__main__":
    main()
