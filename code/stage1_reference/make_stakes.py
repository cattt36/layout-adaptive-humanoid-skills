"""Stake layouts + raised references for ThreeStepPoleRun/2 (stakes: cylinder diameter 0.30 m, height H, as ThreeStepPoleRun/FiveStepPoleRun.usd).

For every trial and every force-plate step of the obstacle CSV (W2: 5, W1: 3):
  * the step's frame is known from the CSV (mocap 120 Hz); within +-0.3 s the corresponding robot foot's planted
    frame (lowest, slowest sole) gives the stake centre = sole centre (ankle + yaw * 0.04 m), z = 0;
  * stake stance interval = frames where that foot's sole centre is within 0.15 m of the stake and low.
Reference: from the first stake stance to the last one the whole body is raised by H; the offset ramps
(smoothstep) during the flight before the first stake and after the last one, so the stance feet never slide.
Outputs: <out>/<task>/<trial>_motion.npz (mjlab format, 50 Hz) and <out>/<task>/<trial>_stakes.json
"""
import argparse, csv, glob, json, os, sys
import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_reference import OUT_FPS, SOLE_Z, export_mjlab_motion, fk, load_robot, resample  # noqa: E402

STAKE_R, FOOT_CENTER = 0.15, 0.04


def read_steps(csv_path, n_steps):
    out = {}
    with open(csv_path, encoding="gbk") as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            steps = []
            for i in range(n_steps):
                b = 1 + 5 * i
                lab = row[b]
                steps.append(("L" if "LToe" in lab else "R", int(float(row[b + 1]))))
            out[row[0]] = steps
    return out


def smoothstep(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3 - 2 * u)


def soles(r, Q):
    sole = np.zeros((len(Q), 2, 3))
    for t in range(len(Q)):
        fk(r, Q[t])
        for k, fid in enumerate(r.foot_ids):
            R = r.d.xmat[fid].reshape(3, 3)
            sole[t, k] = r.d.xpos[fid] + R @ np.array([FOOT_CENTER, 0.0, SOLE_Z])
    return sole


def rescale_stride(Q, sole, k):
    """Robot-scale stride: shorten the horizontal travel of the whole clip to k x, by scaling only the root's
    horizontal displacement during flight / no-planted-foot frames (planted feet do not slide).  The flight factor
    k_f is chosen so that the start-to-end horizontal travel becomes k times the original."""
    dxy = np.diff(Q[:, :2], axis=0)
    planted = (sole[1:, :, 2] < 0.04).any(1)
    D = np.linalg.norm(dxy.sum(0))
    d_fl = np.linalg.norm(dxy[~planted].sum(0)) if (~planted).any() else 0.0
    if d_fl < 1e-6:
        kf = k
        f = np.full(len(dxy), k)
    else:
        kf = float(np.clip((k * D - (D - d_fl)) / d_fl, 0.3, 1.0))
        f = np.where(planted, 1.0, kf)
    Q = Q.copy()
    Q[1:, :2] = Q[0, :2] + np.cumsum(dxy * f[:, None], axis=0)
    return Q, kf


KEEP_S = 0.5  # seconds before the first stake stance that stay exactly as captured (take-off approach)


def _interp_q(Q, s):
    """Sample a qpos trajectory at fractional indices s (linear; root quaternion normalised)."""
    i0 = np.clip(np.floor(s).astype(int), 0, len(Q) - 1)
    i1 = np.clip(i0 + 1, 0, len(Q) - 1)
    w = (s - i0)[:, None]
    q0, q1 = Q[i0], Q[i1].copy()
    flip = (q0[:, 3:7] * q1[:, 3:7]).sum(1) < 0
    q1[flip, 3:7] *= -1
    out = (1 - w) * q0 + w * q1
    out[:, 3:7] /= np.linalg.norm(out[:, 3:7], axis=1, keepdims=True)
    return out


def stretch_runup(Q, fps, first, r0):
    """Lengthen the run-up with a gentler acceleration.  Between the start of motion and KEEP_S before the first
    stake the source is replayed at a rate rising linearly from r0 to 1 (so the robot accelerates over a longer time
    and arrives with the captured approach speed); the standing preparation, the last KEEP_S of the approach and
    everything from the first stake on are unchanged.  Returns (Q_new, frame shift of the later part, info)."""
    v = np.r_[0, np.linalg.norm(np.diff(Q[:, :2], axis=0), axis=1) * fps]
    k = first - int(KEEP_S * fps)
    move = int(np.argmax(v > 0.3))
    if k - move < int(0.3 * fps):
        return Q, 0, None
    D = k - move
    # new time tau(s) = integral of 1 / r(s), r(s) = r0 + (1 - r0) s / D  (source index s within the warped part)
    s_grid = np.linspace(0, D, 4000)
    r = r0 + (1 - r0) * s_grid / D
    tau = np.concatenate([[0], np.cumsum(0.5 * (1 / r[1:] + 1 / r[:-1]) * np.diff(s_grid))])
    n_new = int(round(tau[-1]))
    src = np.interp(np.arange(n_new), tau, s_grid) + move
    Qw = _interp_q(Q, src)
    Qn = np.concatenate([Q[:move], Qw, Q[k:]])
    return Qn, n_new - D, dict(r0=r0, runup_src_s=D / fps, runup_new_s=n_new / fps)


def process(r, ref_dir, name, steps, height, hscale=1.0, tscale=1.0, runup_r0=1.0):
    d = np.load(os.path.join(ref_dir, f"{name}_Dynamics.npz"))
    Q, crop0, fps = d["qpos"].copy(), int(d["source_crop"][0]), float(d["fps"])
    T = len(Q)
    sole = soles(r, Q)
    kf = 1.0
    if hscale < 1.0:
        Q, kf = rescale_stride(Q, sole, hscale)
        sole = soles(r, Q)
    speed = np.r_[0, np.linalg.norm(np.diff(sole[:, :, :2], axis=0), axis=-1).max(1)] * fps
    stakes = []
    for foot, frame in steps:
        k = 0 if foot == "L" else 1
        c = frame - crop0
        lo, hi = max(c - int(0.3 * fps), 0), min(c + int(0.3 * fps), T - 1)
        if hi <= lo:
            continue
        sp = np.linalg.norm(np.diff(sole[lo:hi + 1, k, :2], axis=0), axis=-1) * fps
        score = sole[lo:hi, k, 2] + 0.02 * sp  # low and slow
        t0 = lo + int(np.argmin(score))
        xy = sole[t0, k, :2]
        near = (np.linalg.norm(sole[:, k, :2] - xy, axis=1) < STAKE_R) & (sole[:, k, 2] < 0.06)
        # contiguous interval around t0
        a = b = t0
        while a > 0 and near[a - 1]:
            a -= 1
        while b < T - 1 and near[b + 1]:
            b += 1
        stakes.append(dict(foot=foot, x=float(xy[0]), y=float(xy[1]), stance=[int(a), int(b)], csv_frame=frame))
    if not stakes:
        return None
    first, last = stakes[0]["stance"][0], stakes[-1]["stance"][1]
    # ramp windows = flights adjacent to the stake sequence (no sole near the ground off-stake)
    ground = (sole[:, :, 2] < 0.04).any(1)
    a = first - 1
    while a > 0 and not ground[a]:
        a -= 1
    b = last + 1
    while b < T - 1 and not ground[b]:
        b += 1
    a = min(a, first - 3)
    b = max(b, last + 3)
    off = np.zeros(T)
    for t in range(T):
        if t <= a or t >= b:
            off[t] = 0.0
        elif first <= t <= last:
            off[t] = 1.0
        elif t < first:
            off[t] = smoothstep((t - a) / max(first - a, 1))
        else:
            off[t] = smoothstep((b - t) / max(b - last, 1))
    Q[:, 2] += height * off
    warp = None
    if runup_r0 < 1.0:  # robot-feasible run-up; the stake jumps (from KEEP_S before the first stake on) are untouched
        Q, shift, warp = stretch_runup(Q, fps, first, runup_r0)
        for s in stakes:
            s["stance"] = [s["stance"][0] + shift, s["stance"][1] + shift]
        first, last, a, b = first + shift, last + shift, a + shift, b + shift
    if tscale != 1.0:  # slowed-down reference: treat the source as sampled at fps * tscale (stance times follow)
        fps = fps * tscale
    Q50 = resample(r, Q, fps, OUT_FPS)
    motion = export_mjlab_motion(r, Q50, OUT_FPS)
    src = np.load(os.path.join(ref_dir, f"{name}_motion.npz"))
    motion["foot_contact"] = src["foot_contact"][: len(Q50)] if len(src["foot_contact"]) >= len(Q50) else \
        np.pad(src["foot_contact"], ((0, len(Q50) - len(src["foot_contact"])), (0, 0)), mode="edge")
    if warp is not None:  # timing changed: recompute contact labels (sole within 4 cm of the floor or a stake top)
        s50 = soles(r, Q50)
        ground = np.zeros(s50.shape[:2])
        for s in stakes:
            near = np.hypot(s50[..., 0] - s["x"], s50[..., 1] - s["y"]) < STAKE_R
            ground = np.where(near, height, ground)
        motion["foot_contact"] = (s50[..., 2] - ground < 0.04).astype(src["foot_contact"].dtype)
    meta = dict(trial=name, height=height, radius=STAKE_R, stride_scale=hscale, flight_scale=kf, runup_warp=warp,
                stakes=stakes, ramp=[int(a), int(first), int(last), int(b)],
                fps=fps, stake_xyz_50hz=[[s["x"], s["y"], height / 2] for s in stakes])
    return motion, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["ThreeStepPoleRun", "FiveStepPoleRun"])
    ap.add_argument("--ref", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--height", type=float, default=0.10)
    ap.add_argument("--hscale", type=float, default=1.0, help="robot-scale stride factor (travel x hscale)")
    ap.add_argument("--tscale", type=float, default=1.0, help="playback speed factor (< 1 = slower reference)")
    ap.add_argument("--runup-r0", type=float, default=1.0,
                    help="run-up only: initial replay rate (<1 = longer, gentler acceleration); stake jumps unchanged")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    r = load_robot()
    steps = read_steps(a.csv, 5 if a.task == "FiveStepPoleRun" else 3)
    os.makedirs(os.path.join(a.out, a.task), exist_ok=True)
    for f in sorted(glob.glob(os.path.join(a.ref, "*_motion.npz"))):
        name = os.path.basename(f).replace("_motion.npz", "")
        if name not in steps:
            print(name, "no csv row"); continue
        res = process(r, a.ref, name, steps[name], a.height, a.hscale, a.tscale, a.runup_r0)
        if res is None:
            print(name, "no stakes"); continue
        motion, meta = res
        np.savez_compressed(os.path.join(a.out, a.task, f"{name}_motion.npz"), **motion)
        json.dump(meta, open(os.path.join(a.out, a.task, f"{name}_stakes.json"), "w"), indent=1)
        gaps = np.diff([[s["x"], s["y"]] for s in meta["stakes"]], axis=0)
        print(name, "stakes", len(meta["stakes"]), "stance frames", [s["stance"][1] - s["stance"][0] + 1 for s in meta["stakes"]],
              "steps (dx,dy)", np.round(gaps, 2).tolist())


if __name__ == "__main__":
    main()
