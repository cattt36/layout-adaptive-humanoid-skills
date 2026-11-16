"""Control-relevant prediction targets for z_skill (v5), per trial at the latent frame rate (Dynamics npz, 120 Hz).

future  (robot reference; allowed for every level, it is the NPZ itself):
        for horizons 0.25 / 0.5 / 0.75 / 1.0 s: joint positions (29), both feet relative to the pelvis in the
        current yaw frame (6), pelvis displacement in the current yaw frame (3), pelvis height (1)  -> 4 x 39 = 156
human   (force plates; only levels with human numeric data, L3/L4):
        foot contact L/R, time to the next touchdown L/R, time to the next lift-off L/R (s, clipped to 1 s),
        vertical GRF / body weight, friction demand  -> 8; masked outside the instrumented step span
All targets are z-scored over all trials (unsupervised statistics).  Output: <out>/<trial>.npz with future,
future_mask, human, human_mask and <out>/meta.json (dims, keys).

  python make_ctrl_aux.py --ref <ref_v1> --phase <data/phase> --forceplate <data/forceplate> --annot <LLM_annotation> --out <dir>
"""
import argparse
import csv
import glob
import json
import os
import sys

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "stage1_reference"))
sys.path.insert(0, os.path.join(HERE, "..", "stage5_quality"))
from build_reference import load_robot  # noqa: E402
from make_quality_aux import human_grf  # noqa: E402

HORIZONS = (0.25, 0.5, 0.75, 1.0)
FPS = 120.0


def yaw_of(q):  # wxyz
    return np.arctan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))


def rot2(v, yaw):  # world xy -> yaw frame
    c, s = np.cos(yaw), np.sin(yaw)
    return np.stack([c * v[..., 0] + s * v[..., 1], -s * v[..., 0] + c * v[..., 1]], -1)


def robot_future(r, Q):
    T = len(Q)
    feet = np.zeros((T, 2, 3))
    for t in range(T):
        r.d.qpos[:] = Q[t]
        mujoco.mj_kinematics(r.m, r.d)
        for k, fid in enumerate(r.foot_ids):
            feet[t, k] = r.d.xpos[fid]
    root, yaw, joints = Q[:, :3], yaw_of(Q[:, 3:7]), Q[:, 7:]
    out, mask = [], []
    for h in HORIZONS:
        k = int(round(h * FPS))
        idx = np.minimum(np.arange(T) + k, T - 1)
        ok = (np.arange(T) + k) < T
        rel_f = feet[idx] - root[:, None]
        rel_f = np.concatenate([rot2(rel_f[..., :2], yaw[:, None]), rel_f[..., 2:]], -1).reshape(T, 6)
        disp = root[idx] - root
        disp = np.concatenate([rot2(disp[:, :2], yaw), disp[:, 2:]], -1)
        out.append(np.concatenate([joints[idx], rel_f, disp, root[idx, 2:3]], 1))
        mask.append(np.repeat(ok[:, None], out[-1].shape[1], 1))
    return np.concatenate(out, 1), np.concatenate(mask, 1)


def contact_intervals(phase_dir, task, trial):
    """[(foot 'L'/'R', t_on, t_off)] from the force-plate phase tables (seconds of the mocap trial)."""
    iv = []
    f = glob.glob(os.path.join(phase_dir, task, f"{trial}_ForcePlate_CleanUp_steps.csv"))
    if f:
        for row in csv.DictReader(open(f[0], encoding="utf-8-sig")):
            if row["IC_Time_s"] and row["TO_Time_s"]:
                iv.append((row["Foot"], float(row["IC_Time_s"]), float(row["TO_Time_s"])))
    f = glob.glob(os.path.join(phase_dir, task, f"{trial}_ForcePlate_CleanUp_action_cycles.csv"))
    if f:
        for row in csv.DictReader(open(f[0], encoding="utf-8-sig")):
            if row["PhaseEN"] in ("LeftFootContact", "RightFootContact"):
                iv.append(("L" if row["PhaseEN"].startswith("Left") else "R", float(row["StartTime_s"]), float(row["EndTime_s"])))
    return iv


def human_targets(iv, t, grf):
    T = len(t)
    out = np.zeros((T, 8))
    mask = np.zeros((T, 8), bool)
    if not iv:
        return out, mask
    lo, hi = min(a for _, a, _ in iv), max(b for _, _, b in iv)
    span = (t >= lo - 0.3) & (t <= hi + 0.3)
    for k, foot in enumerate(("L", "R")):
        ons = sorted(a for f, a, _ in iv if f == foot)
        offs = sorted(b for f, _, b in iv if f == foot)
        c = np.zeros(T)
        for f, a, b in iv:
            if f == foot:
                c[(t >= a) & (t <= b)] = 1.0
        out[:, k] = c
        nxt_on = np.array([min([o - x for o in ons if o >= x] or [1.0]) for x in t])
        nxt_off = np.array([min([o - x for o in offs if o >= x] or [1.0]) for x in t])
        out[:, 2 + k] = np.clip(nxt_on, 0, 1.0)
        out[:, 4 + k] = np.clip(nxt_off, 0, 1.0)
    mask[:, :6] = span[:, None]
    if grf is not None:
        out[:, 6:8] = np.nan_to_num(grf[:, :2])
        mask[:, 6:8] = np.isfinite(grf[:, :2])
    return out, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--phase", required=True)
    ap.add_argument("--forceplate", required=True)
    ap.add_argument("--annot", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    r = load_robot()
    raw = {}
    for f in sorted(glob.glob(os.path.join(a.ref, "*", "*_Dynamics.npz"))):
        trial = os.path.basename(f).replace("_Dynamics.npz", "")
        task = trial.rsplit("_", 1)[0]
        d = np.load(f)
        Q, crop0 = d["qpos"], int(d["source_crop"][0])
        fut, fm = robot_future(r, Q)
        t = (crop0 + np.arange(len(Q))) / FPS
        bw = 70.0
        j = os.path.join(a.annot, task, f"{trial}.json")
        if os.path.exists(j):
            bw = float((json.load(open(j, encoding="utf-8-sig")).get("subject_model") or {}).get("body_weight_kg") or 70.0)
        fp = os.path.join(a.forceplate, task, f"{trial}_ForcePlate_CleanUp.csv")
        grf = None
        if os.path.exists(fp):
            G = human_grf(fp, bw, crop0 + len(Q) + 2)
            grf = G[crop0 + np.arange(len(Q))]
        hum, hm = human_targets(contact_intervals(a.phase, task, trial), t, grf)
        raw[trial] = (fut, fm, hum, hm)
    F = np.concatenate([v[0][v[1].all(1)] for v in raw.values()])
    fmu, fsd = F.mean(0), F.std(0) + 1e-6
    Hs = np.concatenate([v[2] for v in raw.values()])
    Hm = np.concatenate([v[3] for v in raw.values()])
    hmu = np.array([Hs[Hm[:, k], k].mean() if Hm[:, k].any() else 0.0 for k in range(8)])
    hsd = np.array([Hs[Hm[:, k], k].std() + 1e-6 if Hm[:, k].any() else 1.0 for k in range(8)])
    for trial, (fut, fm, hum, hm) in raw.items():
        np.savez_compressed(os.path.join(a.out, f"{trial}.npz"),
                            future=((fut - fmu) / fsd).clip(-5, 5).astype(np.float32), future_mask=fm.astype(np.float32),
                            human=((hum - hmu) / hsd).clip(-5, 5).astype(np.float32), human_mask=hm.astype(np.float32))
    meta = dict(future_dim=int(F.shape[1]), human_dim=8, horizons=list(HORIZONS),
                human_keys=["contact_L", "contact_R", "t_next_touchdown_L", "t_next_touchdown_R",
                            "t_next_liftoff_L", "t_next_liftoff_R", "grf_vertical_bw", "grf_friction_demand"],
                trials=len(raw), human_coverage=float(Hm[:, 0].mean()))
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
