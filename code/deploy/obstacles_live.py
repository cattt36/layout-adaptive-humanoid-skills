"""Obstacle positions from the motion capture -> the policy's obstacle layout.

The policy computes its obstacle features from a layout (stakes / bridge posts) and its own mocap pose.  With this
module the layout is the MEASURED one: bridge posts and stakes are read from the motion capture at the take-over
(command 1) and replace the planned positions, so placement errors -- and every error common to the robot and the
obstacles (mocap origin / axes, aligned_shift_xy) -- no longer reach the policy.  The policy was trained for it: in
training the obstacles were moved at random and perceived where they were (bridge +-5 cm as a whole and every post
+-10 cm along the beam; stakes +-3 cm along / +-6 cm across their line).

Robot side (deployment software, at command 1, motion capture already connected):
    from obstacles_live import live_context
    ctx, report = live_context(pol.meta, layout_dir, z_table, measured)
    print("\\n".join(report)); pol.reset(ctx, state, 0)
  measured = {"posts": [[x, y], ...]} or {"stakes": [[x, y], ...]}: centres in the layout's aligned frame (= mocap +
  aligned_shift_xy, the frame of layout_aligned.json), numbered as in layout_aligned.json (post 1 = the first post the
  robot passes, stakes in stepping order), each the MEAN over ~1 s of frames; None for an obstacle not seen -> its
  planned position moved by the mean offset of the seen ones.  Read once, then keep the context for the whole run.
  Without measurements (measured=None) -> the planned layout (as before).

Command line (check / write the measured positions into the layout files, e.g. from place_live.py):
    python obstacles_live.py <layout_dir> --posts x1,y1 ... | --stakes x1,y1 ... [--write] | --reset
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
import tempfile

import numpy as np

COVER = dict(posts=dict(shift=0.05, along=0.10, lat=0.03), stakes=dict(shift=0.05, along=0.03, lat=0.06))


def _rot(t):
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


class Layout:
    """planned_layouts/<layout>: aligned <-> clip frame, planned posts / stakes, the clip's bridge / stakes files."""

    def __init__(self, L):
        self.L = L.rstrip("/\\")
        self.la = json.load(open(os.path.join(self.L, "layout_aligned.json")))
        mo = np.load(glob.glob(os.path.join(self.L, "*_motion.npz"))[0])
        self.s0 = mo["body_pos_w"][0, 0, :2].astype(float)
        w, x, y, z = mo["body_quat_w"][0, 0].astype(float)
        # start frame = clip frame at the reference pelvis of frame 0, turned to its heading; aligned = start frame
        # rotated by -theta:  aligned = R(-theta - psi0) (clip - start)
        self.th = np.deg2rad(self.la["theta_deg"]) + np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        self.kind = "posts" if self.la.get("posts") else ("stakes" if self.la.get("stakes") else None)
        self.plan = np.array([[p["x"], p["y"]] for p in self.la.get(self.kind, [])]) if self.kind else np.zeros((0, 2))
        clip = self.la["clip"]
        self.bridge_f = os.path.join(self.L, f"{clip}_bridge.json")
        self.stakes_f = os.path.join(self.L, f"{clip}_stakes.json")

    def to_al(self, p):
        return _rot(-self.th) @ (np.asarray(p, float)[:2] - self.s0)

    def to_clip(self, p):
        return _rot(self.th) @ np.asarray(p, float)[:2] + self.s0

    def sheet(self, f):
        """the planned version of a layout file (kept once as <name>_sheet.json)"""
        s = f.replace(".json", "_sheet.json")
        return s if os.path.exists(s) else f


def fill_missing(plan, meas):
    """None entries -> planned position + mean offset of the measured ones"""
    seen = [i for i, m in enumerate(meas) if m is not None]
    off = np.mean([np.asarray(meas[i], float) - plan[i] for i in seen], 0) if seen else np.zeros(2)
    return np.array([np.asarray(m, float) if m is not None else plan[i] + off for i, m in enumerate(meas)]), \
        [i for i, m in enumerate(meas) if m is None]


def check(kind, plan, meas):
    """deviation from the plan and the training coverage -> (lines, ok)"""
    c = COVER[kind]
    d = meas - plan
    shift = d.mean(0)
    res = d - shift
    if kind == "posts":
        ax = plan[-1] - plan[0]
        axes = [ax / np.linalg.norm(ax)] * len(plan)
    else:  # stakes: local line direction (previous -> next stake)
        axes = []
        for i in range(len(plan)):
            v = plan[min(i + 1, len(plan) - 1)] - plan[max(i - 1, 0)]
            axes.append(v / max(np.linalg.norm(v), 1e-6))
    ok = bool(np.all(np.abs(shift) <= c["shift"]))
    name = "post" if kind == "posts" else "stake"
    warn = "" if ok else f"  ** beyond training (+-{100 * c['shift']:.0f} cm) **"
    lines = [f"whole layout shift dx {100 * shift[0]:+5.1f} dy {100 * shift[1]:+5.1f} cm{warn}"]
    for i, (e, r, a) in enumerate(zip(d, res, axes)):
        al, lat = float(r @ a), float(r @ np.array([-a[1], a[0]]))
        b1, b2 = abs(al) > c["along"], abs(lat) > c["lat"]
        ok &= not (b1 or b2)
        lines.append(f"{name} {i + 1}: vs plan dx {100 * e[0]:+6.1f} dy {100 * e[1]:+6.1f} cm | after the common shift along "
                     f"{100 * al:+5.1f}{' **' if b1 else ''} across {100 * lat:+5.1f}{' **' if b2 else ''} cm")
    lines.append("within the training coverage" if ok else
                 f"OUTSIDE the training coverage (shift +-{100 * c['shift']:.0f} cm, {name} along +-{100 * c['along']:.0f} / "
                 f"across +-{100 * c['lat']:.0f} cm): less certain -- move it closer to the plan or check in sim2sim")
    return lines, ok


def bridge_json(lay: Layout, meas, src=None):
    """bridge file with the posts at the measured positions; beam / ramp / leg moved by the rigid 2-D fit"""
    sheet = json.load(open(src or lay.sheet(lay.bridge_f)))
    parts = sheet["parts"]
    names = [k for k in parts if k.startswith("post")]
    pa = {k: lay.to_al(parts[k]["center"]) for k in names}
    match = [min(names, key=lambda k: np.linalg.norm(pa[k] - q)) for q in lay.plan]
    err = max(float(np.linalg.norm(pa[k] - q)) for k, q in zip(match, lay.plan))
    assert len(set(match)) == len(lay.plan) and err < 0.02, f"layout files inconsistent ({100 * err:.1f} cm)"
    P0, P1 = lay.plan - lay.plan.mean(0), meas - meas.mean(0)
    U, _, Vt = np.linalg.svd(P0.T @ P1)
    Rm = (U @ Vt).T
    if np.linalg.det(Rm) < 0:
        Vt[-1] *= -1
        Rm = (U @ Vt).T
    dyaw = float(np.arctan2(Rm[1, 0], Rm[0, 0]))
    for k, v in parts.items():
        c = np.asarray(v["center"], float)
        if k in match:
            xy = lay.to_clip(meas[match.index(k)])
        else:
            xy = lay.to_clip(Rm @ (lay.to_al(c) - lay.plan.mean(0)) + meas.mean(0))
            v["yaw"] = float(v["yaw"]) + dyaw
        v["center"] = [float(xy[0]), float(xy[1]), float(c[2])]
    sheet["measured"] = dict(posts_aligned=meas.tolist())
    return sheet


def stakes_json(lay: Layout, meas, src=None):
    """stakes file with the stakes at the measured positions (stance windows / timing unchanged)"""
    sheet = json.load(open(src or lay.sheet(lay.stakes_f)))
    clip_xy = [lay.to_al([s["x"], s["y"]]) for s in sheet["stakes"]]
    match = [int(np.argmin([np.linalg.norm(c - q) for c in clip_xy])) for q in lay.plan]
    err = max(float(np.linalg.norm(clip_xy[j] - q)) for j, q in zip(match, lay.plan))
    assert len(set(match)) == len(lay.plan) and err < 0.02, f"layout files inconsistent ({100 * err:.1f} cm)"
    for i, j in enumerate(match):
        xy = lay.to_clip(meas[i])
        sheet["stakes"][j]["x"], sheet["stakes"][j]["y"] = float(xy[0]), float(xy[1])
        if j < len(sheet.get("stake_xyz_50hz", [])):
            sheet["stake_xyz_50hz"][j][0], sheet["stake_xyz_50hz"][j][1] = float(xy[0]), float(xy[1])
    sheet["measured"] = dict(stakes_aligned=meas.tolist())
    return sheet


def measured_files(lay: Layout, measured, out_dir):
    """measured dict -> (stakes_json path, bridge_json path, report lines, ok)"""
    sj = lay.stakes_f if os.path.exists(lay.stakes_f) else None
    bj = lay.bridge_f if os.path.exists(lay.bridge_f) else None
    m = (measured or {}).get(lay.kind) if lay.kind else None
    if m is None or all(v is None for v in m):
        return sj, bj, [f"{lay.la['id']}: no {lay.kind} measured -> planned layout"], False
    if len(m) != len(lay.plan):
        raise ValueError(f"{len(m)} {lay.kind} measured, {len(lay.plan)} in the layout")
    meas, missing = fill_missing(lay.plan, m)
    lines, ok = check(lay.kind, lay.plan, meas)
    if missing:
        lines.insert(0, f"not seen: {[i + 1 for i in missing]} -> planned position + mean offset of the others")
    if lay.kind == "posts":
        bj = os.path.join(out_dir, os.path.basename(lay.bridge_f))
        json.dump(bridge_json(lay, meas), open(bj, "w"), indent=1)
    else:
        sj = os.path.join(out_dir, os.path.basename(lay.stakes_f))
        json.dump(stakes_json(lay, meas), open(sj, "w"), indent=1)
    return sj, bj, [f"{lay.la['id']}: measured {lay.kind}"] + lines, ok


def apply_to_dir(layout_dir, measured, target_dir):
    """write the measured obstacle positions into the bridge / stakes file of another directory of the same layout
    (e.g. a planner output / preview directory whose stakes file carries the candidate's planned stance windows); the
    stance windows and everything else of that file are kept.  -> report lines"""
    lay = Layout(layout_dir)
    m = (measured or {}).get(lay.kind) if lay.kind else None
    if m is None or all(v is None for v in m):
        return [f"{lay.la['id']}: no {lay.kind} measured -> planned layout kept in {target_dir}"]
    meas, missing = fill_missing(lay.plan, m)
    lines, _ = check(lay.kind, lay.plan, meas)
    name = os.path.basename(lay.bridge_f if lay.kind == "posts" else lay.stakes_f)
    f = os.path.join(target_dir, name)
    new = bridge_json(lay, meas, src=f) if lay.kind == "posts" else stakes_json(lay, meas, src=f)
    json.dump(new, open(f, "w"), indent=1)
    return [f"{lay.la['id']}: measured {lay.kind} written into {f}"] + lines


def live_context(meta, layout_dir, z_table, measured=None):
    """ClipContext with the measured obstacle positions (see the module docstring) -> (ctx, report lines)"""
    from hnu_deploy import ClipContext
    lay = Layout(layout_dir)
    tmp = tempfile.mkdtemp(prefix="hnu_obst_")
    sj, bj, lines, _ = measured_files(lay, measured, tmp)
    return ClipContext(meta, z_table, sj, bj), lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("layout_dir")
    ap.add_argument("--posts", nargs="*", default=[], help="measured post centres x,y (aligned frame), post 1..n ('-' = not seen)")
    ap.add_argument("--stakes", nargs="*", default=[], help="measured stake centres x,y (aligned frame), stake 1..n")
    ap.add_argument("--write", action="store_true", help="write the measured positions into the layout files (sheet kept)")
    ap.add_argument("--reset", action="store_true", help="restore the planned (sheet) files")
    a = ap.parse_args()
    lay = Layout(a.layout_dir)
    files = [f for f in (lay.bridge_f, lay.stakes_f) if os.path.exists(f)]
    if a.reset:
        for f in files:
            if os.path.exists(f.replace(".json", "_sheet.json")):
                shutil.copy(f.replace(".json", "_sheet.json"), f)
                print("restored", os.path.basename(f))
        return
    vals = a.posts or a.stakes
    meas = [None if v == "-" else [float(x) for x in v.split(",")] for v in vals]
    tmp = tempfile.mkdtemp(prefix="hnu_obst_")
    sj, bj, lines, ok = measured_files(lay, {lay.kind: meas} if vals else None, tmp)
    print("\n".join(lines))
    if a.write and vals:
        for f, new in ((lay.bridge_f, bj), (lay.stakes_f, sj)):
            if new and os.path.dirname(new) == tmp:
                if not os.path.exists(f.replace(".json", "_sheet.json")):
                    shutil.copy(f, f.replace(".json", "_sheet.json"))
                shutil.copy(new, f)
                print("written:", os.path.basename(f), "(planned version kept as", os.path.basename(f.replace(".json", "_sheet.json")) + ")")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
