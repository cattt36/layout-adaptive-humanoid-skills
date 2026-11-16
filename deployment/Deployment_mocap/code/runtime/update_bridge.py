"""Measured bridge posts -> layout bridge file.

The policy perceives the bridge from planned_layouts/<layout>/<clip>_bridge.json.  If the real posts are not where
that file says, the robot threads between posts that are somewhere else: in sim2sim (D1, posts measured  =
sheet + 14 / 22 / 27 / 36 cm along the beam) success falls from 97 % to 8 % when the file keeps the sheet positions,
83 % of the trials hit post 2 with the right arm.  So after every (re)placement of the bridge: measure the posts with
the motion capture and write them into the layout with this script.

  python update_bridge.py <layout_dir> --posts x1,y1 x2,y2 x3,y3 x4,y4          # check only
  python update_bridge.py <layout_dir> --posts x1,y1 x2,y2 x3,y3 x4,y4 --write  # check + update the bridge file
  python update_bridge.py <layout_dir> --reset                                   # back to the sheet positions

Posts: centre of each post (top centre), in the layout's obstacle-aligned frame (the frame of layout_aligned.json and
check_placement.py: origin = planned robot start, +x = obstacle line), numbered as in layout_aligned.json (post 1 =
the first post the robot passes).  The beam, ramp and leg follow the posts (rigid 2-D fit); the post bases follow
their posts automatically.

Training coverage (student rtGm): whole bridge +-5 cm in x and y, every post +-10 cm along the beam, posts on the
beam line.  Outside it the file is still written (--write), but the run is less certain: check it in sim2sim first
(command printed at the end).  Exit code: 0 within coverage, 1 outside / inconsistent measurement.
"""
import argparse
import glob
import json
import os
import shutil
import sys
import time

import numpy as np

SHIFT_MAX, ALONG_MAX, LAT_MAX = 0.05, 0.10, 0.03  # m


def rot(t):
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("layout_dir")
    ap.add_argument("--posts", nargs="*", default=[], help="measured post centres x,y (m), aligned frame, post 1..n")
    ap.add_argument("--write", action="store_true", help="write the measured posts into the bridge file")
    ap.add_argument("--reset", action="store_true", help="restore the sheet bridge file")
    a = ap.parse_args()
    L = a.layout_dir.rstrip("/\\")
    la = json.load(open(os.path.join(L, "layout_aligned.json")))
    bf = glob.glob(os.path.join(L, "*_bridge.json"))
    bf = [f for f in bf if not f.endswith("_bridge_sheet.json")]
    if not bf or "posts" not in la:
        sys.exit(f"{L}: no bridge in this layout")
    bf = bf[0]
    sheet_f = bf.replace("_bridge.json", "_bridge_sheet.json")
    if not os.path.exists(sheet_f):  # keep the planned (sheet) file once
        shutil.copy(bf, sheet_f)
    if a.reset:
        shutil.copy(sheet_f, bf)
        print(f"{os.path.basename(bf)} restored from {os.path.basename(sheet_f)} (sheet positions)")
        return
    sheet = json.load(open(sheet_f))
    parts = sheet["parts"]

    # aligned <-> clip frame: start frame = clip frame moved to the reference pelvis at frame 0 and turned to its
    # heading psi0; aligned = start frame rotated by -theta:  aligned = R(-theta - psi0) (clip - start)
    mo = np.load(glob.glob(os.path.join(L, "*_motion.npz"))[0])
    s0 = mo["body_pos_w"][0, 0, :2].astype(float)
    w, x, y, z = mo["body_quat_w"][0, 0].astype(float)
    psi0 = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    th = np.deg2rad(la["theta_deg"]) + psi0
    to_al = lambda p: rot(-th) @ (np.asarray(p, float)[:2] - s0)  # noqa: E731
    to_clip = lambda p: rot(th) @ np.asarray(p, float) + s0  # noqa: E731

    # sheet posts (aligned, numbered 1..n) and the matching post parts of the bridge file
    plan = np.array([[p["x"], p["y"]] for p in la["posts"]])
    names = [k for k in parts if k.startswith("post")]
    pa = {k: to_al(parts[k]["center"]) for k in names}
    match = [min(names, key=lambda k: np.linalg.norm(pa[k] - q)) for q in plan]
    err = max(float(np.linalg.norm(pa[k] - q)) for k, q in zip(match, plan))
    assert len(set(match)) == len(plan) and err < 0.02, f"layout files inconsistent ({100 * err:.1f} cm)"
    if not a.posts:
        sys.exit("give --posts (or --reset)")
    meas = np.array([[float(v) for v in s.split(",")] for s in a.posts])
    if len(meas) != len(plan):
        sys.exit(f"{len(meas)} posts measured, {len(plan)} in the layout")

    # beam axis (aligned frame) from the sheet posts; deviation = common shift + per-post residual (along / across)
    ax = plan[-1] - plan[0]
    ax = ax / np.linalg.norm(ax)
    nx = np.array([-ax[1], ax[0]])
    d = meas - plan
    shift = d.mean(0)
    res = d - shift
    ok = True
    print(f"layout {la['id']} ({la['clip']}), {os.path.basename(bf)}")
    print(f"  whole bridge shift  dx {100 * shift[0]:+5.1f}  dy {100 * shift[1]:+5.1f} cm"
          f"   {'ok' if np.all(np.abs(shift) <= SHIFT_MAX) else '** beyond training (+-5 cm) **'}")
    ok &= bool(np.all(np.abs(shift) <= SHIFT_MAX))
    for i, (e, r) in enumerate(zip(d, res)):
        al, lat = float(r @ ax), float(r @ nx)
        bad_al, bad_lat = abs(al) > ALONG_MAX, abs(lat) > LAT_MAX
        ok &= not (bad_al or bad_lat)
        print(f"  post {i + 1}  vs sheet dx {100 * e[0]:+6.1f} dy {100 * e[1]:+6.1f} cm | after the common shift: along the beam "
              f"{100 * al:+5.1f} cm{' ** beyond training (+-10 cm) **' if bad_al else ''}, across {100 * lat:+5.1f} cm"
              f"{' ** off the beam line: re-measure **' if bad_lat else ''}")
    gp, gm = np.linalg.norm(np.diff(plan, axis=0), axis=1), np.linalg.norm(np.diff(meas, axis=0), axis=1)
    print(f"  post gaps  sheet {np.round(gp, 3).tolist()} m, measured {np.round(gm, 3).tolist()} m")
    if not ok:
        print("  -> outside the training coverage: preferably move the bridge back to the sheet, else check in sim2sim")

    if a.write:
        # rigid 2-D fit sheet -> measured (Kabsch) for the beam / ramp / leg; posts exactly at the measured positions
        P0, P1 = plan - plan.mean(0), meas - meas.mean(0)
        U, _, Vt = np.linalg.svd(P0.T @ P1)
        Rm = (U @ Vt).T
        if np.linalg.det(Rm) < 0:
            Vt[-1] *= -1
            Rm = (U @ Vt).T
        dyaw = float(np.arctan2(Rm[1, 0], Rm[0, 0]))
        fit = lambda p: Rm @ (p - plan.mean(0)) + meas.mean(0)  # noqa: E731
        out = json.loads(json.dumps(sheet))
        for k, v in out["parts"].items():
            c = np.asarray(v["center"], float)
            if k in match:
                xy = to_clip(meas[match.index(k)])
            else:
                xy = to_clip(fit(to_al(c)))
                v["yaw"] = float(v["yaw"]) + dyaw
            v["center"] = [float(xy[0]), float(xy[1]), float(c[2])]
        out["measured"] = dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), posts_aligned=meas.tolist(),
                               note="posts measured with the motion capture (tools: update_bridge.py); sheet in "
                                    + os.path.basename(sheet_f))
        json.dump(out, open(bf, "w"), indent=1)
        print(f"  written: {os.path.basename(bf)} (beam / ramp / leg moved by the fit, yaw {np.degrees(dyaw):+.2f} deg); "
              f"sheet kept in {os.path.basename(sheet_f)}")
        print("  check in sim2sim (8 trials, ~1 min):\n    python code/runtime/sim2sim.py model --waist-lock --latency 0-40 "
              "--crit task --trials 8 --start rest --hold 12 --rope 0.8,10 --start-rp 3 --start-xy 0.05 --start-yaw 5 "
              f"--dr-x --head-bias 3 --height-bias 0.02 --motion-dirs {L} --latent-dir {L}")
    print("RESULT:", "OK (within the training coverage)" if ok else "OUTSIDE the training coverage")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
