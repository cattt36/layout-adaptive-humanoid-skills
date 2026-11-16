"""Pre-run placement check: measured stake centres / robot start pose (motion capture, obstacle-aligned
frame of the layout) against planned_layouts/<layout>/layout_aligned.json.

  python check_placement.py <layout_dir> --stakes x1,y1 x2,y2 ... [--start x,y,heading_deg] [--tol 0.05]

Limits ( retraining, sim2sim-tested): every stake within --tol (default 5 cm; trained +-3 cm per stake +
3 cm for the whole layout, stake radius 20 cm), robot start within 5 cm and 10 deg (trained 10 cm / 15 deg, keep a
margin).  The mocap heading calibration (pelvis rigid body) must be done first: its error is not visible to the policy
and moves every foothold sideways by ~1.7 cm per metre and degree.
Exit code 1 if anything is out of tolerance.
"""
import argparse
import json
import os
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("layout_dir")
    ap.add_argument("--stakes", nargs="*", default=[], help="measured stake centres x,y (m), in stake order 1..n")
    ap.add_argument("--start", default=None, help="measured pelvis x,y (m) and heading (deg)")
    ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument("--start-tol", type=float, default=0.05)
    ap.add_argument("--yaw-tol", type=float, default=10.0)
    a = ap.parse_args()
    L = json.load(open(os.path.join(a.layout_dir, "layout_aligned.json")))
    ok = True
    print(f"layout {L['id']} ({L['clip']})")
    if a.start:
        x, y, h = (float(v) for v in a.start.split(","))
        dp = float(np.hypot(x - L["start"]["x"], y - L["start"]["y"]))
        dh = float((h - L["start"]["heading_deg"] + 180) % 360 - 180)
        bad = dp > a.start_tol or abs(dh) > a.yaw_tol
        ok &= not bad
        print(f"  start   offset {100 * dp:5.1f} cm, heading {dh:+5.1f} deg  {'** OUT OF TOLERANCE **' if bad else 'ok'}")
    planned = L.get("stakes", [])
    if a.stakes:
        if len(a.stakes) != len(planned):
            print(f"  ** {len(a.stakes)} stakes measured, {len(planned)} planned **")
            sys.exit(1)
        meas = np.array([[float(v) for v in s.split(",")] for s in a.stakes])
        plan = np.array([[s["x"], s["y"]] for s in planned])
        d = meas - plan
        shift = d.mean(0)
        for i, (s, e) in enumerate(zip(planned, d)):
            bad = float(np.linalg.norm(e)) > a.tol
            ok &= not bad
            print(f"  stake {s['n']} ({s['foot']})  dx {100 * e[0]:+5.1f} dy {100 * e[1]:+5.1f} cm  "
                  f"|d| {100 * np.linalg.norm(e):4.1f} cm  {'** MOVE IT **' if bad else 'ok'}")
        gaps_p = np.linalg.norm(np.diff(plan, axis=0), axis=1)
        gaps_m = np.linalg.norm(np.diff(meas, axis=0), axis=1)
        print(f"  whole layout shift {100 * np.linalg.norm(shift):.1f} cm; stake gaps planned "
              f"{np.round(gaps_p, 2).tolist()} m, measured {np.round(gaps_m, 2).tolist()} m")
    print("RESULT:", "OK" if ok else "NOT OK -- correct the placement before the run")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
