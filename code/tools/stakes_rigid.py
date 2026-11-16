"""DEMO (user: do not change the task): keep the HUMAN's stake pattern (spacing, angles, foot order) and only move
the whole pattern rigidly (2-D translation + rotation, least squares) onto the robot's calibrated stakes (= mean landings,
tools/calib_stakes.py).  Prints the residual per stake (how much the per-stake calibration deformed the pattern).
  python stakes_rigid.py <ref_stakes_json> <calibrated_stakes_json> <out_json> [rotate=1]"""
import json
import sys

import numpy as np

rj, cj, out = sys.argv[1:4]
ROT = (sys.argv[4] if len(sys.argv) > 4 else "1") == "1"
r, c = json.load(open(rj)), json.load(open(cj))
P = np.array([[s["x"], s["y"]] for s in r["stakes"]])
Q = np.array([[s["x"], s["y"]] for s in c["stakes"]])
pm, qm = P.mean(0), Q.mean(0)
if ROT:  # Kabsch 2-D
    H = (P - pm).T @ (Q - qm)
    U, _, Vt = np.linalg.svd(H)
    Rm = Vt.T @ U.T
    if np.linalg.det(Rm) < 0:
        Vt[-1] *= -1
        Rm = Vt.T @ U.T
else:
    Rm = np.eye(2)
f = lambda p: (np.asarray(p) - pm) @ Rm.T + qm  # noqa: E731
N = f(P)
o = json.loads(json.dumps(r))
for k, st in enumerate(o["stakes"]):
    st["x"], st["y"] = float(N[k, 0]), float(N[k, 1])
    o["stake_xyz_50hz"][k][0], o["stake_xyz_50hz"][k][1] = float(N[k, 0]), float(N[k, 1])
o["calibrated"] = f"reference stake pattern moved rigidly onto {cj} (rotation {ROT})"
json.dump(o, open(out, "w"), indent=1)
print(f"rotation {np.degrees(np.arctan2(Rm[1, 0], Rm[0, 0])):+.2f} deg, shift {np.linalg.norm(qm - pm):.3f} m")
for k in range(len(P)):
    print(f"stake {k + 1} ({r['stakes'][k]['foot']}): rigid vs calibrated {np.linalg.norm(N[k] - Q[k]):.3f} m, moved from reference {np.linalg.norm(N[k] - P[k]):.3f} m")
