"""How long does each clip continue after the last stake stance (ThreeStepPoleRun / FiveStepPoleRun) / after leaving the beam (DuMu)?
Also the reference pelvis height / speed at the clip end (is the reference itself still in motion?)."""
import glob
import json
import os

import numpy as np

H = "/root/gpufree-data/hnu_v2/data"
for t, d in (("ThreeStepPoleRun", f"{H}/ref_stakes_s93_h10/ThreeStepPoleRun"), ("FiveStepPoleRun", f"{H}/ref_stakes_s83_h10/FiveStepPoleRun")):
    tails = []
    for f in sorted(glob.glob(f"{d}/*_motion.npz")):
        m = np.load(os.path.realpath(f))
        j = json.load(open(f.replace("_motion.npz", "_stakes.json")))
        T = m["joint_pos"].shape[0]
        end_st = j["stakes"][-1]["stance"][1] / j["fps"]
        v = np.linalg.norm(m["body_lin_vel_w"][-1, 0, :2])
        tails.append(T / 50 - end_st)
        print(f"{os.path.basename(f)[:15]}  length {T / 50:.2f} s  last stake stance ends {end_st:.2f} s  "
              f"-> {T / 50 - end_st:.2f} s after   end pelvis h {m['body_pos_w'][-1, 0, 2]:.2f} m  speed {v:.2f} m/s")
    print(f"== {t}: time after the last stake: min {min(tails):.2f} s, median {np.median(tails):.2f} s, max {max(tails):.2f} s\n")
D = f"{H}/ref_bridge_video_lift/CrouchedTraverse"
for f in sorted(glob.glob(f"{D}/*_motion.npz"))[:6]:
    m = np.load(os.path.realpath(f))
    v = np.linalg.norm(m["body_lin_vel_w"][-1, 0, :2])
    print(f"{os.path.basename(f)[:15]}  length {m['joint_pos'].shape[0] / 50:.2f} s  end pelvis h {m['body_pos_w'][-1, 0, 2]:.2f} m  speed {v:.2f} m/s")
