"""Offline heuristic risk per 50 Hz clip frame (baseline for Q_exec, the 'offline quality prior' of the paper):
max over the next 1 s of the reference of [ max_j |tau_j| / limit_j  +  5 * joint-limit excess (rad) ]
using the contact-aware inverse dynamics of stage 1."""
import glob, os, sys
import numpy as np

ref_root, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
n = 0
for f in glob.glob(os.path.join(ref_root, "*", "*_Dynamics.npz")):
    d = np.load(f)
    name = os.path.basename(f).replace("_Dynamics.npz", "")
    T50 = np.load(f.replace("_Dynamics.npz", "_motion.npz"))["joint_pos"].shape[0]
    risk = (np.abs(d["actuator_tau"]) / d["actuator_torque_limits"]).max(1) + 5.0 * d["joint_limit_excess"].max(1)
    fps = float(d["fps"])
    h = np.zeros(T50, np.float32)
    for i in range(T50):
        c = int(round(i * fps / 50.0))
        h[i] = risk[c : c + int(fps)].max() if c < len(risk) else risk[-1]
    np.save(os.path.join(out, f"{name}.npy"), h)
    n += 1
print("heuristic tables", n)
