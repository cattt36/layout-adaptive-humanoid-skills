import sys, json
import numpy as np
import mujoco
sys.path.insert(0, ".")
from build_reference import load_robot, fk, sole_points

r = load_robot()
d = np.load(sys.argv[1])
Q, V, A, tau = d["qpos"], d["qvel"], d["qacc"], d["actuator_tau"]
c = d["foot_contact"]; low = d["sole_min_height_m"]
names = r.joint_names
T = len(Q)
print("T", T, "fps", float(d["fps"]))
# joint jumps between frames
dq = np.abs(np.diff(Q[:, 7:], axis=0))
j = np.unravel_index(np.argmax(dq), dq.shape)
print("max joint jump", dq.max(), "frame", j[0], names[j[1]])
big = np.where(dq.max(1) > 0.1)[0]
print("frames with joint jump >0.1 rad:", len(big), big[:30])
print("root jumps >2cm:", np.where(np.linalg.norm(np.diff(Q[:, :3], axis=0), axis=1) > 0.02)[0][:30])
acc = np.abs(A[:, 6:])
k = np.unravel_index(np.argmax(acc), acc.shape)
print("max joint acc", acc.max(), "frame", k[0], names[k[1]])
print("penetration worst frames:", np.argsort(low.min(1))[:10], np.sort(low.min(1))[:10])
for t in np.argsort(low.min(1))[:3]:
    print(" frame", t, "contact", c[t], "low", low[t])
print("contact frac L,R", c.mean(0), "flight frac", (~c.any(1)).mean())
lim = d["actuator_torque_limits"]
ratio = np.abs(tau) / lim
t, j = np.unravel_index(np.argmax(ratio), ratio.shape)
print("worst torque frame", t, names[j], "tau", tau[t, j], "joint acc", A[t, 6 + j], "root lin acc", np.round(A[t, :3], 1), "root ang acc", np.round(A[t, 3:6], 1))
print("root lin acc p99/max", np.percentile(np.linalg.norm(A[:, :3], axis=1), 99), np.linalg.norm(A[:, :3], axis=1).max())
print("root ang acc p99/max", np.percentile(np.linalg.norm(A[:, 3:6], axis=1), 99), np.linalg.norm(A[:, 3:6], axis=1).max())
print("wrist acc max", np.abs(A[:, 6 + np.array([19, 20, 21, 26, 27, 28])]).max())
print("frames with torque>1:", np.where(ratio.max(1) > 1)[0][:60])
