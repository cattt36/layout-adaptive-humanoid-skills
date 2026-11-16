"""robot health check from one run of the robot team's logger (data_*.csv) -- every G1 is different
(weak left ankle pitch, waist folding onto its stop), so check the robot of the day BEFORE the tests.

  python robot_health.py data_XXXX.csv [more.csv ...]

Per joint, separately for the robot on the rope (modes ramp / hold / prepare) and on the policy (stand / run / ...):
  ratio   median of tau_est / tau_cmd, tau_cmd = kp (q_des - q) - kd dq + tau_ff (logged kp / kd / tau_ff), only
          samples with |tau_cmd| > 3 Nm and below 90 % of the effort limit.  Healthy ~0.8-1.25.
  sat     % of samples with |tau_cmd| >= the effort limit (the motor cannot give what the PD asks)
  stop    % of samples within 0.03 rad of the mechanical joint limit
  err     median |q_des - q| (rad);  temp  max motor temperature (deg C)
Flags (!) are printed below the table.  The policy modes need some seconds of policy control (standard-start hold);
a run that is only on the rope checks the motors and encoders only.
"""
import csv
import sys

import numpy as np

RANGE = {  # G1 29-dof (mjlab g1.xml)
    "left_hip_pitch_joint": (-2.5307, 2.8798), "left_hip_roll_joint": (-0.5236, 2.9671), "left_hip_yaw_joint": (-2.7576, 2.7576),
    "left_knee_joint": (-0.087267, 2.8798), "left_ankle_pitch_joint": (-0.87267, 0.5236), "left_ankle_roll_joint": (-0.2618, 0.2618),
    "right_hip_pitch_joint": (-2.5307, 2.8798), "right_hip_roll_joint": (-2.9671, 0.5236), "right_hip_yaw_joint": (-2.7576, 2.7576),
    "right_knee_joint": (-0.087267, 2.8798), "right_ankle_pitch_joint": (-0.87267, 0.5236), "right_ankle_roll_joint": (-0.2618, 0.2618),
    "waist_yaw_joint": (-2.618, 2.618), "waist_roll_joint": (-0.52, 0.52), "waist_pitch_joint": (-0.52, 0.52),
    "left_shoulder_pitch_joint": (-3.0892, 2.6704), "left_shoulder_roll_joint": (-1.5882, 2.2515), "left_shoulder_yaw_joint": (-2.618, 2.618),
    "left_elbow_joint": (-1.0472, 2.0944), "left_wrist_roll_joint": (-1.97222, 1.97222), "left_wrist_pitch_joint": (-1.61443, 1.61443),
    "left_wrist_yaw_joint": (-1.61443, 1.61443), "right_shoulder_pitch_joint": (-3.0892, 2.6704),
    "right_shoulder_roll_joint": (-2.2515, 1.5882), "right_shoulder_yaw_joint": (-2.618, 2.618), "right_elbow_joint": (-1.0472, 2.0944),
    "right_wrist_roll_joint": (-1.97222, 1.97222), "right_wrist_pitch_joint": (-1.61443, 1.61443), "right_wrist_yaw_joint": (-1.61443, 1.61443),
}
EFFORT = {"hip_pitch": 88, "hip_roll": 139, "hip_yaw": 88, "knee": 139, "ankle": 50, "waist_yaw": 88, "waist_roll": 50,
          "waist_pitch": 50, "shoulder": 25, "elbow": 25, "wrist_roll": 25, "wrist_pitch": 5, "wrist_yaw": 5}
ROPE_MODES = ("ramp", "hold", "prepare", "preloco")


def effort(j):
    for k, v in EFFORT.items():
        if k in j:
            return v
    return 25


def load(f):
    with open(f, newline="") as fh:
        r = csv.reader(fh)
        head = next(r)
        rows = [x for x in r if len(x) == len(head)]
    col = {h: i for i, h in enumerate(head)}
    mode = np.array([x[col["mode"]] for x in rows])

    def num(h):
        return np.array([float(x[col[h]]) for x in rows]) if h in col else None
    return mode, num


def stats(j, s, num):
    q, qd, dq, tau = num(f"q.{j}")[s], num(f"q_des.{j}")[s], num(f"dq.{j}")[s], num(f"tau_est.{j}")[s]
    kp, kd = num(f"kp.{j}")[s], num(f"kd.{j}")[s]
    ff = num(f"tau_ff.{j}")
    ff = ff[s] if ff is not None else 0.0
    temp = num(f"temp.{j}")
    cmd = kp * (qd - q) - kd * dq + ff
    e = effort(j)
    ok = (np.abs(cmd) > 3.0) & (np.abs(cmd) < 0.9 * e)
    ratio = float(np.median(tau[ok] / cmd[ok])) if ok.sum() > 20 else float("nan")
    lo, hi = RANGE[j]
    return dict(ratio=ratio, n_ratio=int(ok.sum()), sat=100 * float(np.mean(np.abs(cmd) >= e)),
                stop=100 * float(np.mean((q < lo + 0.03) | (q > hi - 0.03))), err=float(np.median(np.abs(qd - q))),
                err_max=float(np.max(np.abs(qd - q))), temp=float(np.max(temp[s])) if temp is not None else float("nan"),
                q_mean=float(np.mean(q)), qd_mean=float(np.mean(qd)))


def report(f):
    mode, num = load(f)
    groups = [("rope", np.isin(mode, ROPE_MODES)), ("policy", ~np.isin(mode, ROPE_MODES))]
    print(f"== {f}  (modes: {', '.join(f'{m} {int((mode == m).sum())}' for m in dict.fromkeys(mode))})")
    flags = []
    for g, s in groups:
        if s.sum() < 50:
            continue
        print(f"\n  [{g}]  {s.sum()} samples\n  {'joint':26s} {'ratio':>6s} {'sat%':>5s} {'stop%':>5s} {'err':>5s} {'errmax':>6s} {'temp':>5s}")
        for j in RANGE:
            if num(f"q.{j}") is None:
                continue
            r = stats(j, s, num)
            print(f"  {j[:-6]:26s} {r['ratio']:6.2f} {r['sat']:5.1f} {r['stop']:5.1f} {r['err']:5.2f} {r['err_max']:6.2f} {r['temp']:5.0f}")
            name = f"{j[:-6]} ({g})"
            if r["n_ratio"] > 20 and not 0.7 <= r["ratio"] <= 1.4:
                flags.append(f"{name}: measured torque = {r['ratio']:.2f} x the command (motor / gearbox / torque sensor?)")
            if r["sat"] > 5:
                flags.append(f"{name}: command at the effort limit {r['sat']:.0f} % of the time (too weak for the load)")
            if r["stop"] > 2 and g == "policy":
                flags.append(f"{name}: at the mechanical limit {r['stop']:.0f} % of the time")
            if r["temp"] > 70:
                flags.append(f"{name}: {r['temp']:.0f} deg C")
            if j.startswith("waist") and j != "waist_yaw_joint" and abs(r["q_mean"] - r["qd_mean"]) > 0.10:
                flags.append(f"{name}: mean angle {r['q_mean']:+.2f} vs target {r['qd_mean']:+.2f} rad (waist sags; robot)")
        # left / right asymmetry of the leg torque ratios
        for k in ("hip_pitch", "hip_roll", "knee", "ankle_pitch", "ankle_roll"):
            a, b = stats(f"left_{k}_joint", s, num), stats(f"right_{k}_joint", s, num)
            if a["n_ratio"] > 20 and b["n_ratio"] > 20 and abs(a["ratio"] - b["ratio"]) > 0.3:
                flags.append(f"{k} ({g}): left {a['ratio']:.2f} vs right {b['ratio']:.2f} (asymmetric)")
    print("\n  FLAGS:" if flags else "\n  no flags")
    for x in flags:
        print("   ! " + x)
    print()


if __name__ == "__main__":
    for f in sys.argv[1:]:
        report(f)
