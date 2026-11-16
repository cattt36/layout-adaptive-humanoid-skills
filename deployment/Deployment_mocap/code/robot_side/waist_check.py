"""waist roll / pitch check of a robot run (data_*.csv of the robot team's logger).

  python waist_check.py data_H_154536_x86_D1.csv [more.csv ...]

Per control mode (ramp / hold / preloco / run ...): waist angle, target, measured torque (tau_est) and the effective
stiffness tau / (target - angle).  A healthy waist on PD gives an effective stiffness close to the commanded kp; the 
runs gave ~28 Nm/rad in "run" (= the policy kp 28.5) but only ~2-4 Nm/rad in "hold" (KP_PREP 40) -- check this.
"at stop" = |angle| > 0.50 rad (mechanical limit 0.52 rad): the upper body is folded onto the stop.
"""
import csv
import sys

import numpy as np


def load(f):
    with open(f, newline="") as fh:
        r = csv.reader(fh)
        head = next(r)
        rows = [x for x in r if len(x) == len(head)]
    col = {h: i for i, h in enumerate(head)}
    mode = np.array([x[col["mode"]] for x in rows])
    num = lambda h: np.array([float(x[col[h]]) for x in rows])  # noqa: E731
    return mode, num


def report(f):
    mode, num = load(f)
    print(f"== {f}")
    for j in ("waist_pitch_joint", "waist_roll_joint"):
        q, qd, tau = num(f"q.{j}"), num(f"q_des.{j}"), num(f"tau_est.{j}")
        for m in dict.fromkeys(mode):  # modes in order of appearance
            s = mode == m
            if s.sum() < 10:
                continue
            err = qd[s] - q[s]
            big = np.abs(err) > 0.05
            k_eff = np.median(tau[s][big] / err[big]) if big.sum() > 10 else float("nan")
            print(f"  {j[6:-6]:5s} {m:8s} n {s.sum():6d}  angle mean {q[s].mean():+.2f} max {q[s].max():+.2f} min {q[s].min():+.2f}"
                  f"  target mean {qd[s].mean():+.2f}  |tau| max {np.abs(tau[s]).max():5.1f} Nm"
                  f"  stiffness {k_eff:6.1f} Nm/rad  at stop {100 * np.mean(np.abs(q[s]) > 0.50):4.1f} %")


if __name__ == "__main__":
    for f in sys.argv[1:]:
        report(f)
