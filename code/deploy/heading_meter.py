"""DEMO helper (no mocap): start heading of the blind replay trackers.

TrackerPolicy maps the IMU yaw at the takeover to the reference start heading, so a placement heading error delta
(robot turned delta to the LEFT of the floor x line) rotates the whole path by delta.  If delta is known, pass
start_yaw = reference start yaw + delta: the policy then SEES the error and turns back by itself (trained with +-3 deg;
sim2sim , tools/hcomp*.sh: 3-10 deg errors, measurement error +-1 deg -> stakes 84-98 %, bridge 100 %).

Two ways to know delta:
  A. straightedge (every run): 1 m straightedge along the outer edge of the clip's "straight foot"; y of its front and
     rear end relative to the x line (left = +) -> delta_from_straightedge(y_front, y_rear, length)
  B. IMU (recommended, until the next reboot): set_floor(quat, delta0) once; later delta = IMU yaw - floor heading,
     computed automatically (check the IMU drift with drift_test first).  final: the bridge and the stakes
     are placed by tape coordinates, so align the robot CAREFULLY for set_floor and pass delta0 from the straightedge
     (demo_loop_example.run(..., first_delta_deg=...) stores it at the first takeover); an error in delta0 acts like an
     uncompensated start error (bridge: +-3 deg tolerated; stakes: absorbed by the whole-pattern shift).

    from heading_meter import HeadingMeter, delta_from_straightedge
    hm = HeadingMeter("heading_floor.json")
    hm.set_floor(read_state().quat, delta_from_straightedge(0.312, 0.297, 1.0))   # once (method B)
    d = hm.delta_deg(read_state().quat)                                            # later runs
    pol.reset(motion_npz, s0, start_yaw=hm.start_yaw(ref_start_yaw, s0.quat))       # compensated start
read_state() -> RobotState (quat = pelvis IMU attitude, wxyz), the same as in demo_loop_example.py."""
import json
import os
import time

import numpy as np

from hnu_deploy import yaw_of

MAX_COMP_DEG = 10.0  # verified range in sim (larger errors: re-place the robot)


def wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0


def delta_from_straightedge(y_front, y_rear, length=1.0):
    """heading error (deg, + = robot turned LEFT) from the y (m, left +) of the straightedge's front / rear end."""
    return float(np.degrees(np.arctan2(y_front - y_rear, length)))


class HeadingMeter:
    def __init__(self, path="heading_floor.json"):
        self.path = path
        self.floor = None
        if os.path.exists(path):
            j = json.load(open(path))
            self.floor = j["floor_yaw_deg"]
            print(f"floor heading {self.floor:+.2f} deg stored {j.get('time')} -- only valid if the robot was NOT rebooted "
                  f"since then (otherwise delete {path}: the next takeover stores a new one, re-calibrate the stakes)")

    def set_floor(self, quat, delta0_deg=0.0):
        """store the IMU yaw of the floor x direction: robot heading now = delta0_deg left of the x line."""
        self.floor = wrap_deg(float(np.degrees(yaw_of(np.asarray(quat, float)))) - delta0_deg)
        json.dump({"floor_yaw_deg": self.floor, "time": time.strftime("%Y-%m-%d %H:%M:%S")}, open(self.path, "w"))
        print(f"floor heading stored (IMU yaw of +x): {self.floor:+.2f} deg -> {self.path}")

    def delta_deg(self, quat):
        """robot heading - floor x direction (deg); > 0 = robot turned to the LEFT (counter-clockwise from above)."""
        assert self.floor is not None, "no floor heading: call set_floor() after one careful alignment"
        return wrap_deg(float(np.degrees(yaw_of(np.asarray(quat, float)))) - self.floor)

    def start_yaw(self, ref_start_yaw, quat=None, delta_deg=None):
        """start_yaw for TrackerPolicy.reset: reference start heading + the measured error (from the IMU, or given)."""
        d = self.delta_deg(quat) if delta_deg is None else float(delta_deg)
        if abs(d) > MAX_COMP_DEG:
            raise ValueError(f"heading error {d:+.1f} deg > {MAX_COMP_DEG} deg: re-place the robot")
        print(f"heading compensation: {d:+.2f} deg")
        return float(ref_start_yaw) + np.radians(d)

    def monitor(self, read_state, tol=3.0, hz=5.0):
        """print the heading error while the robot is placed (generator; stop when satisfied)."""
        while True:
            d = self.delta_deg(read_state().quat)
            hint = "OK" if abs(d) <= tol else ("turn RIGHT 向右转" if d > 0 else "turn LEFT 向左转")
            print(f"\rheading error {d:+6.2f} deg  {hint:<20}", end="", flush=True)
            yield d
            time.sleep(1.0 / hz)

    @staticmethod
    def drift_test(read_state, minutes=5.0):
        """robot standing still: IMU yaw change over `minutes` (deg)."""
        y0 = float(np.degrees(yaw_of(np.asarray(read_state().quat, float))))
        t0, y = time.time(), y0
        while time.time() - t0 < 60.0 * minutes:
            time.sleep(10.0)
            y = float(np.degrees(yaw_of(np.asarray(read_state().quat, float))))
            print(f"{(time.time() - t0) / 60:5.1f} min: yaw drift {wrap_deg(y - y0):+.2f} deg")
        return wrap_deg(y - y0)
