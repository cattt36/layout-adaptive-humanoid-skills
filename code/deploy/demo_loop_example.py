"""DEMO bundle: how the robot loop drives a blind replay tracker (hnu_deploy.TrackerPolicy).  NO mocap: the
state comes from the robot itself (IMU quaternion + gyro, joint encoders); pos / lin_vel are not used by the tracker.

Same RobotState convention as the earlier bundles (StudentPolicy): quat = pelvis IMU attitude (wxyz), ang_vel = IMU gyro
(body frame), q / dq = joint angles / velocities in meta["joint_names"] order.  PD: tau = kp (q_target - q) - kd dq with
meta["kp"], meta["kd"], |tau| <= meta["effort_limit"], 50 Hz policy (meta["sim"]["control_dt"] = 0.02 s).

Sequence (rope start, as in training):
  1. robot on the rope at the floor mark, facing +x (the obstacles); take over -> reset(); frame 0 is HELD
     (holding=True: reference velocities 0) while the rope is released over ~10 s; wait 1-2 s after the rope is slack
  2. start -> holding=False, frame += 1 every control step (the reference itself does the step-in and the slow-in)
  3. after the last frame: hold the last frame (holding=True) and hand over to the damping / stand controller + rope

heading compensation (heading_meter.py): pass the measured start heading error (deg, + = robot turned LEFT
of the floor x line) or a HeadingMeter (IMU method); the policy then turns back by itself.
"""
import numpy as np

from heading_meter import MAX_COMP_DEG
from hnu_deploy import RobotState, TrackerPolicy, yaw_of  # noqa: F401


def run(model_dir, motion_npz, read_state, send_targets, started, dt=0.02, heading_err=None, first_delta_deg=0.0):
    """read_state() -> RobotState; send_targets(q_target, kp, kd); started() -> bool (operator start button).
    heading_err: None (no compensation: the robot must stand exactly facing +x), a number (deg, measured with the
    straightedge) or a HeadingMeter (IMU method, recommended): the first takeover without a stored floor heading
    stores it, every later takeover is compensated to it.
    first_delta_deg (final): the props are placed by tape coordinates, so the stored 0 deg must be the
    floor x line: at the first takeover align the robot carefully and pass its measured heading error
    (delta_from_straightedge, deg, + = robot turned LEFT); an error here acts like an uncompensated start error."""
    pol = TrackerPolicy(model_dir)
    m = pol.meta
    kp, kd = np.array(m["kp"]), np.array(m["kd"])
    ref = np.load(motion_npz)
    T = len(ref["joint_pos"])
    s = read_state()
    # heading origin: the robot's IMU yaw at the takeover is mapped to the reference start heading.  Without
    # compensation a placement heading error rotates the whole path; with the measured error added the policy sees
    # the error and turns back by itself (sim 3-10 deg errors, measurement +-1 deg)
    y0 = yaw_of(ref["body_quat_w"][0, 0])
    if heading_err is None:
        start_yaw = y0
    elif hasattr(heading_err, "start_yaw"):
        if heading_err.floor is None:  # first takeover of the session: this heading becomes 0 (no reboot afterwards)
            heading_err.set_floor(s.quat, first_delta_deg)
        start_yaw = heading_err.start_yaw(y0, s.quat)
    else:
        if abs(float(heading_err)) > MAX_COMP_DEG:
            raise ValueError(f"heading error {heading_err:+.1f} deg > {MAX_COMP_DEG} deg: re-place the robot")
        start_yaw = y0 + np.radians(float(heading_err))
    pol.reset(motion_npz, s, start_yaw=start_yaw)
    frame, holding = 0, True
    while True:
        s = read_state()
        if holding and frame == 0 and started():
            holding = False
        q_t = pol.step(s, frame, holding)
        send_targets(q_t, kp, kd)
        if not holding:
            frame += 1
            if frame >= T - 1:  # end of the motion: keep the last frame, then the operator switches controllers
                frame, holding = T - 1, True
        yield frame, holding  # the caller sleeps to the next 20 ms tick
