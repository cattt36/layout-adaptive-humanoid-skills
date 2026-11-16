"""policy_meta.json of the LiDAR student export (tools/rtH_lidar.sh)."""
import json
import sys

f = sys.argv[1]
m = json.load(open(f))
cut = lambda L: dict(bar_len=round(2 * L, 3), brace_d=round(max(L - 0.05, 0.06), 3), brace_h=round(max(L - 0.05, 0.06), 3))  # noqa: E731
m["bridge"]["post_base"].update(mid=cut(0.15), end=cut(0.20))
m["obstacle_input"] = "lidar"
m["lidar_pose"] = ("pelvis pose (position, heading) from LiDAR odometry localised in a prior map of the venue; "
                   "trained for 10 Hz scan updates with 40-100 ms delay or an IMU-propagated pose (every 20 ms, 20-60 ms "
                   "delay), 1 cm / 0.5 deg noise per update, 4 cm / 5 deg constant registration bias.  Give it as "
                   "RobotState.pos / quat (or perc_pos / perc_quat); the attitude and gyro for the policy stay the IMU's")
m["lin_vel_input"] = "zero"
m["waist_lock"] = ["waist_roll_joint", "waist_pitch_joint"]
m["start"] = dict(hold_s="rope release time + 1-2 s",
                  note="obstacle features from the LiDAR pose (robot pelvis relative to the planned obstacles; "
                       "no base linear velocity, input 0), the rest onboard sensors: hang on the rope in the frame-0 pose at "
                       "the planned start (pelvis +-5 cm, heading +-3 deg), set the feet down, switch the policy on with "
                       "frame 0, release the rope over ~10 s (trained 1-12 s, up to 90 % body weight at the switch), keep "
                       "frame 0 until 1-2 s after the rope is slack, then advance the frame every 20 ms.  Waist roll / pitch "
                       "get the constant target 0 (action_scale 0).")
json.dump(m, open(f, "w"), indent=1)
