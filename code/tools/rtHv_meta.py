"""policy_meta.json of the stakes student WITH the mocap body velocity (tools/rtH_vel.sh)."""
import json
import sys

f = sys.argv[1]
m = json.load(open(f))
cut = lambda L: dict(bar_len=round(2 * L, 3), brace_d=round(max(L - 0.05, 0.06), 3), brace_h=round(max(L - 0.05, 0.06), 3))  # noqa: E731
m["bridge"]["post_base"].update(mid=cut(0.15), end=cut(0.20))
m["obstacle_input"] = "mocap"
m["lin_vel_input"] = "mocap"
m["lin_vel_note"] = ("base linear velocity (body frame, at the IMU site) estimated from the motion-capture pelvis "
                     "positions as in Deployment_F (least-squares slope over the last 60 ms, outlier rejection; trained "
                     "with 10-40 ms delay and noise) -- config.yaml mocap.lin_vel_window_s / lin_vel_outlier_m")
m["waist_lock"] = ["waist_roll_joint", "waist_pitch_joint"]
m["start"] = dict(hold_s="rope release time + 1-2 s",
                  note="obstacle features AND base velocity from the motion capture, the rest onboard sensors: "
                       "hang on the rope in the frame-0 pose at the planned start (pelvis +-5 cm, heading +-3 deg), set the "
                       "feet down, switch the policy on with frame 0, release the rope over ~10 s (trained 1-12 s, up to 90 % "
                       "body weight at the switch), keep frame 0 until 1-2 s after the rope is slack, then advance the frame "
                       "every 20 ms.  Waist roll / pitch get the constant target 0 (action_scale 0).")
json.dump(m, open(f, "w"), indent=1)
