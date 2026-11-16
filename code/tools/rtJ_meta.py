"""DEPLOYMENT: policy_meta.json of the standard-start students (tools/rtJ_std.sh, rtJw.sh).
  python rtJ_meta.py <policy_meta.json> zero|mocap [waist]   (body velocity input; waist = faulty-waist variant)"""
import json
import sys

f, lv = sys.argv[1], sys.argv[2]
waist = len(sys.argv) > 3 and sys.argv[3] == "waist"
m = json.load(open(f))
cut = lambda L: dict(bar_len=round(2 * L, 3), brace_d=round(max(L - 0.05, 0.06), 3), brace_h=round(max(L - 0.05, 0.06), 3))  # noqa: E731
m["bridge"]["post_base"].update(mid=cut(0.15), end=cut(0.20))
m["obstacle_input"] = "mocap"
m["obstacle_positions"] = "measured live by the motion capture at the takeover (required; code/robot_side/refresh_obstacles_J.py)"
m["lin_vel_input"] = lv
if lv == "mocap":
    m["lin_vel_note"] = ("base linear velocity (body frame, IMU site) estimated from the motion-capture pelvis positions "
                         "(least-squares slope over 60 ms, outlier rejection; trained with 10-40 ms delay and noise)")
else:
    m.pop("lin_vel_note", None)
if waist:
    m["waist_lock"] = ["waist_roll_joint", "waist_pitch_joint"]
    m["waist_offset"] = ("trained with the waist roll / pitch locked around a random per-episode offset (roll -0.20..+0.45, "
                         "pitch -0.05..+0.50 rad, 25 % at 0) for the G1 with the faulty waist; target stays 0 (action_scale 0)")
else:
    m.pop("waist_lock", None)  # new G1, waist free
    m.pop("waist_offset", None)
m["standard_start"] = ("every motion starts with frame 0 = a NORMAL stance (feet parallel, 0.24 m apart) at the "
                       "layout origin facing +x; the reference then makes one step onto the skill's footprints, slows in "
                       "and continues with the skill (tools/make_start.py)")
m["start"] = dict(hold_s="rope release time + 1-2 s",
                  note="standard start: hang on the rope in the normal stance (frame 0 of the motion) at the "
                       "layout origin facing +x (+-5 cm, +-10 deg trained), set the feet down, switch the policy on with "
                       "frame 0, release the rope over ~10 s (trained 1-12 s, up to 90 % body weight), keep frame 0 until "
                       "1-2 s after the rope is slack, then advance the frame every 20 ms: the robot steps into the skill "
                       "by itself (about 1.8 s step-in + 2 s slow-in)")
json.dump(m, open(f, "w"), indent=1)
