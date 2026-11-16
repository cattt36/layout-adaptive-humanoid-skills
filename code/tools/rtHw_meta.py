"""policy_meta.json of the waist-offset students (tools/rtH_waist.sh), after rtHv_meta.py.
  python rtHw_meta.py <policy_meta.json> zero|mocap   (body velocity input of the model)"""
import json
import sys

f, lv = sys.argv[1], sys.argv[2]
m = json.load(open(f))
m["lin_vel_input"] = lv
if lv == "zero":
    m.pop("lin_vel_note", None)
m["waist_offset"] = ("trained with the waist roll / pitch locked around a random per-episode offset (roll "
                     "-0.15..+0.10, pitch -0.05..+0.25 rad, 30 % at 0) for the robot with the faulty waist; the waist "
                     "target stays 0 (action_scale 0), the policy reads the actual waist angle from the encoders")
m["start"]["note"] = m["start"]["note"].replace("obstacle features AND base velocity from the motion capture",
                                                "obstacle features from the motion capture" + ("" if lv == "zero" else
                                                                                              " and the base velocity too"))
json.dump(m, open(f, "w"), indent=1)
