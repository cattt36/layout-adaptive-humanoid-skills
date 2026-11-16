import json, sys
old = json.load(open(sys.argv[1], encoding="utf-8-sig"))
new = json.load(open(sys.argv[2], encoding="utf-8"))
sid = sys.argv[3]
so = next(s for s in old["segments"] if s["segment_id"] == sid)
sn = next(s for s in new["segments"] if s["segment_id"] == sid)
ro, rn = so["Retargeted_Robot_Metrics"], sn["Retargeted_Robot_Metrics"]
print("gait_phase", so.get("gait_phase"), "| human support ankle Nm/kg",
      so["Human_Metrics"]["dynamics"]["joint_moment"].get("support_ankle_moment_peak_abs_Nm_per_kg", {}).get("max"))
for k in ("support_ankle", "support_knee", "support_hip", "swing_hip"):
    key = f"{k}_moment_peak_abs_Nm_per_kg"
    print(f"{k:14s} old {ro['dynamics']['joint_moment'].get(key, {}).get('max')}  new {rn['dynamics']['joint_moment'].get(key, {}).get('max')}")
print("base wrench max N  old", ro["dynamics"]["floating_base_wrench"]["force_norm_N"]["max"],
      " new(residual)", rn["dynamics"]["floating_base_wrench"]["force_norm_N"]["max"])
print("GRF/W max new", rn["dynamics"]["ground_reaction"]["total_over_weight"]["max"])
for side in ("support_chain", "whole_body"):
    o, n = ro["feasibility"]["torque_screening"][side], rn["feasibility"]["torque_screening"][side]
    print(side, "| old", o["label"], o.get("peak_limit_ratio"), "| new", n["label"], n.get("peak_limit_ratio"), n.get("peak_joint_name"))
print("old atoms", ro["robot_semantics"]["skill_atoms"], ro["robot_semantics"]["feasibility_tags"])
print("new atoms", rn["robot_semantics"]["skill_atoms"], rn["robot_semantics"]["feasibility_tags"])
