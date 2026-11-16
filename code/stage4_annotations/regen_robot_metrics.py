"""Stage 4: regenerate the robot side of the annotations from the stage-1 references (+ expert rollouts).

For every trial of a task this writes, in the original schemas,
  <out>/Retargeted/Kinematics/<task>/<trial>_Kinematics.json
  <out>/Retargeted/Dynamics/<task>/<trial>_Dynamics.json
  <out>/LLM_annotation/<task>/<trial>.json        (old file; only Retargeted_Robot_Metrics replaced)

What changed w.r.t. the old robot metrics
  * joint moments come from contact-aware inverse dynamics (ground reaction on the stance soles), so the
    support-leg torques are physical; `floating_base_wrench` is now the *residual* base wrench that the
    contacts could not explain (about 0 when the motion is dynamically consistent)
  * new `ground_reaction` block (total GRF / body weight)
  * new `execution` block from simulated expert rollouts (success, executed torque, measured GRF,
    tracking error) when --rollouts is given
  * `robot_semantics` is regenerated with deterministic, evidence-backed rules (provenance "rules_v1");
    human fields are untouched.
"""
from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import sys

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "stage1_reference"))
from build_reference import FOOT_BODIES, fk, load_robot, sole_points  # noqa: E402

G = 9.81
DEG = 180.0 / np.pi
KIN_JOINTS = ("waist_pitch_joint", "left_knee_joint", "left_ankle_pitch_joint", "left_hip_pitch_joint",
              "right_hip_pitch_joint", "right_knee_joint", "right_ankle_pitch_joint")
LEG = {"L": ("left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
             "left_ankle_pitch_joint", "left_ankle_roll_joint"),
       "R": ("right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
             "right_ankle_pitch_joint", "right_ankle_roll_joint")}


def stats(x, t):
    if len(x) == 0:
        return None
    i, j = int(np.argmin(x)), int(np.argmax(x))
    return dict(min=round(float(x[i]), 6), min_time_s=round(float(t[i]), 6), max=round(float(x[j]), 6),
                max_time_s=round(float(t[j]), 6), start=round(float(x[0]), 6), end=round(float(x[-1]), 6),
                mean=round(float(np.mean(x)), 6))


def rom(s):
    return None if s is None else dict(ROM=round(s["max"] - s["min"], 6), min_time_s=s["min_time_s"], max_time_s=s["max_time_s"])


def quat_pitch(q):
    w, x, y, z = q
    return np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))


class Trial:
    def __init__(self, r, ref_dir, name):
        d = np.load(os.path.join(ref_dir, f"{name}_Dynamics.npz"))
        self.q, self.v, self.tau = d["qpos"], d["qvel"], d["actuator_tau"]
        self.contact, self.ff, self.res = d["foot_contact"], d["foot_force_world"], d["base_residual_wrench"]
        self.excess, self.low = d["joint_limit_excess"], d["sole_min_height_m"]
        self.fps = float(d["fps"])
        self.crop0 = int(d["source_crop"][0])
        self.limits = d["actuator_torque_limits"]
        self.names = list(d["actuator_joint_names"])
        self.T = len(self.q)
        self.mass = float(d["robot_mass_kg"])
        self.jidx = {n: i for i, n in enumerate(self.names)}
        # per-frame geometry
        self.torso_z = np.zeros(self.T)
        self.foot_xy = np.zeros((self.T, 2, 2))
        self.selfpen = np.zeros(self.T)
        tid = mujoco.mj_name2id(r.m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        for t in range(self.T):
            fk(r, self.q[t])
            self.torso_z[t] = r.d.xpos[tid][2]
            for k, fid in enumerate(r.foot_ids):
                self.foot_xy[t, k] = r.d.xpos[fid][:2]
        for t in range(0, self.T, 2):
            r.d.qpos[:] = self.q[t]
            mujoco.mj_forward(r.m, r.d)
            pen = [-c.dist for c in r.d.contact[: r.d.ncon] if c.dist < -0.005]
            self.selfpen[t] = max(pen) if pen else 0.0
        self.standing_torso_z = float(np.percentile(self.torso_z, 90))

    def idx(self, t_src):
        return int(round(t_src * self.fps)) - self.crop0

    def time(self, i):
        return (np.asarray(i) + self.crop0) / self.fps


def support_side(tr, a, b, human_seg):
    c = tr.contact[a:b]
    fl, fr = c[:, 0].mean(), c[:, 1].mean()
    if fl < 0.2 and fr < 0.2:
        return []
    if fl >= 0.2 and fr >= 0.2:
        return ["L", "R"] if abs(fl - fr) < 0.3 else (["L"] if fl > fr else ["R"])
    return ["L"] if fl > fr else ["R"]


def robot_metrics(tr, a, b, sides, rollout_seg):
    t = tr.time(np.arange(a, b))
    q = tr.q[a:b]
    kin = {}
    for n in KIN_JOINTS:
        kin[n] = stats(q[:, 7 + tr.jidx[n]] * DEG, t)
    kin["pelvis_pitch_deg"] = stats(np.array([quat_pitch(x[3:7]) for x in q]) * DEG, t)
    kin["torso_height_m"] = stats(tr.torso_z[a:b], t)
    dyn = {"joint_moment": {}}
    if len(sides) == 1:
        s = sides[0]
        w = "R" if s == "L" else "L"
        for role, side in (("support", s), ("swing", w)):
            for j, jn in (("hip", "hip_pitch"), ("knee", "knee"), ("ankle", "ankle_pitch")):
                n = f"{'left' if side == 'L' else 'right'}_{jn}_joint"
                dyn["joint_moment"][f"{role}_{j}_moment_peak_abs_Nm_per_kg"] = stats(np.abs(tr.tau[a:b, tr.jidx[n]]) / tr.mass, t)
    res = tr.res[a:b]
    dyn["floating_base_wrench"] = dict(
        definition="residual base wrench not explained by sole contact forces (0 = dynamically consistent)",
        force_norm_N=stats(np.linalg.norm(res[:, :3], axis=1), t),
        moment_norm_Nm=stats(np.linalg.norm(res[:, 3:], axis=1), t))
    grf = np.linalg.norm(tr.ff[a:b], axis=-1)
    dyn["ground_reaction"] = dict(total_over_weight=stats(grf.sum(1) / (tr.mass * G), t),
                                  left_over_weight=stats(grf[:, 0] / (tr.mass * G), t),
                                  right_over_weight=stats(grf[:, 1] / (tr.mass * G), t))
    ratio = np.abs(tr.tau[a:b]) / tr.limits

    def screen(cols, label_prefix):
        rr = ratio[:, cols]
        k = np.unravel_index(np.argmax(rr), rr.shape)
        peak = float(rr.max())
        jn = tr.names[cols[k[1]]]
        return dict(label=f"{label_prefix}_{'exceeds' if peak > 1 else 'within'}_declared_torque_limits",
                    peak_limit_ratio=round(peak, 6), violation_frame_ratio=round(float((rr > 1).any(1).mean()), 6),
                    any_limit_violation=bool(peak > 1), peak_joint_name=jn, peak_frame=int(a + k[0] + tr.crop0),
                    peak_time_s=round(float(t[k[0]]), 6), peak_required_Nm=round(float(abs(tr.tau[a + k[0], cols[k[1]]])), 6),
                    peak_limit_Nm=float(tr.limits[cols[k[1]]]))

    if sides:
        cols = [tr.jidx[n] for s in sides for n in LEG[s]]
        support = screen(cols, "support_chain")
        support["support_sides"] = sides
        support["torque_source"] = "contact_aware_inverse_dynamics"
    else:
        support = dict(label="support_chain_not_evaluated_no_support_side", support_sides=[])
    whole = screen(list(range(len(tr.names))), "whole_body")
    ex = tr.excess[a:b]
    viol = (ex > 1e-3).any(1)
    jl = dict(status="joint_limit_violation_detected" if viol.any() else "within_limits",
              violation_frame_count=int(viol.sum()), violation_frame_ratio=round(float(viol.mean()), 6),
              max_excess=round(float(ex.max()), 6),
              violating_joint_names=[tr.names[j] for j in np.where((ex > 1e-3).any(0))[0]])
    pen_floor = float(max(-tr.low[a:b].min(), 0.0))
    selfpen = float(tr.selfpen[a:b].max())
    col = dict(status="clear" if (pen_floor < 0.02 and selfpen < 0.01) else
               ("self_collision_detected" if selfpen >= 0.01 else "floor_penetration_detected"),
               floor_contact_frame_count=int(tr.contact[a:b].any(1).sum()),
               max_floor_penetration_mm=round(pen_floor * 1000, 3),
               self_collision_frame_count=int((tr.selfpen[a:b] > 0.005).sum()),
               max_self_penetration_mm=round(selfpen * 1000, 3))
    out = dict(kinematics=kin, dynamics=dyn,
               feasibility=dict(torque_screening=dict(support_chain=support, whole_body=whole),
                                state_validity=dict(joint_limit=jl, collision=col)))
    if rollout_seg is not None:
        out["execution"] = rollout_seg
    return out


def rollout_segment_stats(rolls, tr, a, b):
    """Execution statistics of the expert over [a, b) (reference frames @ tr.fps)."""
    if not rolls:
        return None
    s0, s1 = a / tr.fps, b / tr.fps  # seconds since clip start
    reached, passed, tq, grf, err = 0, 0, [], [], []
    for R in rolls:
        fr = R["frame"] / 50.0
        for e in range(R["frame"].shape[1]):
            v = R["valid"][:, e]
            tt = fr[:, e]
            inseg = v & (tt >= s0) & (tt < s1)
            last_t = tt[v].max() if v.any() else -1
            if last_t >= s0:
                reached += 1
                ok = bool(R["success"][e]) or last_t >= s1
                passed += int(ok)
            if inseg.any():
                tq.append(np.abs(R["joint_torque"][inseg, e]) / tr.limits)
                grf.append(np.linalg.norm(R["feet_force"][inseg, e], axis=-1).sum(-1) / (tr.mass * G))
                err.append(R["body_pos_err"][inseg, e])
    if reached == 0:
        return dict(status="not_reached")
    tq = np.concatenate(tq) if tq else np.zeros((0, len(tr.limits)))
    grf = np.concatenate(grf) if grf else np.zeros(0)
    err = np.concatenate(err) if err else np.zeros(0)
    return dict(status="evaluated", source="simulated_expert_rollouts", rollouts_reaching=reached,
                success_rate=round(passed / reached, 4),
                executed_torque_ratio_p95=round(float(np.percentile(tq.max(1), 95)), 4) if len(tq) else None,
                executed_torque_ratio_max=round(float(tq.max()), 4) if len(tq) else None,
                measured_grf_over_weight_max=round(float(grf.max()), 4) if len(grf) else None,
                tracking_body_pos_err_mean_m=round(float(err.mean()), 4) if len(err) else None)


# ------------------------------------------------------------------ deterministic robot semantics

def rule_semantics(seg, rm, sides, tr, a, b, task):
    from robot_semantic_schema import CANONICAL_ATOMS, phase_known_absent_atoms
    from robot_semantic_generation_utils import derive_control_objectives, derive_feasibility_tags
    pos, neg, evidence = [], [], []

    def decide(atom, cond, source, value):
        if cond is None:
            return
        (pos if cond else neg).append(atom)
        if cond:
            evidence.append(dict(atom=atom, source=source, value=json.dumps(value)))

    km = rm["kinematics"]
    jm = rm["dynamics"]["joint_moment"]
    grf = rm["dynamics"]["ground_reaction"]["total_over_weight"]
    c = tr.contact[a:b]
    single = float((c.sum(1) == 1).mean())
    double = float((c.sum(1) == 2).mean())
    phase = str(seg.get("gait_phase", "")).lower()
    if len(sides) == 1:
        s = "left" if sides[0] == "L" else "right"
        w = "right" if s == "left" else "left"
        sk, wk, wh = km.get(f"{s}_knee_joint"), km.get(f"{w}_knee_joint"), km.get(f"{w}_hip_pitch_joint")
        decide("support_knee_flexion", sk["max"] - sk["min"] >= 20.0,
               f"Retargeted_Robot_Metrics.kinematics.{s}_knee_joint.ROM", round(sk["max"] - sk["min"], 3))
        decide("swing_leg_flexion", wk["max"] - wk["min"] >= 30.0,
               f"Retargeted_Robot_Metrics.kinematics.{w}_knee_joint.ROM", round(wk["max"] - wk["min"], 3))
        decide("swing_leg_advancement", wh["max"] - wh["min"] >= 25.0,
               f"Retargeted_Robot_Metrics.kinematics.{w}_hip_pitch_joint.ROM", round(wh["max"] - wh["min"], 3))
        decide("support_loading", grf["max"] >= 0.8,
               "Retargeted_Robot_Metrics.dynamics.ground_reaction.total_over_weight.max", grf["max"])
        early_peak = grf["max_time_s"] - tr.time(a) < 0.45 * max(tr.time(b) - tr.time(a), 1e-6)
        decide("load_acceptance", bool(early_peak and grf["max"] >= 1.0 and "take_off" not in phase),
               "Retargeted_Robot_Metrics.dynamics.ground_reaction.total_over_weight.max_time_s", grf["max_time_s"])
        decide("push_off_preparation", ("take_off" in phase) or (not early_peak and grf["max"] >= 1.0),
               "gait_phase", phase)
    decide("single_support_balance", single >= 0.7, "robot_contact.single_support_fraction", round(single, 3))
    decide("double_support_stabilization", double >= 0.5, "robot_contact.double_support_fraction", round(double, 3))
    if c.any(1).any():
        lf = c[:, 0].astype(float) / np.maximum(c.sum(1), 1)
        shift = float(np.abs(lf[c.any(1)][-1] - lf[c.any(1)][0])) if c.any(1).sum() > 1 else 0.0
        decide("weight_transfer", double > 0.1 and shift >= 0.5, "robot_contact.load_side_shift", round(shift, 3))
    if task == "CrouchedTraverse":
        drop = tr.standing_torso_z - km["torso_height_m"]["min"]
        low = drop >= 0.10
        decide("overhead_clearance", low, "Retargeted_Robot_Metrics.kinematics.torso_height_m.min", km["torso_height_m"]["min"])
        decide("torso_clearance", low, "Retargeted_Robot_Metrics.kinematics.torso_height_m.min", km["torso_height_m"]["min"])
        col_ok = rm["feasibility"]["state_validity"]["collision"]["status"] == "clear"
        decide("collision_avoidance", bool(low and col_ok), "Retargeted_Robot_Metrics.feasibility.state_validity.collision.status",
               rm["feasibility"]["state_validity"]["collision"]["status"])
    absent_phase = phase_known_absent_atoms(seg)
    pos = [x for x in pos if x not in absent_phase]
    known_absent = sorted(set(neg) | absent_phase)
    unknown = [x for x in CANONICAL_ATOMS if x not in pos and x not in known_absent]
    pos = [x for x in CANONICAL_ATOMS if x in pos]
    seg_view = {"Retargeted_Robot_Metrics": rm}
    return dict(skill_atoms=pos, unknown_atoms=unknown, known_absent_atoms=known_absent,
                robot_control_objectives=derive_control_objectives(pos),
                feasibility_tags=derive_feasibility_tags(seg_view), evidence=evidence,
                provenance="rules_v1 (deterministic, stage-4 regen )")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--ref", required=True, help="stage-1 output dir for the task")
    ap.add_argument("--old-annotations", required=True, help="old LLM_annotation/<task> dir")
    ap.add_argument("--rollouts", default=None, help="dir with rollout_<trial>*.npz from eval_expert.py")
    ap.add_argument("--data-code", required=True, help="dir containing robot_semantic_schema.py etc.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sys.path.insert(0, args.data_code)
    r = load_robot()
    for sub in ("Retargeted/Kinematics", "Retargeted/Dynamics", "LLM_annotation"):
        os.makedirs(os.path.join(args.out, sub, args.task), exist_ok=True)
    report = []
    for f in sorted(glob.glob(os.path.join(args.old_annotations, f"{args.task}_*.json"))):
        name = os.path.basename(f)[:-5]
        if not os.path.exists(os.path.join(args.ref, f"{name}_Dynamics.npz")):
            report.append(dict(trial=name, status="no_reference"))
            continue
        doc = json.load(open(f, encoding="utf-8-sig"))
        tr = Trial(r, args.ref, name)
        rolls = []
        if args.rollouts:
            for p in sorted(glob.glob(os.path.join(args.rollouts, f"rollout_{name}*.npz"))):
                rolls.append(dict(np.load(p)))
        kin_json = dict(robot_model=dict(name="G1", nq=36, nv=35, nu=29, mass_kg=tr.mass, source="mjlab g1.xml"), segments={})
        dyn_json = dict(robot_model=kin_json["robot_model"], segments={})
        new = copy.deepcopy(doc)
        n_ok = 0
        for seg in new["segments"]:
            sid = seg["segment_id"]
            ts, te = seg["time"]["start_s"], seg["time"]["end_s"]
            a, b = tr.idx(ts), tr.idx(te)
            a_c, b_c = max(a, 0), min(b, tr.T)
            align = dict(sample_rate_hz=tr.fps, start_index=a_c, end_index=b_c, source_start_s=ts, source_end_s=te,
                         alignment_method="timestamp_minus_crop", crop_start_frame=tr.crop0)
            if b_c - a_c < 3:
                status = dict(status="unavailable_outside_task_crop", start_index=a_c, end_index=b_c)
                rm = dict(npz_alignment=align, kinematics={}, obstacle_relation={}, dynamics={}, feasibility={},
                          data_availability=status)
                seg["Retargeted_Robot_Metrics"] = rm
                continue
            sides = support_side(tr, a_c, b_c, seg)
            roll = rollout_segment_stats(rolls, tr, a_c, b_c)
            m = robot_metrics(tr, a_c, b_c, sides, roll)
            avail = dict(status="complete" if (a_c == a and b_c == b) else "partial", start_index=a_c, end_index=b_c)
            kin_json["segments"][sid] = dict(npz_alignment=align, kinematics=m["kinematics"], obstacle_relation={})
            dyn_json["segments"][sid] = dict(dynamics=m["dynamics"], feasibility=m["feasibility"], data_availability=avail,
                                             **({"execution": m["execution"]} if "execution" in m else {}))
            rm = dict(npz_alignment=align,
                      kinematics={k: rom(v) for k, v in m["kinematics"].items() if v is not None},
                      obstacle_relation={}, dynamics=m["dynamics"], feasibility=m["feasibility"], data_availability=avail)
            if "execution" in m:
                rm["execution"] = m["execution"]
            rm["robot_semantics"] = rule_semantics(seg, dict(m, kinematics=m["kinematics"]), sides, tr, a_c, b_c, args.task)
            seg["Retargeted_Robot_Metrics"] = rm
            n_ok += 1
        new["Robot_model"] = dict(kin_json["robot_model"])
        new["robot_annotation_version"] = "stage4_regen_v1"
        for sub, obj, suffix in (("Retargeted/Kinematics", kin_json, "_Kinematics.json"), ("Retargeted/Dynamics", dyn_json, "_Dynamics.json")):
            with open(os.path.join(args.out, sub, args.task, name + suffix), "w", encoding="utf-8") as fo:
                json.dump(obj, fo, indent=1, ensure_ascii=False)
        with open(os.path.join(args.out, "LLM_annotation", args.task, name + ".json"), "w", encoding="utf-8") as fo:
            json.dump(new, fo, indent=1, ensure_ascii=False)
        report.append(dict(trial=name, status="ok", segments=len(new["segments"]), segments_with_data=n_ok, rollouts=len(rolls)))
        print(json.dumps(report[-1]), flush=True)
    with open(os.path.join(args.out, f"regen_report_{args.task}.json"), "w") as fo:
        json.dump(report, fo, indent=1)


if __name__ == "__main__":
    main()
