"""Representative (unselected) videos for judging the success definition: for every test trial, the FIRST rollout
(fixed reset seed) of each student, side by side, half speed.  Task criterion (fall = pelvis < 0.30 m or tilt > 85 deg;
success = no fall and pelvis within 1 m of the reference end).  Labels:
  stakes  (foot-stake and foot-floor contact sensors; see stake_tag)
          ON     a foot stood fully on the stake (touching the stake, not the floor)
          PART   a foot touched the stake and the floor at the same time (heel on the stake, toes on the floor, ...)
          GROUND no foot touched the stake;   LATE- prefix: only outside the reference stance +-0.2 s (rhythm)
  bridge  time of the first bridge contact
Writes runs/videos/batch/<clip>.mp4 and summary.json.

  python record_batch.py --split test --student L0=<pt>:<z_dir> --student L4=<pt>:<z_dir> [--clips a,b] [--slow 2]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import hnu_tasks  # noqa: F401
from hnu_tasks.distill import hnu_distill_env_cfg
from hnu_tasks.multi_motion import resolve_motion_files
from hnu_tasks.stakes import N_STAKES, STAKE_H, STAKE_TOL
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from distill_train import student_from_ckpt

GREEN, RED, YEL, WHITE = (80, 255, 80), (255, 90, 90), (255, 220, 60), (255, 255, 255)
# task rename (display only, data keep the old names until the server-side rename):
# ThreeStepPoleRun -> ThreeStepPoleRun, FiveStepPoleRun -> FiveStepPoleRun, CrouchedTraverse -> CrouchedTraverse.
# HNU_DISPLAY_RENAME=1 shows the new names.
DISP = os.environ.get("HNU_DISPLAY_RENAME") == "1"


def disp(s):
    if not DISP:
        return s
    return s.replace("ThreeStepPoleRun", "ThreeStepPoleRun").replace("FiveStepPoleRun", "FiveStepPoleRun").replace(
        "CrouchedTraverse", "CrouchedTraverse")


def fall(env):
    d = env.scene["robot"].data
    h = d.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return (h < 0.30) | (d.projected_gravity_b[:, 2] > -float(np.cos(np.radians(85.0))))


def make_env(pt, zdir, size):
    """pt = student checkpoint, or 'TEACHER@<bank file>' (file with --ckpt-<task> <pt> pairs) for the teacher bank."""
    teacher = pt.startswith("TEACHER@")
    if teacher:
        toks = open(pt.split("@", 1)[1]).read().split()
        bank = {toks[i][len("--ckpt-"):]: toks[i + 1] for i in range(0, len(toks) - 1, 2) if toks[i].startswith("--ckpt-")}
        ck = dict(z_offsets=(0, 5, 10, 15, 20, 25, 30, 40, 50), scene="geom")
    else:
        ck = torch.load(pt, map_location="cuda:0", weights_only=False)
    cfg = hnu_distill_env_cfg(zdir, play=True, z_offsets=tuple(ck["z_offsets"]), scene=ck.get("scene", "geom"),
                              hist=ck.get("hist", 5), lin_vel=ck.get("lin_vel", False))
    cmd = cfg.commands["motion"]
    names = [os.path.basename(f).replace("_motion.npz", "") for f in resolve_motion_files(cmd.motion_file)]
    cmd.fixed_clip = 0
    cmd.sampling_mode = "start"
    cfg.terminations = {k: v for k, v in cfg.terminations.items() if k in ("time_out", "bad_state")}
    cfg.terminations["fall"] = TerminationTermCfg(func=fall)
    if os.environ.get("HNU_REC_PUSH"):  # same pushes as eval_student.py --push
        from mjlab.tasks.registry import load_env_cfg
        cfg.events["push_robot"] = load_env_cfg("HNU-Track-FiveStepPoleRun").events["push_robot"]
    cfg.scene.num_envs = 1
    # feet-vs-stake contact sensors (the robot's feet_ground_contact only sees the terrain, not the stakes)
    if os.environ.get("HNU_REC_NO_STAKE_SENS"):
        pass
    else:
        cfg.scene.sensors = tuple(cfg.scene.sensors) + tuple(ContactSensorCfg(
        name=f"foot_stake{i}",
        primary=ContactMatch(mode="subtree", pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$", entity="robot"),
        secondary=ContactMatch(mode="body", pattern=r"^stake$", entity=f"stake{i}"),
        fields=("found", "force"), reduce="netforce", num_slots=1) for i in range(N_STAKES))
    cfg.viewer.height, cfg.viewer.width = size
    cfg.viewer.distance, cfg.viewer.elevation, cfg.viewer.azimuth = 3.6, -30.0, 90.0
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode="rgb_array")
    if teacher:
        from hnu_tasks.distill import MoETeacher, scene_teacher_groups
        pol = MoETeacher(bank, "cuda:0", scene_teacher_groups("geom"))
        return env, (lambda o: pol(o, o["task"])), names
    st = student_from_ckpt(ck, env.action_manager.total_action_dim, "cuda:0")
    grp = ck.get("obs_group", "student")
    return env, (lambda o: st(o[grp])), names


def run_clip(env, act, names, clip, label, seed):
    c = env.command_manager.get_term("motion")
    c.cfg.fixed_clip = names.index(clip)
    m = c.motion
    cid = names.index(clip)
    T = int(m.clip_len[cid])
    sw = c.stake_windows
    fidx = [c.cfg.body_names.index(n) for n in ("left_ankle_roll_link", "right_ankle_roll_link")]
    font = ImageFont.load_default(size=20)
    n_st = int((sw[0][cid] >= 0).sum()) if sw is not None else 0
    win, on, near = np.ones(N_STAKES), np.zeros(N_STAKES), np.zeros(N_STAKES, bool)
    best_d = np.full(N_STAKES, np.inf)
    late_lv = np.zeros(N_STAKES)
    frames, fell, touch = [], None, None
    torch.manual_seed(seed)
    last_pel = last_ref = None
    with torch.inference_mode():
        obs, _ = env.reset()
        for t in range(T - 1):
            obs, _, term, trunc, _ = env.step(act(obs))
            if term.any():
                fell = t / 50
                break
            pel = c.robot_body_pos_w[0, 0, :2].clone()
            # clip end: the command resamples and teleports the robot to the clip start inside env.step -> stop
            # before recording that frame; the verdict uses the last state before the teleport (as eval_student)
            if trunc.any() or (last_pel is not None and float(torch.norm(pel - last_pel)) > 0.5):
                break
            last_pel, last_ref = pel, c.body_pos_w[0, 0, :2].clone()
            fr = int(c.time_steps[0] - m.clip_start[c.clip_id[0]])
            if touch is None and bool(hnu_tasks.bridge_touch(env)[0]):
                touch = t / 50
            if n_st:
                foot, f0, f1, sxy = sw
                fp = (c.robot_body_pos_w[0, fidx] - env.scene.env_origins[0]).cpu().numpy()
                for i in range(n_st):
                    a, b = int(f0[cid, i]), int(f1[cid, i])
                    cxy = sxy[cid, i].cpu().numpy()
                    # STANDING on stake i: physical contact foot <-> stake i (edge steps included) with the ankle
                    # raised above ground level (robot: on a stake >= 0.09 m, on the ground 0.035 m) -- excludes
                    # kicking the side of the stake from the ground
                    if f"foot_stake{i}" not in env.scene.sensors:
                        continue
                    fnd = env.scene.sensors[f"foot_stake{i}"].data.found.reshape(-1)[:2].cpu().numpy() > 0
                    gnd = env.scene.sensors["feet_ground_contact"].data.found.reshape(-1)[:2].cpu().numpy() > 0
                    dist = [np.linalg.norm(p[:2] - cxy) for p in fp]
                    tops = [bool(fnd[k]) and fp[k, 2] >= 0.07 for k in range(2)]
                    full = [tops[k] and not bool(gnd[k]) for k in range(2)]  # on the stake only (not also the floor)
                    if any(tops):
                        best_d[i] = min(best_d[i], min(d for d, tp in zip(dist, tops) if tp))
                        level = 2 if any(full) else 1  # 2 = fully on the stake, 1 = straddling stake + floor
                        if a - 10 <= fr <= b + 10:
                            on[i] = max(on[i], level)  # on time (reference stance window +-0.2 s)
                        else:
                            late_lv[i] = max(late_lv[i], level)  # stepped on it, but early / late
            img = Image.fromarray(env.render())
            dr = ImageDraw.Draw(img)
            dr.text((10, 8), f"{label}   {disp(clip)}   t = {t / 50:.2f} s   (half speed)", fill=WHITE, font=font)
            x = 10
            for i in range(n_st):
                if fr > int(sw[2][cid, i]) + 25:
                    tag, col = stake_tag(on[i], late_lv[i])
                    dr.text((x, 34), f"S{i + 1} {tag}", fill=col, font=font)
                    x += 125
            if touch is not None:
                dr.text((10, 34), f"bridge contact at {touch:.2f} s", fill=YEL, font=font)
            frames.append(np.asarray(img))
        end_ok = fell is None and last_pel is not None and float(torch.norm(last_pel - last_ref)) < 1.0
    tags = [stake_tag(on[i], late_lv[i])[0] for i in range(n_st)]
    verdict = ("SUCCESS" if end_ok else (f"FELL at {fell:.2f} s" if fell is not None else "did not finish")) + \
              (f"   stakes: {' '.join(tags)}" if n_st else (f"   bridge contact at {touch:.2f} s" if touch is not None else "   no bridge contact"))
    img = Image.fromarray(frames[-1].copy())
    ImageDraw.Draw(img).text((10, 60), verdict, fill=GREEN if end_ok else RED, font=font)
    frames += [np.asarray(img)] * 60
    return frames, dict(success=bool(end_ok), fell=fell, stakes=tags, bridge_contact=touch, verdict=verdict)


def stake_tag(on, late):
    """ON = the foot stood fully on the stake (stake contact, no floor contact); PART = only straddling (the foot
    touched the stake and the floor at the same time, e.g. heel on the stake, toes on the floor); GROUND = never
    touched the stake.  Within the reference stance +-0.2 s, otherwise prefixed LATE- (rhythm)."""
    if on == 2:
        return "ON", GREEN
    if late == 2:
        return "LATE-ON", GREEN
    if on == 1:
        return "PART", YEL
    if late == 1:
        return "LATE-PART", YEL
    return "GROUND", RED


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", action="append", required=True, help="LABEL=pt:z_dir")
    ap.add_argument("--split", default="test")
    ap.add_argument("--clips", default="")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--slow", type=float, default=2.0)
    ap.add_argument("--out", default="/root/gpufree-data/hnu_v2/runs/videos/batch")
    a = ap.parse_args()
    split = json.load(open(os.environ["HNU_SPLIT_JSON"]))
    clips = a.clips.split(",") if a.clips else split[a.split]
    os.makedirs(a.out, exist_ok=True)
    cols = {c: [] for c in clips}
    summ = {c: {} for c in clips}
    for s in a.student:
        label, rest = s.split("=", 1)
        pt, zdir = rest.split(":")
        env, act, names = make_env(pt, zdir, (480, 640))
        for clip in clips:
            frames, info = run_clip(env, act, names, clip, label, a.seed)
            cols[clip].append(frames)
            summ[clip][label] = info
            print(f"{clip} {label}: {info['verdict']}", flush=True)
        env.close()
    import imageio.v2 as imageio
    for clip in clips:
        col = cols[clip]
        n = max(len(f) for f in col)
        col = [f + [f[-1]] * (n - len(f)) for f in col]
        sep = np.full((col[0][0].shape[0], 6, 3), 255, np.uint8)
        frames = [np.concatenate(sum([[cc[i], sep] for cc in col], [])[:-1], 1) for i in range(n)]
        imageio.mimsave(f"{a.out}/{clip}.mp4", frames, fps=50 / a.slow, codec="libx264", quality=7)
    json.dump(summ, open(f"{a.out}/summary.json", "w"), indent=1)
    print("->", a.out)


if __name__ == "__main__":
    main()
