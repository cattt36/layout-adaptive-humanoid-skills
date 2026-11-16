"""Export a distilled student for deployment (runs in the mjlab venv, GPU).

  python deploy/export_student.py --student <student_2000.pt> --latent-dir <z_tables_v5/Zk> --out <dir> [--trace-steps 150]

Writes to <dir>:
  policy.onnx      obs [N, D] -> actions [N, 29]; the observation normaliser is part of the graph
  policy_meta.json everything a controller needs to build the observation and apply the action (term layout,
                   joint order, default pose, action scale/offset, PD gains, effort limits, z_skill offsets, sim
                   settings, termination rules for evaluation, push event, stake / bridge parameters)
  scene.mjb        the compiled single-env training scene (G1 + 5 stakes + bridge parts as mocap bodies)
  trace.npz        nominal rollout recorded in the training env (one clip per action): MuJoCo state, sensors,
                   mocap poses, student observation and action per step -> deploy/parity_check.py
"""
import argparse
import json
import os

import mujoco
import numpy as np
import torch

import hnu_tasks  # noqa: F401
from hnu_tasks import TASKS
from hnu_tasks.distill import hnu_distill_env_cfg
from hnu_tasks.multi_motion import resolve_motion_files
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

from distill_train import student_from_ckpt


def to_list(v):
    if isinstance(v, torch.Tensor):
        return v.detach().cpu().tolist()
    if isinstance(v, (tuple, list)):
        return [to_list(x) for x in v]
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True)
    ap.add_argument("--latent-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--trace-steps", type=int, default=150)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda:0"
    ck = torch.load(a.student, map_location=dev, weights_only=False)
    z_off = tuple(ck.get("z_offsets", (0, 25)))
    hist, lin_vel = ck.get("hist", 5), ck.get("lin_vel", False)
    cfg = hnu_distill_env_cfg(a.latent_dir, play=True, z_offsets=z_off, scene=ck.get("scene", "geom"), hist=hist,
                              lin_vel=lin_vel)
    cmd_cfg = cfg.commands["motion"]
    cmd_cfg.start_clip_mode = "env_mod"
    files = resolve_motion_files(cmd_cfg.motion_file)
    names = [os.path.basename(f).replace("_motion.npz", "") for f in files]
    cfg.scene.num_envs = len(files)
    env = ManagerBasedRlEnv(cfg=cfg, device=dev)
    obs, _ = env.reset()
    robot = env.scene["robot"]
    act = env.action_manager.get_term("joint_pos")
    st = student_from_ckpt(ck, env.action_manager.total_action_dim, dev)
    om = env.observation_manager
    terms = om.active_terms["student"]
    dims = [int(np.prod(d)) for d in om.group_obs_term_dim["student"]]
    tcfg = {n: om.get_term_cfg("student", n) for n in terms}
    layout = [dict(name=n, dim=d // max(tcfg[n].history_length, 1), history=int(tcfg[n].history_length), total=d)
              for n, d in zip(terms, dims)]
    m = env.sim.mj_model
    jn = list(robot.joint_names)
    # per-joint PD gains / effort limits in the policy joint order (actuators are grouped by motor type in the model)
    kp, kd, eff = {}, {}, {}
    for i in range(m.nu):
        j = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, int(m.actuator_trnid[i, 0])).split("/")[-1]
        kp[j], kd[j] = float(m.actuator_gainprm[i, 0]), float(-m.actuator_biasprm[i, 2])
        eff[j] = float(m.actuator_forcerange[i, 1])
    push = load_env_cfg("HNU-Track-FiveStepPoleRun").events["push_robot"]
    from hnu_tasks import stakes as stk
    from hnu_tasks.bridge import BASE_DIMS, _metas, _parts, part_names
    bdir = os.path.join(os.environ["HNU_BRIDGE_ROOT"], "CrouchedTraverse")
    meta = dict(
        student=a.student, latent_dir=a.latent_dir, profile=ck.get("profile"), it=ck.get("it"),
        obs_dim=int(ck["obs_dim"]), act_dim=int(env.action_manager.total_action_dim), obs_layout=layout,
        z_offsets=list(z_off), hist=hist, lin_vel=lin_vel,
        joint_names=jn, body_names=list(robot.body_names),
        default_joint_pos=to_list(robot.data.default_joint_pos[0]),
        action_scale=to_list(act._scale[0]), action_offset=to_list(act._offset[0]),
        kp=[kp[j] for j in jn], kd=[kd[j] for j in jn], effort_limit=[eff[j] for j in jn],
        sim=dict(timestep=float(m.opt.timestep), decimation=int(cfg.decimation), integrator="implicitfast",
                 control_dt=float(env.step_dt)),
        sensors=dict(ang_vel="robot/imu_ang_vel", lin_vel="robot/imu_lin_vel"),
        projected_gravity="root (pelvis) quaternion: R^T [0, 0, -1]",
        noise_in_training={n: to_list([tcfg[n].noise.n_min, tcfg[n].noise.n_max]) for n in terms
                           if getattr(tcfg[n], "noise", None) is not None},
        latency=dict(model="0-20 ms (0 or 1 control step), one lag per episode segment of 0.5 s, shared by the "
                           "sensor terms; last action and z_skill undelayed",
                     sensor_terms=[n for n in terms if not (n == "actions" or n.startswith("z_"))]),
        tracking=dict(body_names=list(cmd_cfg.body_names), anchor=cmd_cfg.anchor_body_name,
                      anchor_pos_z=0.25, anchor_ori=0.8,
                      ee_bodies=["left_ankle_roll_link", "right_ankle_roll_link", "left_wrist_yaw_link",
                                 "right_wrist_yaw_link"], ee_z=0.25,
                      reset_joint_noise=to_list(cmd_cfg.joint_position_range)),
        push=dict(interval_s=to_list(push.interval_range_s), velocity_range=to_list(push.params.get("velocity_range"))),
        stakes=dict(height=stk.STAKE_H, radius=stk.STAKE_R, tol=stk.STAKE_TOL, n=stk.N_STAKES, n_ahead=2),
        bridge=dict(parts=part_names(bdir),
                    beam_half=float(_parts(_metas(bdir)[0])["plank"]["half"][2]),
                    post_base=BASE_DIMS),  # HNU_POST_BASE: post bases in the scene, else None
        motion_dirs=sorted({os.path.dirname(f) for f in files}),
    )
    json.dump(meta, open(os.path.join(a.out, "policy_meta.json"), "w"), indent=1)
    # compiled scene (env 0); mocap poses are written per clip by the controller
    mujoco.mj_saveModel(m, os.path.join(a.out, "scene.mjb"), None)
    # ONNX (normaliser + MLP)
    dummy = torch.zeros(1, int(ck["obs_dim"]), device=dev)
    torch.onnx.export(st, (dummy,), os.path.join(a.out, "policy.onnx"), input_names=["obs"], output_names=["actions"],
                      dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}}, opset_version=17, dynamo=False)
    # trace: first clip of each action
    pick = [next(i for i, n in enumerate(names) if n.startswith(t)) for t in TASKS]
    rec = {k: [] for k in ("qpos", "qvel", "mocap_pos", "mocap_quat", "sensordata", "obs", "action", "frame", "done")}
    d = env.sim.data
    cmd = env.command_manager.get_term("motion")
    with torch.inference_mode():
        o = obs["student"]
        for _ in range(a.trace_steps):
            ac = st(o)
            for k, v in (("qpos", d.qpos), ("qvel", d.qvel), ("mocap_pos", d.mocap_pos), ("mocap_quat", d.mocap_quat),
                         ("sensordata", d.sensordata), ("obs", o), ("action", ac)):
                rec[k].append(to_np(v)[pick])
            rec["frame"].append(to_np(cmd.time_steps - cmd.motion.clip_start[cmd.clip_id])[pick])
            obs, _, term, trunc, _ = env.step(ac)
            rec["done"].append(to_np(term | trunc)[pick])
            o = obs["student"]
    np.savez_compressed(os.path.join(a.out, "trace.npz"), clips=np.array([names[i] for i in pick]),
                        env_origins=to_np(env.scene.env_origins)[pick], **{k: np.stack(v) for k, v in rec.items()})
    print("exported", a.out, "obs_dim", ck["obs_dim"], "clips", [names[i] for i in pick])


def to_np(v):
    return (v if isinstance(v, torch.Tensor) else torch.as_tensor(v)).detach().float().cpu().numpy()


if __name__ == "__main__":
    main()
