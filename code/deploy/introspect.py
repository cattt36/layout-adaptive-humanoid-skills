"""Print what a deployment needs to replicate from the distillation env (one-off introspection)."""
import os

import torch

import hnu_tasks  # noqa: F401
from hnu_tasks.distill import hnu_distill_env_cfg
from mjlab.envs import ManagerBasedRlEnv

H = "/root/gpufree-data/hnu_v2"
cfg = hnu_distill_env_cfg(f"{H}/data/z_tables_v5/Z0", play=True, z_offsets=(0, 5, 10, 15, 20, 25, 30, 40, 50),
                          scene="geom", hist=20, lin_vel=True)
cfg.scene.num_envs = 2
print("terminations:")
for k, v in cfg.terminations.items():
    print(" ", k, getattr(v.func, "__name__", v.func), v.params)
cmd = cfg.commands["motion"]
print("cmd body_names", cmd.body_names, "anchor", cmd.anchor_body_name)
print("pose_range", cmd.pose_range, "vel_range", cmd.velocity_range, "joint_range", cmd.joint_position_range)
print("decimation", cfg.decimation, "sim", cfg.sim)
print("actions", cfg.actions)
env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
robot = env.scene["robot"]
a = env.action_manager.get_term("joint_pos")
print("action attrs", [k for k in vars(a) if not k.startswith("__")])
for k in ("_scale", "_offset", "_joint_ids", "_joint_names"):
    v = getattr(a, k, None)
    print(k, v if not isinstance(v, torch.Tensor) else v[0].tolist() if v.dim() > 1 else v.tolist())
print("robot.joint_names", robot.joint_names)
print("default_joint_pos", robot.data.default_joint_pos[0].tolist())
m = env.sim.mj_model
import mujoco
print("nq nv nu nbody nmocap", m.nq, m.nv, m.nu, m.nbody, m.nmocap, "timestep", m.opt.timestep, "integrator", m.opt.integrator,
      "iters", m.opt.iterations, m.opt.ls_iterations, "cone", m.opt.cone, "solver", m.opt.solver)
print("joints", [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)])
print("actuators", [(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i), int(m.actuator_trnid[i, 0])) for i in range(m.nu)])
print("sensors", [(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SENSOR, i), int(m.sensor_adr[i]), int(m.sensor_dim[i])) for i in range(m.nsensor)])
print("mocap bodies", [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody) if m.body_mocapid[i] >= 0])
print("student terms", env.observation_manager.active_terms["student"])
print("student dims", env.observation_manager.group_obs_term_dim["student"])
print("env_origins", env.scene.env_origins.tolist())
print("data attrs", [k for k in dir(env.sim.data) if k in ("qpos", "qvel", "mocap_pos", "mocap_quat", "sensordata", "ctrl")])
print("qpos shape", tuple(env.sim.data.qpos.shape))
