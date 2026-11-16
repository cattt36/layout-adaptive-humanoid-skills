"""consistency of the blind-tracker runtime (deploy/hnu_deploy.py TrackerPolicy) with the training
environment: roll out the exported tracker's torch actor in the mjlab env (1 env, play, fixed clip, start at rest with
a 1 s hold), record the actor observation of the env and the robot state, rebuild the observation with
TrackerPolicy.observation from the state, compare term by term.
  HNU_ACTOR_BLIND=1 HNU_WAIST_LOCK=1 ... python tools/tracker_obs_check.py <task_id> <clip> <export_dir> <motion_dir>
"""
import os
import sys

import numpy as np
import torch

import hnu_tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from hnu_tasks.multi_motion import resolve_motion_files

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "deploy"))
from hnu_deploy import RobotState, TrackerPolicy  # noqa: E402


def main():
    task_id, clip, ex, mdir = sys.argv[1:5]
    os.environ.update(HNU_REST_FRAC="1", HNU_STEP_FRAC="0", HNU_HOLD="1.0,1.0", HNU_HOLD_P="1", HNU_REST_XY="0", HNU_REST_YAW="0")
    cfg = load_env_cfg(task_id, play=True)
    cmd = cfg.commands["motion"]
    names = [os.path.basename(f).replace("_motion.npz", "") for f in resolve_motion_files(cmd.motion_file)]
    cmd.fixed_clip = names.index(clip)
    cmd.sampling_mode = "start"
    cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    obs, _ = env.reset()
    pol = TrackerPolicy(ex)
    c = env.command_manager.get_term("motion")
    robot = env.scene["robot"]
    from mjlab.envs import mdp as env_mdp
    mf = os.path.join(mdir, f"{clip}_motion.npz")
    errs, a_err = [], []
    from hnu_tasks.distill import ExpertActor
    actor = ExpertActor(os.environ["HNU_TRACKER_CKPT"], "cuda:0")
    first = True
    with torch.inference_mode():
        for t in range(150):
            o_env = obs["actor"][0].cpu().numpy()
            d = robot.data
            org = env.scene.env_origins[0].cpu().numpy()
            s = RobotState(ang_vel=env_mdp.builtin_sensor(env, sensor_name="robot/imu_ang_vel")[0].cpu().numpy(),
                           lin_vel=np.zeros(3), quat=d.root_link_quat_w[0].cpu().numpy(), pos=d.root_link_pos_w[0].cpu().numpy() - org,
                           q=d.joint_pos[0].cpu().numpy(), dq=d.joint_vel[0].cpu().numpy())
            if first:
                pol.reset(mf, s, None)
                first = False
            frame = int((c.time_steps[0] - c.motion.clip_start[c.clip_id[0]]).item())
            holding = bool(c.hold[0].item() > 0) if hasattr(c, "hold") else False
            pol.last_action = obs["actor"][0, -(29 + pol.n_perc):][:29].cpu().numpy() if pol.n_perc else obs["actor"][0, -29:].cpu().numpy()
            o_rt = pol.observation(s, frame, holding)
            errs.append(np.abs(o_rt - o_env))
            obs, _, term, trunc, _ = env.step(actor(obs["actor"]))
            if term.any() or trunc.any():
                break
    E = np.array(errs)
    seg = [("ref q", 0, 29), ("ref dq", 29, 58), ("anchor pos", 58, 61), ("anchor ori", 61, 67), ("lin vel", 67, 70),
           ("gyro", 70, 73), ("joint pos", 73, 102), ("joint vel", 102, 131), ("action", 131, 160), ("perception", 160, E.shape[1])]
    for n, a, b in seg:
        print(f"{n:11s} max |runtime - env| {E[:, a:b].max():.2e}")
    print("steps", len(E), "OBS CHECK", "OK" if max(E[:, :73].max(), E[:, 102:].max()) < 2e-3 and E[:, 73:102].max() < 2e-2 else "FAILED")


if __name__ == "__main__":
    main()
