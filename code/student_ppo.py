"""PPO fine-tuning of a distilled student (C-2: closes the gap left by the teacher's privileged inputs).

The actor is the student itself (same deployable observation group "student": proprioception history + z_skill
samples + obstacle features), initialised from the DAgger checkpoint; the critic reads privileged state (tracking
critic terms + bridge look-ahead) and is warm-started from the bridge expert's critic.  Rewards / terminations are
the expert's tracking objective (+ bridge-contact penalty in the bridge scene).  Every level is fine-tuned with the
same settings; the result is written back in the student checkpoint format (evaluate with eval_student.py).

  python student_ppo.py --student <student.pt> --latent-dir <z_tables/Zk> --critic-from <bridge expert.pt>
      --iters 1500 --out <dir>
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict

import torch

import hnu_tasks  # noqa: F401
from hnu_tasks.distill import hnu_distill_env_cfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
from mjlab.tasks.tracking.config.g1.rl_cfg import unitree_g1_tracking_ppo_runner_cfg


def student_to_actor(ck: dict, init_std: float, act_dim: int) -> dict:
    sd = ck["student"]
    var = sd["norm.var"]
    out = {"obs_normalizer._mean": sd["norm.mean"][None].clone(), "obs_normalizer._var": var[None].clone(),
           "obs_normalizer._std": var.sqrt()[None].clone(), "obs_normalizer.count": sd["norm.count"].clone(),
           "distribution.std_param": torch.full((act_dim,), init_std)}
    out.update({k: v.clone() for k, v in sd.items() if k.startswith("mlp.")})
    return out


def actor_to_student(actor_sd: dict, ck: dict) -> dict:
    sd = {"norm.mean": actor_sd["obs_normalizer._mean"][0].cpu(), "norm.var": actor_sd["obs_normalizer._var"][0].cpu(),
          "norm.count": actor_sd["obs_normalizer.count"].cpu()}
    sd.update({k: v.cpu() for k, v in actor_sd.items() if k.startswith("mlp.")})
    return dict(ck, student=sd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True)
    ap.add_argument("--latent-dir", required=True)
    ap.add_argument("--critic-from", default=None, help="expert checkpoint whose critic matches the critic group")
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--num-envs", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--init-std", type=float, default=0.15)
    ap.add_argument("--entropy", type=float, default=0.0)
    ap.add_argument("--bridge-w", type=float, default=-5.0)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    device = "cuda:0"
    ck = torch.load(a.student, map_location="cpu", weights_only=False)
    scene = ck.get("scene", "flat")
    cfg = hnu_distill_env_cfg(a.latent_dir, z_offsets=tuple(ck.get("z_offsets", (0, 25))), scene=scene,
                              hist=ck.get("hist", 5), lin_vel=ck.get("lin_vel", False))
    cfg.scene.num_envs = a.num_envs
    cfg.seed = a.seed
    # privileged critic = tracking critic terms (+ bridge look-ahead in the bridge scene)
    base = unitree_g1_flat_tracking_env_cfg()
    critic_terms = {k: ObservationTermCfg(func=v.func, params=dict(v.params)) for k, v in base.observations["critic"].terms.items()}
    if scene == "bridge":
        from mjlab.managers.reward_manager import RewardTermCfg
        from hnu_tasks import bridge_contact
        tp = cfg.observations["teacher_perc"].terms["perception"]
        critic_terms["perception"] = ObservationTermCfg(func=tp.func, params=dict(tp.params))
        cfg.rewards["bridge_contact"] = RewardTermCfg(func=bridge_contact, weight=a.bridge_w)
    cfg.observations["critic"] = ObservationGroupCfg(terms=critic_terms, concatenate_terms=True, enable_corruption=False)
    env = ManagerBasedRlEnv(cfg=cfg, device=device)
    rl = unitree_g1_tracking_ppo_runner_cfg()
    rl.actor.hidden_dims = tuple(ck.get("hidden") or (512, 256, 128))
    rl.actor.distribution_cfg = dict(rl.actor.distribution_cfg, init_std=a.init_std)
    rl.obs_groups = {"actor": (ck.get("obs_group", "student"),), "critic": ("critic",)}
    rl.algorithm.learning_rate = a.lr
    rl.algorithm.entropy_coef = a.entropy
    rl.save_interval = 250
    rl.max_iterations = a.iters
    rl.logger = "tensorboard"
    rl.experiment_name = "student_ppo"
    wrapped = RslRlVecEnvWrapper(env, clip_actions=rl.clip_actions)
    os.makedirs(a.out, exist_ok=True)
    runner = MjlabOnPolicyRunner(wrapped, asdict(rl), log_dir=a.out, device=device)
    act_dim = env.action_manager.total_action_dim
    runner.alg.actor.load_state_dict({k: v.to(device) for k, v in student_to_actor(ck, a.init_std, act_dim).items()})
    if a.critic_from:
        csd = torch.load(a.critic_from, map_location=device, weights_only=False)["critic_state_dict"]
        runner.alg.critic.load_state_dict(csd)
    runner.learn(num_learning_iterations=a.iters)
    out = actor_to_student(runner.alg.actor.state_dict(), ck)
    out.update(ppo=dict(vars(a)), it=int(ck.get("it", 0)))
    torch.save(out, os.path.join(a.out, "student_ppo.pt"))
    json.dump(vars(a), open(os.path.join(a.out, "args.json"), "w"), indent=1)
    print("saved", os.path.join(a.out, "student_ppo.pt"))


if __name__ == "__main__":
    main()
