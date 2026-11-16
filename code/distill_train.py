"""DAgger distillation of the MoE teacher (3 action experts) into a deployable latent-conditioned student.

  python distill_train.py --profile Z0 --latent-dir <.../z_tables/Z0> --ckpt-ThreeStepPoleRun ... \
      --ckpt-FiveStepPoleRun ... --ckpt-CrouchedTraverse ... --num-envs 4096 --iters 4000 --out <dir>

The student acts in the environment (after a short teacher-driven warm-up, beta decays 1 -> 0); every visited
state is labelled with the teacher action and the student is regressed onto the aggregated recent data.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch

import hnu_tasks  # noqa: F401
from hnu_tasks import TASKS
from hnu_tasks.distill import MoETeacher, hnu_distill_env_cfg, scene_teacher_groups
from mjlab.envs import ManagerBasedRlEnv


class Normalizer(torch.nn.Module):
    def __init__(self, dim, until=2_000_000):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.zeros(()))
        self.until = until

    @torch.no_grad()
    def update(self, x):
        if self.count >= self.until:
            return
        n = x.shape[0]
        m, v = x.mean(0), x.var(0, unbiased=False)
        tot = self.count + n
        delta = m - self.mean
        self.mean += delta * n / tot
        self.var = (self.var * self.count + v * n + delta ** 2 * self.count * n / tot) / tot
        self.count = tot

    def forward(self, x):
        return (x - self.mean) / (self.var.sqrt() + 1e-2)


class Student(torch.nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(512, 256, 128)):
        super().__init__()
        self.norm = Normalizer(obs_dim)
        layers, d = [], obs_dim
        for h in hidden:
            layers += [torch.nn.Linear(d, h), torch.nn.ELU()]
            d = h
        layers.append(torch.nn.Linear(d, act_dim))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, obs):
        return self.mlp(self.norm(obs))


class MoEStudent(torch.nn.Module):
    """Latent-gated soft mixture of experts: gate g(z_skill) over K expert MLPs (all see the full observation).
    Roughly parameter-matched to Student (K=3 experts of 256-128-64)."""

    def __init__(self, obs_dim, act_dim, z_dim=32, k=3, hidden=(256, 128, 64)):
        super().__init__()
        self.norm = Normalizer(obs_dim)
        self.z_dim = z_dim

        def mlp(i, o, hs):
            layers, d = [], i
            for h in hs:
                layers += [torch.nn.Linear(d, h), torch.nn.ELU()]
                d = h
            layers.append(torch.nn.Linear(d, o))
            return torch.nn.Sequential(*layers)

        self.experts = torch.nn.ModuleList([mlp(obs_dim, act_dim, hidden) for _ in range(k)])
        self.gate = mlp(z_dim, k, (64,))

    def gates(self, obs):
        return torch.softmax(self.gate(self.norm(obs)[:, -self.z_dim:]), -1)

    def forward(self, obs):
        x = self.norm(obs)
        g = torch.softmax(self.gate(x[:, -self.z_dim:]), -1)
        return torch.stack([e(x) for e in self.experts], 1).mul(g.unsqueeze(-1)).sum(1)


def make_student(arch, obs_dim, act_dim, z_dim=32, hidden=None):
    if arch == "moe":
        return MoEStudent(obs_dim, act_dim, z_dim=z_dim)
    return Student(obs_dim, act_dim, hidden=tuple(hidden) if hidden else (512, 256, 128))


def student_from_ckpt(ck, act_dim, device):
    """Rebuild a student from its checkpoint dict (architecture, z layout and hidden sizes are stored in it)."""
    z_off = tuple(ck.get("z_offsets", (0, 25)))
    st = make_student(ck.get("arch", "mlp"), ck["obs_dim"], act_dim, z_dim=ck.get("z_dim", 16) * len(z_off),
                      hidden=ck.get("hidden")).to(device)
    st.load_state_dict(ck["student"])
    st.eval()
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument("--latent-dir", required=True)
    for t in TASKS:
        ap.add_argument(f"--ckpt-{t}", required=True)
    ap.add_argument("--num-envs", type=int, default=4096)
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--beta-iters", type=int, default=300, help="iterations over which teacher mixing decays to 0")
    ap.add_argument("--buffer-iters", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--minibatch", type=int, default=32768)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--obs-group", default="student", choices=["student", "student_ref", "student_nolatent"])
    ap.add_argument("--arch", default="mlp", choices=["mlp", "moe"])
    ap.add_argument("--seed", type=int, default=None, help="torch + env seed (None = legacy behaviour)")
    ap.add_argument("--z-offsets", default="0,10,25,40", help="future offsets (control steps) of the z_skill samples")
    ap.add_argument("--hidden", default="512,256,128", help="student MLP hidden sizes")
    ap.add_argument("--init-from", default=None, help="continue DAgger from this student checkpoint")
    ap.add_argument("--hist", type=int, default=5, help="proprioceptive history length (control steps)")
    ap.add_argument("--lin-vel", action="store_true", help="v6: noisy estimated base linear velocity as student input")
    ap.add_argument("--fail-weight", type=float, default=0.0,
                    help="DAgger sample weight boost for states that terminate within the next 0.5 s (1 + w)")
    ap.add_argument("--action-noise", type=float, default=0.0,
                    help="DART-style exploration: Gaussian noise on the EXECUTED action of a random half of the envs "
                         "(teacher labels stay clean), so the student also sees recovery states")
    ap.add_argument("--scene", default="flat", choices=["flat", "bridge", "geom"],
                    help="bridge: CrouchedTraverse under the bridge with obstacle features for the student")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = "cuda:0"
    os.makedirs(args.out, exist_ok=True)
    json.dump(vars(args), open(os.path.join(args.out, "args.json"), "w"), indent=1)

    z_offsets = tuple(int(x) for x in args.z_offsets.split(","))
    cfg = hnu_distill_env_cfg(args.latent_dir, z_offsets=z_offsets, scene=args.scene, hist=args.hist, lin_vel=args.lin_vel)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        cfg.seed = args.seed
    cfg.scene.num_envs = args.num_envs
    env = ManagerBasedRlEnv(cfg=cfg, device=device)
    teacher = MoETeacher({t: getattr(args, f"ckpt_{t}") for t in TASKS}, device, scene_teacher_groups(args.scene))
    obs, _ = env.reset()
    s_dim = obs[args.obs_group].shape[1]
    z_dim = int(json.load(open(os.path.join(args.latent_dir, "split.json"))).get("z_dim", 16))
    hidden = [int(x) for x in args.hidden.split(",")]
    student = make_student(args.arch, s_dim, env.action_manager.total_action_dim, z_dim=z_dim * len(z_offsets),
                           hidden=hidden).to(device)
    if args.init_from:
        student.load_state_dict(torch.load(args.init_from, map_location=device, weights_only=False)["student"])
    opt = torch.optim.Adam(student.parameters(), lr=args.lr)
    noisy_env = (torch.rand(args.num_envs, 1, device=device) < 0.5).float()
    buf_o, buf_a, buf_w = [], [], []
    log = open(os.path.join(args.out, "train_log.jsonl"), "a")
    from torch.utils.tensorboard import SummaryWriter
    tb = SummaryWriter(os.path.join(os.environ.get("HNU_TB_ROOT", args.out), os.path.basename(os.path.normpath(args.out))))
    ep_len = torch.zeros(args.num_envs, device=device)
    done_lens, t0 = [], time.time()
    for it in range(args.iters):
        beta = max(0.0, 1.0 - it / max(args.beta_iters, 1))
        it_o, it_a, it_t = [], [], []
        n_term, n_done = 0, 0
        for _ in range(args.steps):
            with torch.no_grad():
                a_t = teacher(obs, obs["task"])
                student.norm.update(obs[args.obs_group])
                a_s = student(obs[args.obs_group])
                use_t = (torch.rand(args.num_envs, 1, device=device) < beta)
                act = torch.where(use_t, a_t, a_s)
                if args.action_noise > 0:
                    act = act + args.action_noise * noisy_env * torch.randn_like(act)
            it_o.append(obs[args.obs_group].to(torch.float16))  # fp16 replay buffer (v6 obs are ~2.2k-dim)
            it_a.append(a_t)
            obs, _, terminated, truncated, _ = env.step(act)
            it_t.append(terminated.clone())
            ep_len += 1
            done = terminated | truncated
            n_term += int(terminated.sum())
            n_done += int(done.sum())
            if done.any():
                done_lens += ep_len[done].tolist()
                ep_len[done] = 0
        # failure proximity: the env terminates (falls / loses track) within the remaining steps of this window
        # (<= 0.5 s); those states get weight 1 + fail_weight (labels are still the teacher's actions)
        Tm = torch.stack(it_t).float()  # [steps, B]
        soon = torch.flip(torch.cummax(torch.flip(Tm, [0]), 0).values, [0])
        buf_w.append((1.0 + args.fail_weight * soon).flatten())
        buf_o.append(torch.cat(it_o))
        buf_a.append(torch.cat(it_a))
        buf_o, buf_a, buf_w = buf_o[-args.buffer_iters:], buf_a[-args.buffer_iters:], buf_w[-args.buffer_iters:]
        O, A, W = torch.cat(buf_o), torch.cat(buf_a), torch.cat(buf_w)
        losses = []
        for _ in range(args.epochs):
            perm = torch.randperm(O.shape[0], device=device)
            for i in range(0, O.shape[0], args.minibatch):
                idx = perm[i : i + args.minibatch]
                err = ((student(O[idx].float()) - A[idx]) ** 2).mean(-1)
                loss = (err * W[idx]).sum() / W[idx].sum()
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                opt.step()
                losses.append(loss.item())
        if it % 10 == 0:
            rec = dict(it=it, beta=round(beta, 3), loss=sum(losses) / len(losses),
                       term_per_step=n_term / (args.steps * args.num_envs),
                       mean_ep_len=(sum(done_lens[-4000:]) / max(len(done_lens[-4000:]), 1)),
                       elapsed_min=round((time.time() - t0) / 60, 1))
            for key in ("beta", "loss", "term_per_step", "mean_ep_len"):
                tb.add_scalar(f"Distill/{key}", rec[key], it)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(json.dumps(rec), flush=True)
        if (it + 1) % 250 == 0 or it == args.iters - 1:
            torch.save(dict(student=student.state_dict(), obs_dim=s_dim, profile=args.profile, it=it + 1,
                            obs_group=args.obs_group, arch=args.arch, z_offsets=list(z_offsets), scene=args.scene,
                            z_dim=z_dim, hidden=hidden, init_from=args.init_from, action_noise=args.action_noise,
                            hist=args.hist, lin_vel=args.lin_vel, fail_weight=args.fail_weight),
                       os.path.join(args.out, f"student_{it + 1}.pt"))
    env.close()


if __name__ == "__main__":
    main()
