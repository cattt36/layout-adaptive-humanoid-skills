"""Rollout-calibrated executability Q_exec (paper Results 8.3).

collect : run a policy (student / MoE teacher) in the distillation env with random pushes and per-env action noise;
          store the deployable observation (proprio history + z_skill of the profile), whether the robot fails
          (termination) within the next H steps, and (clip, frame) bookkeeping for the offline heuristic.
train   : QualityNet -> P(no failure in the next H steps).  Env-disjoint train / calibration / test split.
          Compared scores on the test rollouts:
            heuristic      offline reference risk (inverse-dynamics torque ratio + joint-limit excess, next 1 s)
            uncalibrated   raw network probability
            temperature    temperature scaling fitted on the calibration split
            isotonic       isotonic regression fitted on the calibration split
          Metrics: AUROC, AUPRC (failure detection), Brier, NLL, ECE (15 bins), coverage-risk curve, selective
          dangerous-failure rate at fixed coverage.  z_quality = 12-d penultimate activation.

  python quality.py collect --policy student --student <pt> --latent-dir <z/Zk> --out <npz>
  python quality.py train --data <npz> [...] --heuristic-dir <data/heuristic> --out <dir>
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

HORIZON = 50  # 1 s at 50 Hz


def collect(a):
    import hnu_tasks  # noqa: F401
    from hnu_tasks import TASKS
    from hnu_tasks.distill import MoETeacher, hnu_distill_env_cfg
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg
    from distill_train import student_from_ckpt

    device = "cuda:0"
    torch.manual_seed(a.seed)
    ck = torch.load(a.student, map_location=device, weights_only=False) if a.policy == "student" else None
    z_offsets = tuple(ck.get("z_offsets", (0, 25))) if ck else (0, 25)
    cfg = hnu_distill_env_cfg(a.latent_dir, z_offsets=z_offsets, scene=ck.get("scene", "flat") if ck else "flat",
                              hist=ck.get("hist", 5) if ck else 5, lin_vel=ck.get("lin_vel", False) if ck else False)
    cfg.scene.num_envs = a.num_envs
    cfg.seed = a.seed
    push = load_env_cfg("HNU-Track-FiveStepPoleRun").events["push_robot"]
    push.interval_range_s = (0.5, 2.0)
    vr = dict(push.params["velocity_range"])
    push.params["velocity_range"] = {k: (lo * a.push_scale, hi * a.push_scale) for k, (lo, hi) in vr.items()}
    cfg.events["push_robot"] = push
    env = ManagerBasedRlEnv(cfg=cfg, device=device)
    obs, _ = env.reset()
    if a.policy == "teacher":
        pol = MoETeacher({t: getattr(a, f"ckpt_{t}") for t in TASKS}, device)
        act_fn = lambda o: pol(o["teacher"], o["task"])  # noqa: E731
    else:
        st = student_from_ckpt(ck, env.action_manager.total_action_dim, device)
        grp = ck.get("obs_group", "student")
        act_fn = lambda o: st(o[grp])  # noqa: E731
    noise = torch.rand(a.num_envs, 1, device=device) ** 2  # most envs low noise, some high
    O, TERM, META = [], [], []
    with torch.inference_mode():
        for _ in range(a.steps):
            act = act_fn(obs)
            act = act + a.action_noise * noise * torch.randn_like(act)
            O.append(obs["student"].half().cpu())
            META.append(obs["meta"].to(torch.int32).cpu())
            obs, _, term, trunc, _ = env.step(act)
            TERM.append(term.cpu())
    O, TERM, META = torch.stack(O), torch.stack(TERM), torch.stack(META)
    T, B = TERM.shape
    fail_future = torch.zeros(T, B, dtype=torch.bool)
    next_fail = torch.full((B,), 10 ** 9, dtype=torch.int64)
    for t in range(T - 1, -1, -1):
        next_fail = torch.where(TERM[t], torch.full_like(next_fail, t), next_fail)
        fail_future[t] = (next_fail - t) < HORIZON
    keep = ((torch.arange(T)[:, None] < (T - HORIZON)) | fail_future).flatten()
    names = env.command_manager.get_term("motion").motion.names
    np.savez_compressed(a.out, obs=O.reshape(T * B, -1)[keep].numpy(), fail=fail_future.flatten()[keep].numpy(),
                        clip=META[..., 0].flatten()[keep].numpy(), frame=META[..., 1].flatten()[keep].numpy(),
                        env=np.tile(np.arange(B), T)[keep.numpy()], clip_names=np.array(names))
    print(json.dumps(dict(samples=int(keep.sum()), fail_rate=float(fail_future.flatten()[keep].float().mean()))))


class QualityNet(torch.nn.Module):
    def __init__(self, dim, zq=12, n_aux=0):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("std", torch.ones(dim))
        self.body = torch.nn.Sequential(torch.nn.Linear(dim, 256), torch.nn.ELU(), torch.nn.Linear(256, 128),
                                        torch.nn.ELU(), torch.nn.Linear(128, zq), torch.nn.Tanh())
        self.head = torch.nn.Linear(zq, 1)
        # training-time auxiliary head: z_quality -> level-allowed robot / human segment metrics (stage5/make_quality_aux)
        self.aux = torch.nn.Linear(zq, n_aux) if n_aux else None

    def z_quality(self, x):
        return self.body((x - self.mean) / (self.std + 1e-2))

    def forward(self, x):  # logit of P(success over the next horizon)
        return self.head(self.z_quality(x)).squeeze(-1)


# ------------------------------------------------------------------ metrics

def auroc(score, label):
    o = np.argsort(-score)
    l = label[o]
    tp, fp = np.cumsum(l), np.cumsum(1 - l)
    if tp[-1] == 0 or fp[-1] == 0:
        return float("nan")
    return float(np.trapezoid(np.r_[0, tp / tp[-1]], np.r_[0, fp / fp[-1]]))


def auprc(score, label):
    o = np.argsort(-score)
    l = label[o]
    tp = np.cumsum(l)
    prec = tp / np.arange(1, len(l) + 1)
    return float((prec * l).sum() / max(l.sum(), 1))


def ece(p, y, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    e, rel = 0.0, []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (p >= a) & ((p < b) if b < 1 else (p <= b))
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
            rel.append([float(p[m].mean()), float(y[m].mean()), int(m.sum())])
    return float(e), rel


def isotonic_fit(x, y):
    """Pool-adjacent-violators (non-decreasing); returns knots for np.interp."""
    o = np.argsort(x)
    V, W, X = [], [], []
    for v, xx in zip(y[o].astype(float), x[o]):
        V.append(v); W.append(1.0); X.append(xx)
        while len(V) > 1 and V[-2] > V[-1]:
            v2, w2 = V.pop(), W.pop(); x2 = X.pop()
            V[-1] = (V[-1] * W[-1] + v2 * w2) / (W[-1] + w2); W[-1] += w2; X[-1] = x2
    return np.array(X), np.array(V)


def coverage_risk(p_success, fail, grid=np.linspace(0.1, 1.0, 19)):
    o = np.argsort(-p_success, kind="stable")
    f = fail[o]
    return [[round(float(c), 3), float(f[: max(int(round(c * len(f))), 1)].mean())] for c in grid]


def evaluate(p, fail):
    y = (~fail).astype(float)
    e, rel = ece(p, y)
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    cr = coverage_risk(p, fail)
    at = {f"risk@cov{int(round(c * 100))}": r for c, r in cr if int(round(c * 100)) in (50, 80, 90)}
    return dict(auroc=auroc(p, y), auprc_failure=auprc(-p, fail.astype(float)), brier=float(np.mean((p - y) ** 2)),
                nll=float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc))), ece=e, reliability=rel,
                coverage_risk=cr, **at)


def heuristic_scores(clip, frame, names, hdir):
    out = np.zeros(len(clip), np.float32)
    for i, n in enumerate(names):
        m = clip == i
        if m.any():
            t = np.load(os.path.join(hdir, f"{n}.npy"))
            out[m] = t[np.clip(frame[m], 0, len(t) - 1)]
    return out


def aux_targets(path, clip, frame, names):
    """[N, K] auxiliary targets (NaN = unavailable) for samples (clip index, clip frame)."""
    d = np.load(path)
    keys = list(d["keys"])
    out = np.full((len(clip), len(keys)), np.nan, np.float32)
    for i, n in enumerate(names):
        m = clip == i
        if m.any() and f"t_{n}" in d.files:
            t = d[f"t_{n}"]
            out[m] = t[np.clip(frame[m], 0, len(t) - 1)]
    return out, keys


def train(a):
    Os, Fs, Es, Cs, Rs, Ns = [], [], [], [], [], None
    for i, p in enumerate(a.data):
        d = np.load(p)
        # observations stay float16 in memory (they were stored as float16); minibatches are cast on the GPU.
        # The container has a 100 GB memory limit and the v6 observation is 2225-d.
        Os.append(d["obs"]); Fs.append(d["fail"]); Es.append(d["env"] + 100000 * i)
        Cs.append(d["clip"]); Rs.append(d["frame"]); Ns = list(d["clip_names"])
    cat = lambda xs: xs[0] if len(xs) == 1 else np.concatenate(xs)  # noqa: E731  (no copy for a single file)
    O, F, E, C, R = map(cat, (Os, Fs, Es, Cs, Rs))
    del Os
    rng = np.random.default_rng(0)
    envs = rng.permutation(np.unique(E))
    n5, n10 = len(envs) // 5, len(envs) // 10
    te = np.isin(E, envs[:n5])
    ca = np.isin(E, envs[n5 : n5 + n10])
    tr = ~(te | ca)
    if a.train_frac < 1.0:  # label-scarce regime (e.g. few real-robot rollouts): keep a random subset of train envs
        tr_envs = np.setdiff1d(envs, envs[: n5 + n10])
        keep = np.random.default_rng(100 + a.seed).permutation(tr_envs)[: max(int(a.train_frac * len(tr_envs)), 2)]
        tr &= np.isin(E, keep)
    device = "cuda:0"
    torch.manual_seed(a.seed)
    # training data stays in CPU memory, minibatches go to the GPU (the v6 observation is 2225-d: the full set of
    # ~3 M samples does not fit on a 24 GB GPU)
    Xtr, Ytr = torch.from_numpy(O[tr]), torch.from_numpy(~F[tr]).float()
    A, akeys = (aux_targets(a.aux, C[tr], R[tr], Ns) if a.aux else (None, []))
    if A is not None and A.shape[1] == 0:
        A = None
    net = QualityNet(O.shape[1], n_aux=0 if A is None else A.shape[1]).to(device)
    if A is not None:
        At = torch.from_numpy(np.nan_to_num(A))
        Am = torch.from_numpy(np.isfinite(A)).float()
    s1 = torch.zeros(O.shape[1], dtype=torch.float64, device=device)
    s2 = torch.zeros_like(s1)
    for i in range(0, len(Xtr), 65536):  # chunked mean / std (float64)
        xb = Xtr[i : i + 65536].to(device).double()
        s1 += xb.sum(0); s2 += (xb * xb).sum(0)
    mu = s1 / len(Xtr)
    net.mean.copy_(mu.float()); net.std.copy_((s2 / len(Xtr) - mu * mu).clamp(min=0).sqrt().float())
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    epochs = int(min(a.epochs / a.train_frac, 300))  # keep the number of gradient steps comparable
    for _ in range(epochs):
        p = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 16384):
            idx = p[i : i + 16384]
            zq = net.z_quality(Xtr[idx].to(device, non_blocking=True).float())
            loss = torch.nn.functional.binary_cross_entropy_with_logits(net.head(zq).squeeze(-1), Ytr[idx].to(device))
            if A is not None and a.lambda_aux > 0:
                m = Am[idx].to(device)
                loss = loss + a.lambda_aux * ((net.aux(zq) - At[idx].to(device)) ** 2 * m).sum() / m.sum().clamp(min=1)
            opt.zero_grad(); loss.backward(); opt.step()

    def logits(X):
        with torch.no_grad():
            return torch.cat([net(torch.from_numpy(X[i : i + 65536]).to(device).float())
                              for i in range(0, len(X), 65536)]).cpu().numpy()

    lc, lt = logits(O[ca]), logits(O[te])
    yc = (~F[ca]).astype(float)
    Ts = np.exp(np.linspace(np.log(0.25), np.log(8), 200))
    nll = [np.mean(np.logaddexp(0, -lc / T) * yc + np.logaddexp(0, lc / T) * (1 - yc)) for T in Ts]
    T_best = float(Ts[int(np.argmin(nll))])
    kx, ky = isotonic_fit(1 / (1 + np.exp(-lc)), yc)
    p_unc = 1 / (1 + np.exp(-lt))
    p_tmp = 1 / (1 + np.exp(-lt / T_best))
    p_iso = np.interp(p_unc, kx, ky)
    h_c = heuristic_scores(C[ca], R[ca], Ns, a.heuristic_dir)
    h_t = heuristic_scores(C[te], R[te], Ns, a.heuristic_dir)
    hx, hy = isotonic_fit(-h_c, yc)
    p_heu = np.interp(-h_t, hx, hy)
    res = dict(aux=a.aux, lambda_aux=a.lambda_aux, train_frac=a.train_frac, n_aux_targets=len(akeys),
               n_human_aux_targets=sum(str(k).startswith("human") for k in akeys), seed=a.seed,
               cal_auroc=auroc(1 / (1 + np.exp(-lc)), yc),  # model selection uses this split only (never test)
               cal_nll=float(np.mean(np.logaddexp(0, -lc) * yc + np.logaddexp(0, lc) * (1 - yc))),
               train_samples=int(tr.sum()), cal_samples=int(ca.sum()), test_samples=int(te.sum()),
               test_success_rate=float((~F[te]).mean()), temperature=T_best, no_filter_risk=float(F[te].mean()),
               heuristic=evaluate(p_heu, F[te]), uncalibrated=evaluate(p_unc, F[te]),
               temperature_scaled=evaluate(p_tmp, F[te]), isotonic=evaluate(p_iso, F[te]))
    os.makedirs(a.out, exist_ok=True)
    torch.save(dict(state=net.state_dict(), dim=O.shape[1], horizon=HORIZON, temperature=T_best,
                    isotonic=(kx.tolist(), ky.tolist())), os.path.join(a.out, "quality_net.pt"))
    json.dump(res, open(os.path.join(a.out, "quality_eval.json"), "w"), indent=1)
    keys = ("auroc", "auprc_failure", "brier", "ece", "risk@cov80")
    print(json.dumps({k: ({kk: round(vv, 4) for kk, vv in v.items() if kk in keys} if isinstance(v, dict) else v)
                      for k, v in res.items()}))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--policy", choices=["student", "teacher"], default="student")
    c.add_argument("--student", default=None)
    for t in ("ThreeStepPoleRun", "FiveStepPoleRun", "CrouchedTraverse"):
        c.add_argument(f"--ckpt-{t}", default=None)
    c.add_argument("--latent-dir", required=True)
    c.add_argument("--num-envs", type=int, default=2048)
    c.add_argument("--steps", type=int, default=1500)
    c.add_argument("--push-scale", type=float, default=1.5)
    c.add_argument("--action-noise", type=float, default=0.3)
    c.add_argument("--seed", type=int, default=1)
    c.add_argument("--out", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", nargs="+", required=True)
    t.add_argument("--heuristic-dir", required=True)
    t.add_argument("--epochs", type=int, default=15)
    t.add_argument("--aux", default=None, help="auxiliary target table (make_quality_aux.py) for the level")
    t.add_argument("--lambda-aux", type=float, default=0.0)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--train-frac", type=float, default=1.0, help="share of training envs used (label-scarce study)")
    t.add_argument("--out", required=True)
    a = ap.parse_args()
    collect(a) if a.cmd == "collect" else train(a)


if __name__ == "__main__":
    main()
