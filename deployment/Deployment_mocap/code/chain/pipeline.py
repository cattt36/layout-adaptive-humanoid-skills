"""Full chain in simulation: instruction -> router v4 (gate, grey zone) -> retrieval top-5 -> planner (plan4, per
candidate) -> preview (sim2sim rollouts per candidate, choose the best, refuse if < 2) -> execution (sim2sim).

Deployment_mocap (STANDARD START, new G1, waist free) changes on top of the vH version below:
  * planner = the precomputed standard-start plans planned_layouts/<layout>/cands/<candidate>/ (pipeline_v2
    tools/std_plan.py: the candidate's OWN standing / step-in z + plan4 timing of the skill).  The robot team's
    chain/planner.py plans on the original clock and must NOT be used with the standard start (D1 + candidate 029 on
    the old clock: the robot walked 0.45 m forward while it should stand).  Candidates without a standard-start plan
    (their clip has no valid start, e.g. CrouchedTraverse_017-023 / 026) are skipped and reported.
  * choice = most preview successes among the candidates whose median standing drift (pelvis displacement while the
    frame is held at 0) is <= DRIFT_MAX (bridge 0.20 m, stakes 0.10 m); none -> refuse
  * preview conditions: pendulum rope (HNU_S2S_ROPE_LEN 2 m), waist lock only if the model was trained with it

Deployment_H_mocap version of the robot-side chain/pipeline.py (version by the robot team):
drop-in replacement, same API (Chain, select, run).  Changes:
  * one model per task (models.json in the deploy root): CrouchedTraverse -> model_bridge (no body velocity),
    ThreeStepPoleRun / FiveStepPoleRun -> model_stakes (mocap body velocity); one worker pool per model
  * preview = the REAL conditions of the vH models, never the ideal ones: rest start with the frame-0 pose, rope 80 %
    body weight released over 10 s, frame held 12 s, start error 5 cm / 10 deg, contact / actuator randomisation, 0-40 ms
    latency, mocap heading / height bias, mocap velocity estimate, WAIST LOCK; stakes in addition 3 cm stake placement
    errors; >= 12 rollouts per candidate.  check_preview_opt() refuses ideal settings (the old ideal preview
    scored candidate 006 4/4, on the real conditions it is 0/12 -- the robot fell)
  * measured obstacles (runtime/obstacles_live.py, README 1.9): preview(..., measured=...) / run(..., measured=...)
    write the mocap-measured posts / stakes into every candidate's planned layout before its preview rollouts
Preview rollouts use trial ids >= PREVIEW_SEED0, executions 0..n-1, so a preview never replays an execution."""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
from multiprocessing import Pool

import numpy as np

D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(D)  # policy_meta.json paths are relative to the deploy root
sys.path.insert(0, os.path.join(D, "runtime"))
sys.path.insert(0, os.path.join(D, "chain"))
import sim2sim  # noqa: E402
from planner import Planner, layout_files  # noqa: E402

MODELS = json.load(open(os.path.join(D, os.environ.get("HNU_MODELS_JSON", "models.json"))))  # e.g. models_waist.json
EXPORT = {t: os.path.join(D, MODELS[t]) for t in ("CrouchedTraverse", "ThreeStepPoleRun", "FiveStepPoleRun")}
PREVIEW_SEED0 = 1000
N_PREVIEW_MIN = 12
GREY = "val3"  # grey zone [tau_low, tau) computed as in e2e_demo.py: 0.137-0.719 (handbook fig. 8: 0.14-0.72)
OPT_BASE = dict(latency="0-20", push=False, dr=False, base_rand=True, perc_bias=0.0, perc_noise=0.0, obst_shift=0.0,
                stake_jitter=0.0, drift=0.0, init_off=0.0, bridge_dz=0.0, start_yaw=0.0, crit="task")
# real conditions (README 1.2 / 1.6): rope start + real-robot errors + waist lock
OPT = dict(OPT_BASE, latency="0-40", start_yaw=10.0, start="rest", hold=12.0, start_xy=0.05, dr_x=True, vel_est=True,
           head_bias=3.0, height_bias=0.02, rope="0.8,10", start_rp=3.0)  # waist_lock: from the model (task_opt)
WAIST_LOCK = {t: bool(json.load(open(os.path.join(EXPORT[t], "policy_meta.json"))).get("waist_lock")) for t in EXPORT}
os.environ.setdefault("HNU_S2S_ROPE_LEN", "2.0")  # pendulum rope as in training / acceptance
DRIFT_MAX = {"CrouchedTraverse": 0.20, "ThreeStepPoleRun": 0.10, "FiveStepPoleRun": 0.10}
OPT_STAKES = dict(stake_jitter=0.03, obst_shift=0.03)  # stakes: placement errors on top


def check_preview_opt(opt, n):
    """refuse an ideal preview (env HNU_PREVIEW_ALLOW_IDEAL=1 overrides, e.g. for paper-style simulations)"""
    bad = []
    if opt.get("start") != "rest":
        bad.append("start must be 'rest' (frame-0 pose at rest), not the reference-velocity start")
    if not opt.get("rope"):
        bad.append("rope release missing (rope='0.8,10')")
    if float(opt.get("hold") or 0) < 10:
        bad.append("frame-0 hold < 10 s (rope release + standing)")
    if "waist_lock_model" in opt and bool(opt.get("waist_lock")) != opt["waist_lock_model"]:
        bad.append(f"waist_lock {opt.get('waist_lock')} but the model was trained with waist_lock {opt['waist_lock_model']}")
    if not opt.get("dr_x") or opt.get("latency") in (None, "none"):
        bad.append("real-robot errors missing (dr_x, latency 0-40)")
    if n < N_PREVIEW_MIN:
        bad.append(f"{n} rollouts per candidate < {N_PREVIEW_MIN}")
    if bad and os.environ.get("HNU_PREVIEW_ALLOW_IDEAL") != "1":
        raise RuntimeError("preview conditions are not the real ones:\n  - " + "\n  - ".join(bad))
    return bad


def select(previews, drifts=None, drift_max=None):
    """candidate choice = e2e_demo.py: most preview successes, ties -> earlier retrieval rank; refuse if < 2.
    only candidates with median standing drift <= drift_max (no drift data -> not eligible)."""
    ok = list(range(len(previews)))
    if drifts is not None and drift_max is not None:
        ok = [i for i in ok if drifts[i] is not None and drifts[i] <= drift_max]
    if not ok:
        return 0, True
    ch = max(ok, key=lambda i: (sum(previews[i]), -i))
    return ch, sum(previews[ch]) < 2


def _job(args):
    """one sim2sim rollout with a per-job planned geometry / z directory."""
    plan_dir, clip, trial, opt = args
    sim2sim.G["meta"]["motion_dirs"] = [plan_dir]
    sim2sim.G["meta"]["latent_dir"] = plan_dir
    r = sim2sim.run((clip, trial, opt))
    return dict(plan_dir=plan_dir, trial=trial, success=bool(r["success"]), success_v2=bool(r["success_v2"]),
                cause=r["cause"], hold_drift=(r.get("hold_drift") or [None])[0])


class Chain:
    def __init__(self, workers=None, grey=GREY, opt=None, router=True):
        self.opt = dict(OPT, **(opt or {}))
        self.workers = workers or os.cpu_count()
        self.planner = Planner()
        self.router = None
        if router:
            from router import Router
            self.router = Router(grey=grey)
        self.pools = {}  # model folder -> worker pool (sim2sim.G of that model)
        self.tmp = tempfile.mkdtemp(prefix="chain_")

    def pool(self, task):
        ex = EXPORT[task]
        if ex not in self.pools:
            self.pools[ex] = Pool(self.workers, initializer=sim2sim.init_worker, initargs=(ex, 0.005, None))
        return self.pools[ex]

    def task_opt(self, task):
        o = dict(self.opt, **(OPT_STAKES if task != "CrouchedTraverse" else {}))
        o.setdefault("waist_lock", WAIST_LOCK[task])
        o["waist_lock_model"] = WAIST_LOCK[task]
        return o

    def close(self):
        for p in self.pools.values():
            p.close()
            p.join()

    def rollouts(self, plan_dir, clip, trials, task):
        opt = self.task_opt(task)
        return self.pool(task).map(_job, [(plan_dir, clip, t, opt) for t in trials], chunksize=1)

    def plan(self, layout_dir, cand, measured=None):
        """the layout files + the candidate's precomputed standard-start plan (cands/<cand>/: z table, for
        stakes the planned stance windows); (None, None) if the candidate has none."""
        src = os.path.join(layout_dir, "cands", cand)
        if not os.path.isdir(src):
            return None, None
        out = os.path.join(self.tmp, os.path.basename(layout_dir.rstrip("/")), cand)
        if os.path.isdir(out):
            shutil.rmtree(out)
        os.makedirs(out)
        for f in os.listdir(layout_dir):
            if os.path.isfile(os.path.join(layout_dir, f)) and not f.endswith("_z.npy"):
                shutil.copy(os.path.join(layout_dir, f), out)
        for f in os.listdir(src):
            shutil.copy(os.path.join(src, f), out)
        res = json.load(open(os.path.join(src, "std_plan.json")))
        if measured:
            from obstacles_live import apply_to_dir
            print("\n".join(apply_to_dir(layout_dir, measured, out)), flush=True)
        return out, res

    def preview(self, layout_dir, cands, n=N_PREVIEW_MIN, measured=None):
        """all candidates' preview rollouts in one parallel batch; choice = most successes (ties: retrieval order)."""
        clip, task, _ = layout_files(layout_dir)
        opt = self.task_opt(task)
        check_preview_opt(opt, n)
        print(f"[预演] {task}: 模型 {os.path.basename(EXPORT[task])}，每个候选 {n} 次，吊绳起步 {opt['rope']}、保持 {opt['hold']} s、"
              f"锁腰 {opt.get('waist_lock')}、实物误差（起点 {opt['start_xy']} m / {opt['start_yaw']}°，0-40 ms，dr_x）"
              + (f"、摆桩误差 {opt['stake_jitter']} m" if task != "CrouchedTraverse" else "")
              + ("、实测障碍物" if measured else "、规划障碍物"), flush=True)
        t0 = time.time()
        dirs = [self.plan(layout_dir, c, measured)[0] for c in cands]
        miss = [c for c, d in zip(cands, dirs) if d is None]
        if miss:
            print(f"[预演] 没有标准起步规划的候选（跳过）：{miss}", flush=True)
        jobs = [(d, clip, PREVIEW_SEED0 + i, opt) for d in dirs if d is not None for i in range(n)]
        res = self.pool(task).map(_job, jobs, chunksize=1) if jobs else []
        out = []
        for c, d in zip(cands, dirs):
            rr = [r for r in res if d is not None and r["plan_dir"] == d]
            dr = [r["hold_drift"] for r in rr if r["hold_drift"] is not None]
            out.append(dict(clip=c, plan_dir=d, preview=[int(r["success_v2"]) for r in rr],
                            drift=float(np.median(dr)) if dr else None))
            print(f"[预演]   {c}: {sum(out[-1]['preview'])}/{len(rr)} 成功，站立挪动中位 "
                  + (f"{100 * out[-1]['drift']:.0f} cm" if dr else "-"), flush=True)
        ch, _ = select([c["preview"] for c in out], [c["drift"] for c in out], DRIFT_MAX[task])
        return out, ch, time.time() - t0

    def run(self, text, layout_dir, answer=None, n_exec=4, n_preview=int(os.environ.get("HNU_N_PREVIEW", "12")), top_k=5,
            trial0=0, measured=None):
        """answer: grey-zone reply ("yes" / "no"); None -> ask on the terminal.  Executions use trial ids
        trial0 .. trial0 + n_exec - 1 (different commands on one layout need disjoint ids to be independent).
        measured: mocap-measured obstacles {"posts" | "stakes": [[x, y] | None, ...]} (aligned frame), README 1.9."""
        clip, task, _ = layout_files(layout_dir)
        rec = dict(text=text, layout=os.path.basename(layout_dir.rstrip("/")))
        r = self.router(text, top_k=top_k)
        rec["route"] = {k: r[k] for k in ("action", "gate_prob", "tau", "tau_low", "decision", "route_scores")}
        if r["decision"] == "confirm":
            if answer is None:
                answer = input(f"机器人：“是要 {r['action_zh']}（{r['action']}）吗？” [是/否] ").strip()
                answer = "yes" if answer.lower() in ("是", "y", "yes") else "no"
            rec["route"]["answer"] = answer
            if answer != "yes":
                return dict(rec, outcome="rejected_after_confirm", moved=False)
        elif r["decision"] == "reject":
            return dict(rec, outcome="rejected", moved=False)
        if r["action"] != task:  # executed the wrong skill on this layout: counted as a failure, not simulated
            return dict(rec, outcome="wrong_action", moved=True)
        cands, ch, dt = self.preview(layout_dir, [c["clip"] for c in r["retrieval"]], n_preview, measured)
        for c, rr in zip(cands, r["retrieval"]):
            c["retrieval"] = rr["score"]
        rec.update(candidates=[{k: c[k] for k in ("clip", "retrieval", "preview", "drift")} for c in cands], chosen=ch,
                   preview_time_s=round(dt, 2), model=os.path.basename(EXPORT[task]),
                   preview_opt={k: v for k, v in self.task_opt(task).items() if v not in (None, False, 0, 0.0)})
        if select([c["preview"] for c in cands], [c["drift"] for c in cands], DRIFT_MAX[task])[1]:
            return dict(rec, outcome="preview_refused", moved=False)
        if n_exec <= 0:  # plan only (the robot executes: RoboJuDo HnuPipeline)
            return dict(rec, outcome="executed", moved=True)
        ex = self.rollouts(cands[ch]["plan_dir"], clip, range(trial0, trial0 + n_exec), task)
        rec["exec"] = [dict(success=e["success"], success_v2=e["success_v2"], cause=e["cause"]) for e in ex]
        rec["success_rate_v2"] = float(np.mean([e["success_v2"] for e in ex]))
        rec["success_rate"] = float(np.mean([e["success"] for e in ex]))
        return dict(rec, outcome="executed", moved=True)
