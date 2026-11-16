"""E2E videos of the 9 real-robot layouts (user request), current deployment = Deployment_D (S2k student,
bridge post bases as measured: bar 4 cm).  One English command of the handbook per layout:
  ① command -> ② router v4 (gate + action; grey zone -> confirmation, the operator answers "是" = the command's intent)
  -> ③ retrieval top-5 library clips -> ④ planner (candidate clock / z on THIS layout) + preview = 4 nominal rollouts
  of every candidate with the deployed ONNX policy (CPU MuJoCo, deploy/sim2sim.py, 0-20 ms latency)
  -> ⑤ execution = the best previewed candidate (ties: retrieval order), a NEW rollout (trial index 4, unselected),
  rendered from the saved states (HNU_S2S_TRACE).
Candidate planning: test layouts = data/z_cand_plan_f1 slot of (layout clip, candidate) + geom_cand_f1 (the tables of
the paper's chain); N3 (= D2 shifted 10 cm) uses D2's tables with N3's bridge; N1 / N2 (stake spacing x1.05 / x0.95)
re-plan every candidate on the perturbed stakes with the plan4 rules of make_layout_perturb.py (library clip given
instead of the nearest one).
  python e2e_robot_demo.py prep     (mjlab venv + pylibs_sk / pylibs_tf; router + retrieval) -> OUT/demo.json
  python e2e_robot_demo.py cands    (candidate dirs OUT/cand/<layout>/<clip>/)
  python e2e_robot_demo.py choose   (after the preview sim2sim runs) -> demo.json chosen / preview
  python e2e_robot_demo.py render   (mjlab venv, MUJOCO_GL=egl)    -> OUT/exec_<layout>.mp4
  python e2e_robot_demo.py compose                                   -> OUT/<layout>.mp4
Driver: tools/e2e_robot_demo.sh"""
import json
import os
import subprocess
import sys

import numpy as np

H = "/root/gpufree-data/hnu_v2"
H4 = "/root/gpufree-data/hnu_v2_4090"  # router assets + candidate tables (from the 4090 server)
LD = "/root/gpufree-data/robot_sim_check/v6/in/planned_layouts"
EXPORT = f"{H}/deploy_exports/v6g10xm_base_S2k_s5_mDh40/L3"  # = Deployment_D/model
OUT = os.environ.get("HNU_DEMO_OUT", f"{H}/runs/videos/e2e_robot_D")
ACTS = ["CrouchedTraverse", "ThreeStepPoleRun", "FiveStepPoleRun"]
ZH = {"CrouchedTraverse": "独木桥", "ThreeStepPoleRun": "三步桩", "FiveStepPoleRun": "五步桩"}
EN = {"CrouchedTraverse": "Crouched traverse under a bridge", "ThreeStepPoleRun": "Three-step Pole Run",
      "FiveStepPoleRun": "Five-step Pole Run"}
CMD = {  # handbook v7 commands (explicit E / implicit I)
    "E1": "Crouch down and go under the beam in front of you to the other side.",
    "I1": "There's a pole across the way at about head height; get to the far side without climbing over it.",
    "E2": "Get to the end by stepping on the three short posts one after another.",
    "I2": "Three little round blocks are on the floor ahead; keep your feet off the ground between them all the way to the end.",
    "E3": "Step across the row of five posts, alternating left and right feet.",
    "I3": "Treat the five wooden blocks ahead like stepping stones across a river.",
}
LAYOUTS = [("D1", "E1"), ("D2", "I1"), ("N3", "E1"), ("S1", "E2"), ("S2", "I2"), ("N1", "E2"),
           ("T1", "E3"), ("T2", "I3"), ("N2", "E3")]
NOTE = {"N1": "S1 with stake spacing x1.05", "N2": "T2 with stake spacing x0.95", "N3": "D2 shifted 10 cm left"}
SHORT = {"CrouchedTraverse": "Crouched traverse", "ThreeStepPoleRun": "Three-step Pole Run", "FiveStepPoleRun": "Five-step Pole Run"}
FONT_R, FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"  # no CJK on the videos (user, )
PREVIEW, EXEC_TRIAL = 4, 4


def folder(lid):
    return next(f"{LD}/{d}" for d in sorted(os.listdir(LD)) if d.split("_")[0] == lid)


def layout_clip(lid):
    return next(f for f in os.listdir(folder(lid)) if f.endswith("_motion.npz"))[:-len("_motion.npz")]


# ------------------------------------------------------------------------------------------------ router + retrieval
def prep():
    import torch
    import torch.nn.functional as F
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sys.path.insert(0, f"{H4}/code/stage4_latent")
    os.environ["HNU_TEXT_ENCODER"] = "minilm"
    from retrieve_skill import build_model, hashed_text_vector
    DL = f"{H4}/code/data_lang"
    ins = json.load(open(f"{DL}/instructions_v1.json", encoding="utf-8"))
    v3 = json.load(open(f"{DL}/instructions_val_v3.json", encoding="utf-8"))
    aug = json.load(open(f"{DL}/instructions_aug_v3.json", encoding="utf-8"))
    augb = json.load(open(f"{DL}/instructions_aug_v3b.json", encoding="utf-8"))
    zdir = f"{H4}/runs/latent_v5t/seed42/Z3"
    ck = torch.load(f"{zdir}/best_motion_latent.pt", map_location="cpu", weights_only=False)
    model, _, _ = build_model(ck)
    q = model.task_semantic_encoder
    q.load_state_dict(torch.load(f"{zdir}/router_query_encoder_ft3.pt", map_location="cpu", weights_only=False)["state"])
    q.eval()
    g = np.load(f"{zdir}/train_latents.npz", allow_pickle=True)
    trial = np.array([str(t) for t in g["trial_id"]])
    gacts = np.array([t.rsplit("_", 1)[0] for t in trial])
    K = F.normalize(torch.tensor(g["z_skill"], dtype=torch.float32), dim=-1)
    masks = [torch.tensor(gacts == a) for a in ACTS]
    goal = {a: sorted(set(str(x) for x, y in zip(g["task_goal"], gacts) if y == a)) for a in ACTS}

    def enc(texts):
        X = torch.tensor(np.stack([np.r_[hashed_text_vector(f"task_goal: {t}".strip(), 384), 1.0] for t in texts]).astype(np.float32))
        with torch.no_grad():
            return F.normalize(q(X), dim=-1)

    route = lambda E: torch.stack([(E @ K.T)[:, m].max(1).values for m in masks], 1).numpy()  # noqa: E731
    pos = {a: list(dict.fromkeys(goal[a] + ins["train"][a] + ins["val"][a] + aug["train"][a])) for a in ACTS}
    negs = list(dict.fromkeys(ins["val_reject"] + aug["reject"] + augb["reject"]))
    P_act = np.array([i for i, a in enumerate(ACTS) for _ in pos[a]])
    EP, EN_ = enc([t for a in ACTS for t in pos[a]]), enc(negs)

    def feats(E, self_pos=None, self_neg=None):  # = router_gate4.py
        SP, SN = (E @ EP.T).numpy(), (E @ EN_.T).numpy()
        if self_pos is not None:
            SP[np.arange(len(E)), self_pos] = -1.0
        if self_neg is not None:
            SN[np.arange(len(E)), self_neg] = -1.0
        per = [np.sort(SP[:, P_act == i], 1)[:, ::-1] for i in range(3)]
        top1 = np.stack([s[:, 0] for s in per], 1)
        top3 = np.stack([s[:, :3].mean(1) for s in per], 1)
        o = np.sort(top1, 1)
        k3 = top3[np.arange(len(E)), top1.argmax(1)]
        ns = np.sort(SN, 1)[:, ::-1]
        r = np.sort(route(E), 1)
        return np.c_[o[:, -1], o[:, -1] - o[:, -2], k3, ns[:, 0], ns[:, :3].mean(1), o[:, -1] - ns[:, 0], r[:, -1], r[:, -1] - r[:, -2]]

    X = np.r_[np.c_[feats(EP, self_pos=np.arange(len(EP))), EP.numpy()], np.c_[feats(EN_, self_neg=np.arange(len(EN_))), EN_.numpy()]]
    y = np.r_[np.ones(len(EP)), np.zeros(len(EN_))]
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000).fit(sc.transform(X), y)
    prob = lambda E: clf.predict_proba(sc.transform(np.c_[feats(E), E.numpy()]))[:, 1]  # noqa: E731
    vt = [t for a in ACTS for t in v3["val"][a]] + v3["val_reject"]
    yv = np.array([i for i, a in enumerate(ACTS) for _ in v3["val"][a]] + [-1] * len(v3["val_reject"]))
    Ev = enc(vt)
    pv, qv = route(Ev).argmax(1), prob(Ev)
    best, tau = None, 0.5
    for th in np.unique(np.r_[0.0, np.quantile(qv, np.linspace(0, 1, 101))]):  # val3 rule of router_gate4.py
        rej = qv < th
        ex, rj = np.mean((pv == yv) & ~rej & (yv >= 0)) / np.mean(yv >= 0), np.mean(rej[yv < 0])
        if rj >= 0.90 and (best is None or (ex, rj) > best):
            best, tau = (ex, rj), float(th)
    insc = yv >= 0
    tau_low = float(max(th for th in np.unique(np.r_[0.0, qv]) if th <= tau and np.mean(qv[insc] < th) <= 0.01))
    PJ = json.load(open(f"{H4}/data/z_cand_plan_f1/pairs.json"))
    demos = []
    for lid, cid in LAYOUTS:
        text = CMD[cid]
        E = enc([text])
        S = route(E)[0]
        a = ACTS[int(S.argmax())]
        p = float(prob(E)[0])
        grey = bool(tau_low <= p < tau)
        clip = layout_clip(lid)
        intent = clip.rsplit("_", 1)[0]
        answer = ("是" if a == intent else "否") if grey else None
        d = dict(id=lid, cmd_id=cid, text=text, folder=os.path.basename(folder(lid)), clip=clip, routed=a,
                 gate_prob=round(p, 3), tau=round(tau, 3), tau_low=round(tau_low, 3), grey=grey, answer=answer,
                 accepted=bool(a == intent and (p >= tau or answer == "是")),
                 route_scores={ACTS[i]: round(float(S[i]), 3) for i in range(3)})
        if d["accepted"]:
            sc_ = (E @ K.T).numpy()[0]
            score = {c: float(sc_[trial == c].max()) for c in PJ["library"][a] if (trial == c).any()}
            base = layout_clip("D2") if lid == "N3" else clip
            slot = {c: kk for kk, pr in PJ["pairs"].items() for n, c in pr.items() if n == base}
            ok = (lambda c: c in slot) if lid in ("D1", "D2", "N3", "S1", "S2", "T1", "T2") else \
                (lambda c: nstakes(c) == nstakes(clip))
            top = [c for c in sorted(score, key=score.get, reverse=True) if ok(c)][:5]
            d["candidates"] = [dict(clip=c, slot=slot.get(c), retrieval=round(score[c], 3)) for c in top]
        demos.append(d)
        print(json.dumps(d, ensure_ascii=False))
    os.makedirs(OUT, exist_ok=True)
    json.dump(dict(tau=tau, tau_low=tau_low, demos=demos), open(f"{OUT}/demo.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)


def nstakes(c):
    t = c.rsplit("_", 1)[0]
    if t == "CrouchedTraverse":
        return 0
    geo = "ref_stakes_s93_h10" if t == "ThreeStepPoleRun" else "ref_stakes_s83_h10"
    return len(json.load(open(f"{H}/data/{geo}/{t}/{c}_stakes.json"))["stakes"])


# ------------------------------------------------------------------------------------------------ candidate tables
def link(src, dst):
    if os.path.lexists(dst):
        os.remove(dst)
    os.symlink(os.path.realpath(src), dst)


def cands():
    D = f"{H}/data"
    GEO = {"ThreeStepPoleRun": f"{D}/ref_stakes_s93_h10", "FiveStepPoleRun": f"{D}/ref_stakes_s83_h10"}
    split = json.load(open(f"{D}/clip_split_cv/fold1/split.json"))
    train = set(split["train"])

    def yaw_of(q):
        w, x, y, z = q
        return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    def stk(t, name, j=None):
        j = j or json.load(open(os.path.join(GEO[t], t, f"{name}_stakes.json")))
        ref = np.load(os.path.join(GEO[t], t, f"{name}_motion.npz"))
        pts = np.array([p[:2] for p in j["stake_xyz_50hz"]])
        win = np.array([(s["stance"][0] * 50.0 / j["fps"], s["stance"][1] * 50.0 / j["fps"]) for s in j["stakes"]])
        return dict(j=j, p0=ref["body_pos_w"][0, 0, :2], yaw0=yaw_of(ref["body_quat_w"][0, 0]), pts=pts, win=win,
                    T=len(ref["body_pos_w"]))

    names = {t: sorted(f[:-len("_stakes.json")] for f in os.listdir(os.path.join(GEO[t], t)) if f.endswith("_stakes.json")) for t in GEO}
    ST = {n: stk(t, n) for t in GEO for n in names[t]}
    REG = {t: np.polyfit([np.linalg.norm(ST[n]["pts"][0] - ST[n]["p0"]) for n in names[t] if n in train],
                         [ST[n]["win"][0, 0] for n in names[t] if n in train], 1) for t in GEO}

    def warp(z, T, ad, asrc):  # = make_layout_perturb.warp
        if not (np.all(np.diff(ad) > 0) and np.all(np.diff(asrc) > 0)):
            ad, asrc = [0.0, T - 1.0], [0.0, len(z) - 1.0]
        tt = np.clip(np.interp(np.arange(T), ad, asrc), 0, len(z) - 1)
        return np.stack([np.interp(tt, np.arange(len(z)), z[:, k]) for k in range(z.shape[1])], 1).astype(z.dtype)

    def plan_for(n, pts, c):  # = make_layout_perturb.plan with the library clip c given (not the nearest one)
        t = n.split("_")[0]
        a, b = ST[n], ST[c]
        seg = lambda p0, p: np.linalg.norm(np.diff(np.vstack([p0, p]), axis=0), axis=1)  # noqa: E731
        r = np.clip(seg(a["p0"], pts) / np.maximum(seg(b["p0"], b["pts"]), 1e-3), 0.6, 1.6)
        ws = b["win"]
        gaps = np.diff(np.concatenate([[0.0], ws.flatten()]))[0::2]
        stance = ws[:, 1] - ws[:, 0]
        out, tt = [], 0.0
        for k in range(len(ws)):
            s0 = float(np.polyval(REG[t], np.linalg.norm(pts[0] - a["p0"]))) if k == 0 else tt + gaps[k] * r[k]
            out.append((s0, s0 + stance[k]))
            tt = s0 + stance[k]
        out = np.array(out)
        if out[-1, 1] > a["T"] - 11:
            out *= (a["T"] - 11) / out[-1, 1]
        return out

    dm = json.load(open(f"{OUT}/demo.json", encoding="utf-8"))
    for d in dm["demos"]:
        if not d.get("candidates"):
            continue
        L, clip = folder(d["id"]), d["clip"]
        t = clip.rsplit("_", 1)[0]
        for c in d["candidates"]:
            o = f"{OUT}/cand/{d['id']}/{c['clip']}"
            os.makedirs(o, exist_ok=True)
            link(f"{L}/{clip}_motion.npz", f"{o}/{clip}_motion.npz")
            if t == "CrouchedTraverse":  # bridge: this layout's bridge, candidate z of the (D2) table (no clock)
                link(f"{L}/{clip}_bridge.json", f"{o}/{clip}_bridge.json")
                link(f"{H4}/data/z_cand_plan_f1/{c['slot']}/Z3/{clip}_z.npy", f"{o}/{clip}_z.npy")
            elif d["id"] not in ("N1", "N2"):  # test stake layout: the chain's candidate table (planned clock + z)
                g = f"{H4}/data/geom_cand_f1/{c['slot']}/{os.path.basename(GEO[t])}/{t}/{clip}_stakes.json"
                link(g, f"{o}/{clip}_stakes.json")
                link(f"{H4}/data/z_cand_plan_f1/{c['slot']}/Z3/{clip}_z.npy", f"{o}/{clip}_z.npy")
            else:  # N1 / N2: re-plan candidate c on the perturbed stakes of this layout
                j = json.load(open(f"{L}/{clip}_stakes.json"))
                pts = np.array([p[:2] for p in j["stake_xyz_50hz"]])
                wn = plan_for(clip, pts, c["clip"])
                for k, s_ in enumerate(j["stakes"]):
                    s_["stance"] = [int(round(wn[k, 0] * j["fps"] / 50.0)), int(round(wn[k, 1] * j["fps"] / 50.0))]
                j["planned_from"] = c["clip"]
                if os.path.lexists(f"{o}/{clip}_stakes.json"):
                    os.remove(f"{o}/{clip}_stakes.json")
                json.dump(j, open(f"{o}/{clip}_stakes.json", "w"))
                a = ST[clip]
                ad = [0.0] + wn.flatten().tolist() + [a["T"] - 1.0]
                asrc = [0.0] + ST[c["clip"]]["win"].flatten().tolist() + [ST[c["clip"]]["T"] - 1.0]
                zn = np.load(f"{D}/z_tables_cv_f1/Z3/{clip}_z.npy")
                np.save(f"{o}/{clip}_z.npy", warp(np.load(f"{D}/z_tables_cv_f1/Z3/{c['clip']}_z.npy"), len(zn), ad, asrc))
            print(d["id"], c["clip"], "->", o)


def choose():
    dm = json.load(open(f"{OUT}/demo.json", encoding="utf-8"))
    for d in dm["demos"]:
        if not d.get("candidates"):
            continue
        for c in d["candidates"]:
            R = json.load(open(f"{OUT}/cand/{d['id']}/{c['clip']}/preview.json"))["rows"]
            c["preview"] = [int(r["success_v2"]) for r in sorted(R, key=lambda r: r["trial"])][:PREVIEW]
        d["chosen"] = max(range(len(d["candidates"])), key=lambda i: (sum(d["candidates"][i]["preview"]), -i))
        print(d["id"], [(c["clip"][-3:], sum(c["preview"])) for c in d["candidates"]], "->", d["candidates"][d["chosen"]]["clip"])
    json.dump(dm, open(f"{OUT}/demo.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)


# ------------------------------------------------------------------------------------------------ rendering
def render():
    import mujoco
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype(FONT_R, 20)
    dm = json.load(open(f"{OUT}/demo.json", encoding="utf-8"))
    m = mujoco.MjModel.from_binary_path(f"{EXPORT}/scene.mjb")
    m.vis.global_.offwidth, m.vis.global_.offheight = 1280, 720
    W, Hh, slow = 660, 720, float(os.environ.get("HNU_DEMO_SLOW", "1.5"))
    for d in dm["demos"]:
        if "chosen" not in d:
            continue
        c = d["candidates"][d["chosen"]]
        tr = np.load(f"{OUT}/cand/{d['id']}/{c['clip']}/trace/{d['clip']}_{EXEC_TRIAL}.npz")
        dd = mujoco.MjData(m)
        dd.mocap_pos[:], dd.mocap_quat[:] = tr["mocap_pos"], tr["mocap_quat"]
        vis = [p for p in tr["mocap_pos"] if p[2] > -1.0]  # obstacles of this layout (others parked below the floor)
        ob = np.array(vis)[:, :2] if vis else tr["qpos"][:, :2]
        q0 = tr["qpos"]
        # obstacle line (PCA of the obstacle centres); camera at the side of the line, looking across it
        u = np.linalg.svd(ob - ob.mean(0))[2][0] if len(ob) > 1 else np.array([1.0, 0.0])
        az = np.degrees(np.arctan2(u[1], u[0])) - 90.0
        cam = mujoco.MjvCamera()
        cam.type, cam.distance, cam.elevation, cam.azimuth = mujoco.mjtCamera.mjCAMERA_FREE, 4.8, -24.0, az
        rnd = mujoco.Renderer(m, Hh, W)
        dt = float(tr["control_dt"])
        out = f"{OUT}/exec_{d['id']}.mp4"
        ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{Hh}",
                               "-r", "25", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", out],
                              stdin=subprocess.PIPE)
        look = q0[0, :3].copy()
        n = len(q0)
        step = 1.0 / (25 * slow * dt)  # control steps per output frame
        res = "Success" if bool(tr["success_v2"]) else ("Fall" if str(tr["cause"]) == "fall" else "Not completed")
        i = 0.0
        while True:
            k = min(int(round(i)), n - 1)
            dd.qpos[:] = q0[k]
            mujoco.mj_forward(m, dd)
            look = 0.9 * look + 0.1 * np.r_[dd.qpos[0:2], 0.55]
            cam.lookat[:] = look
            rnd.update_scene(dd, cam)
            im = Image.fromarray(rnd.render())
            dr = ImageDraw.Draw(im)
            dr.text((14, 10), f"{d['id']} {SHORT[d['clip'].rsplit('_', 1)[0]]}  |  sim, nominal  |  {slow:g}x slower", font=font,
                    fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
            dr.text((14, 40), f"t = {k * dt:4.1f} s", font=font, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
            if k == n - 1:
                dr.text((14, 70), f"Result: {res}", font=font, fill=(120, 255, 120) if res == "Success" else (255, 110, 110),
                        stroke_width=2, stroke_fill=(0, 0, 0))
            ff.stdin.write(np.asarray(im).tobytes())
            if k == n - 1:
                for _ in range(25):  # hold the last frame 1 s
                    ff.stdin.write(np.asarray(im).tobytes())
                break
            i += step
        ff.stdin.close()
        ff.wait()
        d["exec_result"] = res
        print("->", out, res)
    json.dump(dm, open(f"{OUT}/demo.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)


# ------------------------------------------------------------------------------------------------ composition
def compose():
    import re
    from PIL import Image, ImageDraw, ImageFont
    FONT, FB = FONT_R, FONT_B
    f = lambda s, b=False: ImageFont.truetype(FB if b else FONT, s)  # noqa: E731
    W, Hh, PW, fps = 1280, 720, 620, 25
    dm = json.load(open(f"{OUT}/demo.json", encoding="utf-8"))

    def panel(d, stage):
        im = Image.new("RGB", (PW, Hh), (24, 26, 32))
        dr = ImageDraw.Draw(im)
        y = 18
        dr.text((24, y), f"{d['id']}  {SHORT[d['clip'].rsplit('_', 1)[0]]}  |  {NOTE.get(d['id'], 'test trial ' + d['clip'][-3:])}",
                font=f(18, True), fill=(150, 200, 255))
        y += 36
        dr.text((24, y), "1. Command (" + ("explicit" if d["cmd_id"].startswith("E") else "implicit") + ")", font=f(21, True), fill=(255, 255, 255))
        y += 32
        line, lines = "", []
        for ch in re.findall(r"[A-Za-z0-9’'.,;:!?-]+\s*|\s+|.", f"“{d['text']}”"):
            if dr.textlength(line + ch, font=f(20)) > PW - 60 and line.strip():
                lines.append(line)
                line = ch.lstrip()
            else:
                line += ch
        for ln in lines + [line]:
            dr.text((40, y), ln, font=f(20), fill=(255, 230, 150))
            y += 28
        y += 10
        if stage < 2:
            return im
        dr.text((24, y), "2. Router v4 (gate + action)", font=f(21, True), fill=(255, 255, 255))
        y += 32
        col = (120, 230, 120) if d["accepted"] else (255, 110, 110)
        if d["grey"]:
            dr.text((40, y), f"Gate {d['gate_prob']:.2f} (grey zone {d['tau_low']:.2f}-{d['tau']:.2f}) -> confirm", font=f(19), fill=(255, 210, 90))
            y += 28
            dr.text((40, y), f"Robot: \"Do you mean {EN[d['routed']]}?\"", font=f(18), fill=(255, 230, 150))
            y += 28
            dr.text((40, y), f"Operator: \"{'Yes' if d['answer'] == '是' else 'No'}\"", font=f(18), fill=(255, 230, 150))
        else:
            dr.text((40, y), f"Gate {d['gate_prob']:.2f} (threshold {d['tau']:.2f}) -> {'accept' if d['accepted'] else 'reject'}", font=f(19), fill=col)
        y += 28
        dr.text((40, y), f"Action: {EN[d['routed']]}", font=f(19), fill=(220, 220, 220))
        y += 40
        if stage < 3 or not d.get("candidates"):
            return im
        dr.text((24, y), "3. Retrieval top 5  ->  4. Planner + preview", font=f(21, True), fill=(255, 255, 255))
        y += 32
        for x, s in ((60, "Candidate clip"), (300, "Retrieval"), (410, f"Preview x{PREVIEW}")):
            dr.text((x, y), s, font=f(17), fill=(160, 160, 160))
        y += 26
        for i, c in enumerate(d["candidates"]):
            colr = (120, 230, 120) if i == d["chosen"] else (220, 220, 220)
            if i == d["chosen"]:
                dr.text((36, y), ">", font=f(18, True), fill=colr)
            dr.text((60, y), c["clip"], font=f(17), fill=colr)
            dr.text((300, y), f"{c['retrieval']:.2f}", font=f(17), fill=colr)
            dr.text((410, y), " ".join("✓" if x else "✗" for x in c["preview"]) + f"   {sum(c['preview'])}/{PREVIEW}", font=f(17), fill=colr)
            y += 26
        y += 14
        if stage < 4:
            return im
        ch = d["candidates"][d["chosen"]]
        dr.text((24, y), "5. Execution", font=f(21, True), fill=(255, 255, 255))
        y += 32
        dr.text((40, y), f"Deployed policy + {ch['clip']}", font=f(19), fill=(220, 220, 220))
        y += 28
        dr.text((40, y), "ONNX, CPU MuJoCo, 0-20 ms latency; new unselected rollout", font=f(17), fill=(170, 170, 170))
        if d.get("exec_result"):
            y += 30
            dr.text((40, y), f"Result: {d['exec_result']}", font=f(20, True),
                    fill=(120, 230, 120) if d["exec_result"] == "Success" else (255, 110, 110))
        return im

    dark = np.full((Hh, W - PW, 3), 12, np.uint8)
    for d in dm["demos"]:
        out = f"{OUT}/{d['id']}_{d['folder'].split('_', 1)[1]}.mp4"
        ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{Hh}",
                               "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", out],
                              stdin=subprocess.PIPE)

        def hold(st, sec, right=dark, final=False):
            dd = dict(d) if final else {k: v for k, v in d.items() if k != "exec_result"}
            fr = np.concatenate([np.array(panel(dd, st)), right], 1).tobytes()
            for _ in range(max(1, int(round(sec * fps)))):
                ff.stdin.write(fr)
            return right

        hold(1, 2.5)
        hold(2, 5.0 if d["grey"] else 3.0)
        last = dark
        if d.get("candidates"):
            hold(3, 4.5)
            r = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", f"{OUT}/exec_{d['id']}.mp4", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                "-"], capture_output=True).stdout
            frames = np.frombuffer(r, np.uint8).reshape(-1, Hh, W - PW, 3)
            p4 = np.array(panel({k: v for k, v in d.items() if k != "exec_result"}, 4))
            for fr in frames[:-25]:
                ff.stdin.write(np.concatenate([p4, fr], 1).tobytes())
            last = frames[-1]
        hold(4 if d.get("candidates") else 2, 2.5, last, final=True)
        ff.stdin.close()
        ff.wait()
        print("->", out)


if __name__ == "__main__":
    {"prep": prep, "cands": cands, "choose": choose, "render": render, "compose": compose}[sys.argv[1]]()
