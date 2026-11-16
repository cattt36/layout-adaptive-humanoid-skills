"""E2E demonstration videos (user request): instruction -> router v4 (gate) -> retrieval top-5 -> planner ->
simulation preview -> execution by the new L3 student, for a few hand-written commands (NOT from the sealed test set)
on fold-1 TEST layouts.  Same components as the evaluation: router v4 = ft3 query encoder + "lremb" gate trained and
thresholded exactly as stage4_latent/router_gate4.py (val3, rejection >= 90 %), retrieval = max cosine to the library
clips' z_skill windows (tools/expA_rank.py), planner clock / z of every (layout, candidate) pair = data/z_cand_plan_f1,
preview = the first 4 rollouts of that pair in the nominal candidate sweep (results_candplan_xm_k*_s1_dr0, L3).
  python e2e_demo.py prep  -> runs/videos/e2e_demo/demo.json      (CPU)
  python e2e_demo.py compose                                     (after tools/e2e_demo.sh recorded the executions)"""
import json
import os
import sys

import numpy as np

H = "/root/gpufree-data/hnu_v2"
OUT = f"{H}/runs/videos/e2e_demo"
ACTS = ["CrouchedTraverse", "ThreeStepPoleRun", "FiveStepPoleRun"]
# task rename, display only (server data keep the old names until the server-side rename):
# ThreeStepPoleRun -> ThreeStepPoleRun "Three-step Pole Run" 三步桩, FiveStepPoleRun -> FiveStepPoleRun "Five-step Pole Run" 三步桩/五步桩,
# CrouchedTraverse -> CrouchedTraverse "Crouched traverse under a bridge" 独木桥
NAME_ZH = {"CrouchedTraverse": "Crouched traverse under a bridge（独木桥）", "ThreeStepPoleRun": "Three-step Pole Run（三步桩）",
           "FiveStepPoleRun": "Five-step Pole Run（三步桩/五步桩）"}


def disp(s):
    return str(s).replace("ThreeStepPoleRun", "ThreeStepPoleRun").replace("FiveStepPoleRun", "FiveStepPoleRun").replace(
        "CrouchedTraverse", "CrouchedTraverse")
DEMOS = [  # (id, command, test layout used if accepted)
    ("d1_explicit", "沿着那排五根桩一路踩过去，速度快一点。", "FiveStepPoleRun_027"),
    ("d2_implicit", "前面那根横木比你的头还低，想办法到它后面去。", "CrouchedTraverse_008"),
    ("d3_asr", "跑过前面三根庄 每根都要采到", "ThreeStepPoleRun_011"),
    ("d4_out_of_scope", "把桌子推到墙边去。", None),
    ("d5_code_switch", "前面那三根 stakes，quick 一点 run 过去，每根都踩稳。", "ThreeStepPoleRun_002"),
    ("d6_english_implicit", "A short row of three wooden posts is ahead; hop across without touching the floor.", "ThreeStepPoleRun_012"),
    # grey zone (chosen AFTER scoring tools/grey_candidates.json, demonstration only): intent = bridge -> "是";
    # intent = stakes but routed to the bridge -> "否"
    ("d7_grey_yes", "把身子放低一点，往前走。", "CrouchedTraverse_024"),
    ("d8_grey_no", "那边地上的东西，踩着它们到那头去。", "FiveStepPoleRun_013"),
    # the d7 execution fell (chosen candidate 4/4 in the preview); the same command on the other bridge test layout,
    # recorded AFTER seeing that failure (demonstration only, both videos kept)
    ("d7b_grey_yes", "把身子放低一点，往前走。", "CrouchedTraverse_008"),
]
if os.environ.get("HNU_DEMO_IDS"):  # only these demos (demo.json / videos of the others are left as they are)
    DEMOS = [d for d in DEMOS if d[0] in os.environ["HNU_DEMO_IDS"].split(",")]


def prep():
    import torch
    import torch.nn.functional as F
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sys.path.insert(0, f"{H}/code/stage4_latent")
    os.environ["HNU_TEXT_ENCODER"] = "minilm"
    from retrieve_skill import build_model, hashed_text_vector
    DL = f"{H}/code/data_lang"
    ins = json.load(open(f"{DL}/instructions_v1.json", encoding="utf-8"))
    v3 = json.load(open(f"{DL}/instructions_val_v3.json", encoding="utf-8"))
    aug = json.load(open(f"{DL}/instructions_aug_v3.json", encoding="utf-8"))
    augb = json.load(open(f"{DL}/instructions_aug_v3b.json", encoding="utf-8"))
    zdir = f"{H}/runs/latent_v5t/seed42/Z3"
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
    EP, EN = enc([t for a in ACTS for t in pos[a]]), enc(negs)

    def feats(E, self_pos=None, self_neg=None):  # = router_gate4.py
        SP, SN = (E @ EP.T).numpy(), (E @ EN.T).numpy()
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

    X = np.r_[np.c_[feats(EP, self_pos=np.arange(len(EP))), EP.numpy()], np.c_[feats(EN, self_neg=np.arange(len(EN))), EN.numpy()]]
    y = np.r_[np.ones(len(EP)), np.zeros(len(EN))]
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
    # grey zone (; rule fixed before any grey-zone command was scored): tau_low = the largest threshold <= tau
    # at which at most 1 % of the in-scope val3 commands are rejected outright.  p >= tau: accept; tau_low <= p < tau:
    # the robot asks "是要 <action> 吗？" and the operator answers from the command's intent; p < tau_low: reject.
    ins = yv >= 0
    tau_low = float(max(th for th in np.unique(np.r_[0.0, qv]) if th <= tau and np.mean(qv[ins] < th) <= 0.01))
    gz = dict(tau=round(tau, 3), tau_low=round(tau_low, 3),
              val3_in_scope=dict(accept=round(float(np.mean(qv[ins] >= tau)), 3), ask=round(float(np.mean((qv[ins] >= tau_low) & (qv[ins] < tau))), 3),
                                 reject=round(float(np.mean(qv[ins] < tau_low)), 3)),
              val3_out_of_scope=dict(accept=round(float(np.mean(qv[~ins] >= tau)), 3), ask=round(float(np.mean((qv[~ins] >= tau_low) & (qv[~ins] < tau))), 3),
                                     reject=round(float(np.mean(qv[~ins] < tau_low)), 3)))
    print("grey zone:", json.dumps(gz, ensure_ascii=False))
    if os.environ.get("HNU_DEMO_SCORE"):  # only score candidate commands (no demo.json)
        for text in json.load(open(os.environ["HNU_DEMO_SCORE"], encoding="utf-8")):
            E = enc([text])
            p, a = float(prob(E)[0]), ACTS[int(route(E)[0].argmax())]
            print(f"{p:.3f} {'accept' if p >= tau else 'ASK' if p >= tau_low else 'reject':6s} {a:12s} {text}")
        return
    PJ = json.load(open(f"{H}/data/z_cand_plan_f1/pairs.json"))
    demos = []
    for did, text, L in DEMOS:
        E = enc([text])
        S = route(E)[0]
        a = ACTS[int(S.argmax())]
        p = float(prob(E)[0])
        grey = bool(tau_low <= p < tau)
        answer = ("是" if L is not None and L.startswith(a) else "否") if grey else None  # operator answers from the intent
        d = dict(id=did, text=text, routed=a, routed_zh=NAME_ZH[a], gate_prob=round(p, 3), tau=round(tau, 3),
                 tau_low=round(tau_low, 3), grey=grey, answer=answer,
                 accepted=bool(p >= tau or answer == "是"), route_scores={ACTS[i]: round(float(S[i]), 3) for i in range(3)})
        if d["accepted"] and L is not None and L.startswith(a):
            sc_ = (E @ K.T).numpy()[0]
            lib = PJ["library"][a]
            score = {c: float(sc_[trial == c].max()) for c in lib if (trial == c).any()}
            slot = {c: kk for kk, pr in PJ["pairs"].items() for n, c in pr.items() if n == L}
            top = [c for c in sorted(score, key=score.get, reverse=True) if c in slot][:5]
            cands = []
            for c in top:
                kk = slot[c]
                z = np.load(f"{H}/runs/results_candplan_xm_{kk}_s1_dr0/qexec_Z3.npz")
                names = [str(s) for s in z["names"]]
                ep = z["success_v2"][z["clip"] == names.index(L)].astype(float)
                cands.append(dict(clip=c, slot=kk, retrieval=round(score[c], 3), preview=[int(x) for x in ep[:4]]))
            ch = max(range(len(cands)), key=lambda i: (sum(cands[i]["preview"]), -i))
            d.update(layout=L, candidates=cands, chosen=ch, refuse_exec=sum(cands[ch]["preview"]) < 2)
        demos.append(d)
        print(json.dumps(d, ensure_ascii=False))
    os.makedirs(OUT, exist_ok=True)
    f = f"{OUT}/demo.json"
    if os.environ.get("HNU_DEMO_IDS") and os.path.exists(f):  # merge into the existing list
        old = [d for d in json.load(open(f, encoding="utf-8")) if d["id"] not in {x["id"] for x in demos}]
        demos = old + demos
    json.dump(demos, open(f, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def compose():
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont
    FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    FB = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
    f = lambda s, b=False: ImageFont.truetype(FB if b else FONT, s)  # noqa: E731
    W, Hh, PW = 1280, 720, 620
    demos = json.load(open(f"{OUT}/demo.json", encoding="utf-8"))
    if os.environ.get("HNU_DEMO_IDS"):
        demos = [d for d in demos if d["id"] in os.environ["HNU_DEMO_IDS"].split(",")]
    fps = 25

    def panel(d, stage):
        im = Image.new("RGB", (PW, Hh), (24, 26, 32))
        dr = ImageDraw.Draw(im)
        y = 24
        dr.text((24, y), "指令 → 路由 → 检索 → 规划 → 预演 → 执行", font=f(20, True), fill=(150, 200, 255))
        y += 44
        dr.text((24, y), "① 指令", font=f(22, True), fill=(255, 255, 255))
        y += 34
        import re
        line, lines = "", []  # wrap the command to the panel width: Latin words kept whole, CJK by character
        for ch in re.findall(r"[A-Za-z0-9’'.,;:!?-]+\s*|\s+|.", f"“{d['text']}”"):
            if dr.textlength(line + ch, font=f(22)) > PW - 60 and line.strip():
                lines.append(line)
                line = ch.lstrip()
            else:
                line += ch
        for ln in lines + [line]:
            dr.text((40, y), ln, font=f(22), fill=(255, 230, 150))
            y += 32
        y += 14
        if stage < 2:
            return im
        dr.text((24, y), "② 路由器 v4（门 + 动作）", font=f(22, True), fill=(255, 255, 255))
        y += 34
        col = (120, 230, 120) if d["accepted"] else (255, 110, 110)
        if d.get("grey"):  # grey zone: confirmation question first
            dr.text((40, y), f"门置信度 {d['gate_prob']:.2f}（灰区 {d['tau_low']:.2f}–{d['tau']:.2f}）→ 先确认", font=f(20), fill=(255, 210, 90))
            y += 30
            dr.text((40, y), f"机器人：“是要 {d['routed_zh'].split('（')[0]} 吗？”", font=f(20), fill=(255, 230, 150))
            y += 30
            dr.text((40, y), f"操作员：“{d['answer']}” → {'接受' if d['accepted'] else '不执行'}", font=f(20), fill=col)
        else:
            dr.text((40, y), f"门置信度 {d['gate_prob']:.2f}（阈值 {d['tau']:.2f}）→ {'接受' if d['accepted'] else '拒识'}", font=f(20), fill=col)
        y += 30
        if d["accepted"]:
            dr.text((40, y), f"动作：{d['routed_zh']}", font=f(20), fill=(220, 220, 220))
        elif d.get("grey"):
            dr.text((40, y), "机器人回复：“好的，我不动。”", font=f(20), fill=(220, 220, 220))
        else:
            dr.text((40, y), "机器人回复：“这个指令我不能执行。”", font=f(20), fill=(220, 220, 220))
        y += 44
        if stage < 3 or not d.get("candidates"):
            return im
        dr.text((24, y), "③ 检索前 5 个候选 + 规划器对齐 + ④ 仿真预演", font=f(22, True), fill=(255, 255, 255))
        y += 34
        for x, s in ((60, "候选片段"), (300, "检索分数"), (420, "预演 4 次")):
            dr.text((x, y), s, font=f(18), fill=(160, 160, 160))
        y += 28
        for i, c in enumerate(d["candidates"]):
            colr = (120, 230, 120) if i == d["chosen"] else (220, 220, 220)
            if i == d["chosen"]:
                dr.text((36, y), ">", font=f(18, True), fill=colr)
            dr.text((60, y), disp(c["clip"]), font=f(18), fill=colr)
            dr.text((300, y), f"{c['retrieval']:.2f}", font=f(18), fill=colr)
            dr.text((420, y), " ".join("成" if x else "败" for x in c["preview"]) + f"   {sum(c['preview'])}/4", font=f(18), fill=colr)
            y += 28
        y += 16
        if stage < 4:
            return im
        ch = d["candidates"][d["chosen"]]
        dr.text((24, y), "⑤ 执行", font=f(22, True), fill=(255, 255, 255))
        y += 34
        dr.text((40, y), f"新学生 L3 + 候选 {disp(ch['clip'])}", font=f(20), fill=(220, 220, 220))
        y += 30
        dr.text((40, y), f"test 布局 {disp(d['layout'])}（未见过的试次）", font=f(20), fill=(220, 220, 220))
        return im

    for d in demos:
        frames = []
        hold = lambda st, sec, right=None: [np.concatenate([np.array(panel(d, st)), right if right is not None else  # noqa: E731
                                                            np.full((Hh, W - PW, 3), 12, np.uint8)], 1)] * int(sec * fps)
        frames += hold(1, 2.0) + hold(2, 5.0 if d.get("grey") else 3.0)
        vid = f"{OUT}/exec_{d['id']}/{d.get('layout', '')}.mp4"
        if d.get("candidates"):
            frames += hold(3, 4.0)
            if os.path.exists(vid):
                for fr in imageio.mimread(vid, memtest=False):
                    img = Image.fromarray(fr).resize((W - PW, int(fr.shape[0] * (W - PW) / fr.shape[1])))
                    right = np.full((Hh, W - PW, 3), 12, np.uint8)
                    y0 = (Hh - img.size[1]) // 2
                    right[y0:y0 + img.size[1]] = np.array(img)
                    frames += hold(4, 1.0 / fps, right)
        frames += hold(4 if d.get("candidates") else 2, 2.0, frames[-1][:, PW:] if d.get("candidates") else None)
        imageio.mimsave(f"{OUT}/{d['id']}.mp4", frames, fps=fps, codec="libx264", quality=7, macro_block_size=8)
        print("->", f"{OUT}/{d['id']}.mp4", len(frames), "frames")


if __name__ == "__main__":
    {"prep": prep, "compose": compose}[sys.argv[1]]()
