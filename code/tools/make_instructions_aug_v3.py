"""Router v3 training data (; deviation: instruction-layer improvement, same for every level).
The v2 gate mostly refused in-scope commands phrased implicitly or with speech-recognition noise.  This adds, for
TRAINING only:
  implicit  : hand-written scene / goal descriptions that do not name the canonical action (written from the action
              definitions, without looking at the sealed test set)
  asr       : programmatic speech-recognition noise on the in-scope training commands (Chinese homophones, English
              typos / phonetic spellings, punctuation dropped)
  reject    : out-of-scope and ambiguous commands (negatives for the gate classifier, never for routing), plus noisy copies
Every generated text with MiniLM cosine > 0.92 to any val2, Results 8.4 benchmark or sealed test command is dropped
automatically, so validation stays independent and the sealed-test exclusion set is unchanged.
  PYTHONPATH=data/data_code python tools/make_instructions_aug_v3.py -> code/data_lang/instructions_aug_v3.json"""
import json
import os
import random
import re
import sys

import numpy as np

H = "/root/gpufree-data/hnu_v2"
sys.path.insert(0, f"{H}/code/stage4_latent")
os.environ["HNU_TEXT_ENCODER"] = "minilm"
from retrieve_skill import hashed_text_vector  # noqa: E402

D = f"{H}/code/data_lang"
ACTS = ["CrouchedTraverse", "ThreeStepPoleRun", "FiveStepPoleRun"]
rnd = random.Random(0)

IMPLICIT = {
    "CrouchedTraverse": [
        "There's a narrow wooden bridge in front of you and I want you on the other side of it, but not on top.",
        "The plank across the path is about chest high; standing up you'd hit it.",
        "That log bridge is blocking the way and the only route is the space beneath it.",
        "You're too tall for the opening under that bridge, so make yourself shorter and go.",
        "The gap under the walkway is low. Get to the far side through it.",
        "Imagine a low tunnel under the beam; come out the other end.",
        "Your head is higher than that crossbar, keep that in mind and get past it from below.",
        "The finish is behind the bridge, and climbing it is not allowed.",
        "Get through the low opening without letting your head touch the wood above.",
        "Treat the bridge like a low doorway.",
        "前面那座独木桥挺矮的，从上面走不行，你自己想办法到对面。",
        "桥面比你的胸口还低，直着身子过不去。",
        "那根横着的木头底下有空间，到桥那边去。",
        "终点在桥的另一边，不能爬上桥。",
        "上面有根木梁，别让头碰到它，走到对面去。",
        "就像过一个很矮的门洞那样过去。",
        "桥底下的空间不高，你得把身子放低才能过。",
        "目标在独木桥后面，从桥下的空当过去。",
        "那块木板横在半空，人站着会撞头，过去吧。",
        "前方横着一根原木，高度只到你肩膀，想办法穿过去。",
        "桥下面能钻人，到对面去。",
        "别从桥上走，桥下那条路才对。",
    ],
    "ThreeStepPoleRun": [
        "Three short wooden posts stand ahead; I want you to get across using only their tops.",
        "There's a short row of three stumps. The floor between them is off limits, and do it at a run.",
        "Three stepping posts in a line, treat the ground as lava and run.",
        "The short stake course has three posts. Run it.",
        "Ahead are three little pillars. Run over them, one landing per pillar.",
        "Only three stakes this time, cross them quickly.",
        "The three-post line is next; keep your feet on the posts all the way.",
        "Your path is three stumps long. Go fast and land on every one.",
        "前面有三个小木桩排成一行，地面不能踩，跑过去。",
        "那排短的桩子只有三根，速度快一点踩过去。",
        "把地面当成岩浆，踩着那三根桩冲过去。",
        "三根木桩一字排开，一步一个跑到头。",
        "这回是三根的那组，每根都要踩到。",
        "前方立着三个木墩子，从墩子上跑过去。",
        "短桩那条线，三个桩，跑着过。",
        "脚只能落在那三根桩顶上，快速通过。",
        "三个桩子挨个踩，别掉到地上。",
        "这一组桩数是三，跑过去别停。",
    ],
    "FiveStepPoleRun": [
        "Five short wooden posts stand ahead; I want you to get across using only their tops.",
        "There's a long row of five stumps. The floor between them is off limits, and do it at a run.",
        "Five stepping posts in a line, treat the ground as lava and run.",
        "The long stake course has five posts. Run it.",
        "Ahead are five little pillars. Run over them, one landing per pillar.",
        "This time it's the five stakes, cross them quickly.",
        "The five-post line is next; keep your feet on the posts all the way.",
        "Your path is five stumps long. Go fast and land on every one.",
        "前面有五个小木桩排成一行，地面不能踩，跑过去。",
        "那排长的桩子有五根，速度快一点踩过去。",
        "把地面当成岩浆，踩着那五根桩冲过去。",
        "五根木桩一字排开，一步一个跑到头。",
        "这回是五根的那组，每根都要踩到。",
        "前方立着五个木墩子，从墩子上跑过去。",
        "长桩那条线，五个桩，跑着过。",
        "脚只能落在那五根桩顶上，快速通过。",
        "五个桩子挨个踩，别掉到地上。",
        "这一组桩数是五，跑过去别停。",
    ],
}
REJECT = [
    # unrelated
    "Bring me the remote control.", "Close the door behind you.", "Stand on one leg.", "Clap your hands twice.",
    "Walk to the charging station.", "Pick up the box on the floor.", "Look to your left.", "Nod your head.",
    "Go upstairs.", "Step over the rope.", "Crawl on the floor.", "Throw the ball to me.", "Stop moving.",
    "帮我倒杯水。", "去把窗帘拉上。", "原地跳三下。", "往后退五步。", "把桌子推到墙边。", "举起双手。", "去门口等我。",
    "向右转。", "蹲下系鞋带。", "跟我打个招呼。", "关掉电视。", "跑五圈。", "数到三再开始。",
    # wrong way to use the same obstacles (out of scope)
    "Walk along the top of the bridge.", "Climb onto the bridge and balance across it.", "Jump over the beam.",
    "Walk around the stakes instead of stepping on them.", "Kick the stakes over.", "Sit on one of the posts.",
    "走上独木桥，从桥面上走过去。", "从横梁上面跳过去。", "绕开那些桩子走。", "把木桩推倒。", "站到桩子上别动。",
    # ambiguous (no count / no action)
    "Run over the stakes.", "Go across the posts quickly.", "Do the stake run.", "Handle the course ahead.",
    "Do it again.", "Get past that.", "Go now.", "Cross the obstacle course.",
    "跑过那排桩。", "从桩上过去。", "把那个障碍过了。", "再来一次。", "开始吧。", "过去那边。", "完成前面的项目。",
]

HOMO = {"桩": "庄装状妆", "桥": "乔侨敲瞧", "根": "跟更", "三": "山叁", "五": "午舞武", "钻": "转赚", "蹲": "吨墩",
        "弯": "湾完", "腰": "要摇", "跑": "泡抛", "踩": "彩采", "下": "夏吓", "低": "底滴", "过": "国果", "横": "衡恒",
        "梁": "凉粮", "木": "目幕", "独": "读毒", "冲": "充虫", "刺": "次此", "面": "免棉", "身": "深申"}
EN = {"three": ["tree", "thre", "free"], "five": ["fiv", "fife", "fyve"], "bridge": ["brige", "bridg", "brij"],
      "under": ["undr", "unda", "onder"], "stakes": ["steaks", "stakez", "stake"], "posts": ["post", "poasts"],
      "across": ["acros", "accross"], "beam": ["beem", "bean"], "below": ["bellow", "belo"], "run": ["ran", "rum"],
      "through": ["thru", "threw"], "squat": ["squad", "swat"], "pillars": ["pillers", "pilars"], "stumps": ["stamps", "stums"]}


def noisy(t):
    if re.search(r"[一-鿿]", t):
        chars = list(t)
        idx = [i for i, c in enumerate(chars) if c in HOMO]
        for i in rnd.sample(idx, min(len(idx), rnd.choice([1, 1, 2]))):
            chars[i] = rnd.choice(HOMO[chars[i]])
        t = "".join(chars)
    words = t.split(" ")
    for i, w in enumerate(words):
        k = re.sub(r"[^a-z]", "", w.lower())
        if k in EN and rnd.random() < 0.6:
            words[i] = rnd.choice(EN[k])
        elif len(k) > 4 and rnd.random() < 0.12:
            j = rnd.randrange(1, len(k) - 1)
            words[i] = k[:j] + k[j + 1:] if rnd.random() < 0.5 else k[:j] + k[j + 1] + k[j] + k[j + 2:]
    t = " ".join(words)
    if rnd.random() < 0.7:
        t = re.sub(r"[，。,.;；!！?？—]", " " if rnd.random() < 0.5 else "", t).strip()
    return re.sub(r"\s+", " ", t)


unit = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True)  # noqa: E731
sv = lambda ts: unit(np.stack([hashed_text_vector(t, 384) for t in ts]))  # noqa: E731


def main():  # (noisy() is reused by tools/filter_lang_v3.py; output unchanged by this wrapping)
    ins = json.load(open(f"{D}/instructions_v1.json", encoding="utf-8"))
    v2 = json.load(open(f"{D}/instructions_val_v2.json", encoding="utf-8"))
    bench = json.load(open(f"{H}/code/results/router_ood_benchmark_v1.json", encoding="utf-8"))["cases"]
    sealed = json.load(open(f"{D}/router_test_v2_sealed.json", encoding="utf-8"))["cases"]
    GUARD = sv([t for a in ACTS for t in v2["val"][a]] + v2["val_reject"] + [c["query"] for c in bench] + [c["query"] for c in sealed])
    stats = {}

    def keep(ts, tag):
        ts = list(dict.fromkeys(t for t in ts if t.strip()))
        if not ts:
            return []
        E = sv(ts)
        ok = (E @ GUARD.T).max(1) <= 0.92
        stats[tag] = (int(ok.sum()), len(ts))
        return [t for t, o in zip(ts, ok) if o]

    out = {"format": "router_instruction_aug_v3", "purpose": __doc__.strip().splitlines()[0], "train": {}, "reject": []}
    for a in ACTS:
        imp = keep(IMPLICIT[a], f"implicit {a}")
        base = ins["train"][a] + imp
        asr = keep([noisy(t) for t in base for _ in range(2)], f"asr {a}")
        out["train"][a] = imp + asr
    rej = keep(REJECT, "reject")
    out["reject"] = rej + keep([noisy(t) for t in rej], "reject asr")
    out["stats_kept_of_generated"] = stats
    json.dump(out, open(f"{D}/instructions_aug_v3.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(stats, ensure_ascii=False))
    for a in ACTS:
        print(a, len(out["train"][a]), out["train"][a][len(IMPLICIT[a]):][:6])
    print("reject", len(out["reject"]), out["reject"][-5:])


if __name__ == "__main__":
    main()
