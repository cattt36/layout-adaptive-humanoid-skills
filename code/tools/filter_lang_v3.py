"""Filter the LLM-written router data of (user decision: validation set v3 for the gate, LLM-generated):
  data_lang/instructions_val_v3_raw.json  -> instructions_val_v3.json   (validation only)
  data_lang/instructions_aug_v3b_raw.json -> instructions_aug_v3b.json  (gate training negatives + noisy copies)
Near-duplicates (MiniLM cosine > 0.92) are dropped: val3 against the sealed test set, val2, the Results 8.4 benchmark and
every training text (instructions_v1, instructions_aug_v3, aug_v3b); aug_v3b against the sealed test set, val2, the
benchmark and the filtered val3.  Duplicates inside a file are dropped too.  Kept / written counts are printed.
  PYTHONPATH=data/data_code python tools/filter_lang_v3.py"""
import json
import os
import sys

import numpy as np

H = "/root/gpufree-data/hnu_v2"
sys.path.insert(0, f"{H}/code/tools")
import make_instructions_aug_v3 as m3  # noqa: E402  (noisy(), sv(); its main() is not run)

D = f"{H}/code/data_lang"
ACTS = m3.ACTS
ins = json.load(open(f"{D}/instructions_v1.json", encoding="utf-8"))
v2 = json.load(open(f"{D}/instructions_val_v2.json", encoding="utf-8"))
a3 = json.load(open(f"{D}/instructions_aug_v3.json", encoding="utf-8"))
bench = json.load(open(f"{H}/code/results/router_ood_benchmark_v1.json", encoding="utf-8"))["cases"]
sealed = json.load(open(f"{D}/router_test_v2_sealed.json", encoding="utf-8"))["cases"]
vraw = json.load(open(f"{D}/instructions_val_v3_raw.json", encoding="utf-8"))
araw = json.load(open(f"{D}/instructions_aug_v3b_raw.json", encoding="utf-8"))
sv = m3.sv
GUARD = sv([t for a in ACTS for t in v2["val"][a]] + v2["val_reject"] + [c["query"] for c in bench] + [c["query"] for c in sealed])
TRAIN = [t for p in ("train", "val") for a in ACTS for t in ins[p][a]] + ins["val_reject"] \
    + [t for a in ACTS for t in a3["train"][a]] + a3["reject"]
neg_raw = [(cat, t) for cat, ts in araw["reject"].items() for t in ts]
stats = {}


def keep(items, refs, tag):
    """items: list of (label, text); refs: list of matrices"""
    seen, out = set(), []
    for lab, t in items:
        if t.strip() and t not in seen:
            seen.add(t)
            out.append((lab, t))
    if not out:
        return []
    E = sv([t for _, t in out])
    ok = np.ones(len(out), bool)
    for R in refs:
        ok &= (E @ R.T).max(1) <= 0.92
    stats[tag] = (int(ok.sum()), len(out))
    return [x for x, o in zip(out, ok) if o]


R_train = sv(TRAIN + [t for _, t in neg_raw])
val = {a: [t for _, t in keep([(a, t) for t in vraw["val"][a]], [GUARD, R_train], f"val3 {a}")] for a in ACTS}
rej = keep([(cat, t) for cat, ts in vraw["val_reject"].items() for t in ts], [GUARD, R_train], "val3 reject")
json.dump({"format": "router_instruction_val_v3", "purpose": vraw["purpose"], "val": val,
           "val_reject": [t for _, t in rej], "val_reject_category": [c for c, _ in rej], "stats_kept_of_written": stats},
          open(f"{D}/instructions_val_v3.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
R_val3 = sv([t for a in ACTS for t in val[a]] + [t for _, t in rej])
neg = keep(neg_raw, [GUARD, R_val3], "aug3b reject")
noisy = keep([(c, m3.noisy(t)) for c, t in neg], [GUARD, R_val3], "aug3b reject asr")
json.dump({"format": "router_instruction_aug_v3b", "purpose": araw["purpose"], "reject": [t for _, t in neg + noisy],
           "reject_category": [c for c, _ in neg] + [c + "_asr" for c, _ in noisy], "stats_kept_of_written": stats},
          open(f"{D}/instructions_aug_v3b.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(stats, ensure_ascii=False))
print("noisy examples:", [t for _, t in noisy[:8]])
