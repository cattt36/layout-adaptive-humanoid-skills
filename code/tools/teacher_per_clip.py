"""Per-clip teacher success (criterion v2, task) for the fold-1 test clips, 4 conditions -> json (clip selection for the
real-robot experiment: chosen from the teacher only, never from student results)."""
import json
import sys

H = "/root/gpufree-data/hnu_v2/runs"
out = {}
for dr in ("0", "1"):
    for x in ("", "_push"):
        rows = json.load(open(f"{H}/results_v6g10n_bank_teacher_dr{dr}/eval_teacher_teacher{x}.json"))
        rows = rows.get("clips") or rows.get("rows") or rows.get("per_clip")
        for r in rows:
            out.setdefault(r["clip"], {})[f"dr{dr}{x}"] = dict(success_v2=r.get("success_v2"), body_err=r.get("body_err"), split=r.get("split"))
json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else "/tmp/teacher_per_clip.json", "w"), indent=1)
for c in sorted(out):
    if out[c]["dr0"]["split"] == "test":
        print(c, " ".join(f"{k}:{100 * (v['success_v2'] or 0):.0f}" for k, v in out[c].items()))
