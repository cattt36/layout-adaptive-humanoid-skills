"""Score a bridge-teacher gate directory (runs/du_gate_<tag>) against the pre-decided standard (see du_ft_auto.sh).
Fixes the file name of the push run (eval_student.py appends _push: eval_teacher_push_push.json).
usage: du_gate_score.py <tag> [<tag> ...]"""
import json
import os
import sys

H = "/root/gpufree-data/hnu_v2/runs"
ref_err = 100 * json.load(open(f"{H}/accept_prism/dr0/eval_teacher_new.json"))["summary"]["CrouchedTraverse/all"]["body_err"]
for T in sys.argv[1:]:
    O = f"{H}/du_gate_{T}"
    name = lambda n: f"{O}/eval_teacher_{n}.json" if os.path.exists(f"{O}/eval_teacher_{n}.json") else f"{O}/eval_teacher_{n}_push.json"  # noqa: E731
    s = lambda n: json.load(open(name(n)))["summary"]["CrouchedTraverse/all"]  # noqa: E731
    v = {n: 100 * s(n)["success"] for n in ("nom", "rand", "push", "dzm5", "dzm3", "dzm8", "dzp8", "spm10", "spp10", "pj10")}
    err = 100 * s("nom")["body_err"]
    ret = json.load(open(f"{O}/retention.json"))
    checks = {"beam -5 cm >= 90": v["dzm5"] >= 90, "beam -3 cm >= 92": v["dzm3"] >= 92, "beam +8 cm >= 92": v["dzp8"] >= 92,
              "posts +-10 cm >= 92": min(v["spm10"], v["spp10"], v["pj10"]) >= 92, "nominal >= 92": v["nom"] >= 92,
              "randomised >= 90": v["rand"] >= 90, "retention": bool(ret.get("retained")),
              f"body err <= {1.1 * ref_err:.1f} cm": err <= 1.1 * ref_err}
    ok = all(checks.values())
    print(("PASS" if ok else "FAIL") + f" {T} score {min(v['dzm5'], v['nom']):.1f} | " + " ".join(f"{k} {x:.0f}" for k, x in v.items())
          + f" | body err {err:.1f} cm | retention {ret.get('original')}->{ret.get('finetuned')}")
    failed = [k for k, x in checks.items() if not x]
    if failed:
        print("   failed:", ", ".join(failed))
