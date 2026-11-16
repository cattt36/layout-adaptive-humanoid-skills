"""Print sim2sim results: python deploy/s2s_report.py <result.json> ... [--rows]"""
import json
import sys

rows = "--rows" in sys.argv
for f in [a for a in sys.argv[1:] if a != "--rows"]:
    j = json.load(open(f))
    print(f"== {f}  {j['opt']}  physics dt {j['physics_dt']}")
    for t, s in j["summary"].items():
        le = [r["loc_err_max"] for r in j["rows"] if r["task"] == t and r.get("loc_err_max")]
        les = f"  max localisation err {100 * sum(le) / len(le):.1f} cm (mean over episodes)" if le else ""
        print(f"  {t:12s} success {100 * s['success']:5.1f}%  body err {100 * s['body_err']:.1f} cm  n={s['n']}  "
              f"{s['causes']}{les}")
    if rows:
        for r in j["rows"]:
            print(f"    {r['clip']:18s} trial {r['trial']} steps {r['steps']}/{r['T'] - 1} success {r['success']} "
                  f"cause {r['cause']} err {100 * r['body_err']:.1f} cm")
