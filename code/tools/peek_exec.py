import json, sys
d = json.load(open(sys.argv[1]))
for s in d["segments"][:12]:
    rm = s["Retargeted_Robot_Metrics"]
    print(s["segment_id"], s.get("gait_phase"), rm.get("data_availability", {}).get("status"), json.dumps(rm.get("execution"))[:260])
