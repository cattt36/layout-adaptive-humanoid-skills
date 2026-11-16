"""Stake annotations with empty / very short stance windows (json fps and 50 Hz), and which foot / where."""
import json

H = "/root/gpufree-data/hnu_v2/data"
for d, clips in ((f"{H}/ref_stakes_s93_h10/ThreeStepPoleRun", ["022", "025", "028", "001"]),
                 (f"{H}/ref_stakes_s83_h10/FiveStepPoleRun", ["026", "001"])):
    for n in clips:
        t = d.split("/")[-1]
        j = json.load(open(f"{d}/{t}_{n}_stakes.json"))
        print(f"{t}_{n}: fps {j['fps']}, ramp {j.get('ramp')}")
        for i, s in enumerate(j["stakes"]):
            a, b = s["stance"]
            print(f"   stake {i + 1}: foot {s['foot']} xy ({s['x']:.2f}, {s['y']:.2f}) stance {a}-{b} "
                  f"({(b - a) / j['fps']:.3f} s; 50 Hz frames {int(a * 50 / j['fps'])}-{int(b * 50 / j['fps'])})")
