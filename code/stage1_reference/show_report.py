import json, sys
r = json.load(open(sys.argv[1]))
keys = sys.argv[2:] or ["stances"]
for k in keys:
    v = r[k]
    if isinstance(v, list):
        for x in v:
            print(x)
    else:
        print(k, v)
