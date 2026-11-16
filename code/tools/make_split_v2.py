"""Split v2: the trial-level split of data/clip_split_trial/split.json minus the trials excluded by the
data-quality rule (tools/data_quality.py): a stake gap outside 0.5-1.5x the action's median gap = implausible stake
layout (annotation failure: zero-length stance on a spurious stake).  Rule decided from the data only.
Writes split_v2.json (with key "excluded") and clips_train_v2.txt; the original files are not modified."""
import json

H = "/root/gpufree-data/hnu_v2/data/clip_split_trial"
# rule 1: a stake gap outside 0.5-1.5x the action's median gap (implausible layout)
# rule 2 (added ): the reference must continue >= 1 s after the last stake / after leaving the beam (else
#         landing and standing after the obstacle cannot be checked; tools/clip_tail.py, tools/dumu_tail.py)
EXCLUDED = {"ThreeStepPoleRun_022": "rule 1: stake gaps 1.03 / 4.41 m (median 1.48 m); stake 2 annotated with a 0 s stance "
                               "(also rule 2: 0.89 s after the last stake)",
            "ThreeStepPoleRun_025": "rule 1: stake gaps 1.08 / 3.70 m (median 1.48 m); stakes 2 and 3 annotated with 0 s stances",
            "ThreeStepPoleRun_029": "rule 2: the capture ends 0.01 s after the last stake (mid-landing)"}
s = json.load(open(f"{H}/split.json"))
out = {k: [c for c in s[k] if c not in EXCLUDED] for k in ("train", "val", "test")}
out["excluded"] = sorted(EXCLUDED)
out["excluded_reason"] = EXCLUDED
out["protocol"] = s.get("protocol", "") + "; v2 data-quality exclusions (implausible stake layout)"
json.dump(out, open(f"{H}/split_v2.json", "w"), indent=1)
open(f"{H}/clips_train_v2.txt", "w").write("\n".join(out["train"]) + "\n")
print({k: len(v) for k, v in out.items() if isinstance(v, list)})
