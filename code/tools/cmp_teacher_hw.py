"""Teacher bank: L40 (runs/results_*) vs 4090 (runs/ref4090/results_*), success v2 / success / body error per action."""
import json

H = "/root/gpufree-data/hnu_v2/runs"
print(f"{'criterion':5s} {'cond':16s} {'action':12s} {'succ_v2 4090':>12s} {'L40':>6s} {'succ 4090':>9s} {'L40':>6s} {'body_err cm 4090':>16s} {'L40':>6s}")
for crit, v in (("task", "v6g10n"), ("track", "v6g10ntrk")):
    for dr in ("0", "1"):
        for x, cn in (("", ""), ("_push", "+push")):
            f = f"results_{v}_bank_teacher_dr{dr}/eval_teacher_teacher{x}.json"
            a, b = json.load(open(f"{H}/ref4090/{f}"))["summary"], json.load(open(f"{H}/{f}"))["summary"]
            for T in ("ThreeStepPoleRun", "FiveStepPoleRun", "CrouchedTraverse"):
                k = f"{T}/all"
                ga = lambda d, m: d[k].get(m, float("nan"))  # noqa: E731
                print(f"{crit:5s} {('nominal' if dr == '0' else 'randomised') + cn:16s} {T:12s} {100 * ga(a, 'success_v2'):12.1f} {100 * ga(b, 'success_v2'):6.1f}"
                      f" {100 * ga(a, 'success'):9.1f} {100 * ga(b, 'success'):6.1f} {100 * ga(a, 'body_err'):16.2f} {100 * ga(b, 'body_err'):6.2f}")
