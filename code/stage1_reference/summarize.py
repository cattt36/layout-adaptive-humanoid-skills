"""Compact comparison table of stage-1 reports: python summarize.py dirA [dirB ...]"""
import glob, json, os, sys

KEYS = [
    ("plate_offset_s", "plOff"), ("plate_match", "plMat"), ("stance_slip_max_mm", "slip"), ("sole_penetration_max_mm", "pen"), ("torque_ratio_max", "tauX"),
    ("torque_over_frames_pct", "tau>1%"), ("vel_ratio_max", "velX"), ("soft_limit_frames_pct", "soft%"),
    ("root_lin_acc_p99", "acc99"), ("residual_stance_p95", "resS95"), ("residual_flight_p95", "resF95"),
    ("peak_grf_over_weight", "GRF/W"), ("self_penetration_max_mm", "selfpen"), ("root_speed_max", "vmax"),
]
print("dir/trial".ljust(34) + "".join(h.rjust(8) for _, h in KEYS))
for d in sys.argv[1:]:
    for f in sorted(glob.glob(os.path.join(d, "*_report.json"))):
        r = json.load(open(f))
        tag = os.path.basename(os.path.normpath(d))[-8:] + "/" + r["trial"]
        print(tag.ljust(34) + "".join(str(r.get(k, "-")).rjust(8) for k, _ in KEYS))
