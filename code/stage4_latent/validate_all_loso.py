#!/usr/bin/env python3
"""Validate every action/profile/fold produced by train_all_loso.ps1."""
from __future__ import annotations
import argparse, json, subprocess, sys, os
from pathlib import Path

ACTIONS = ("CrouchedTraverse", "ThreeStepPoleRun", "FiveStepPoleRun")
PROFILES = ("Z0", "Z1", "Z2", "Z3", "Z4")

def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument("--latent-root", type=Path, default=Path(r"C:\Users\Administrator\Desktop\Data-HNU\Latent\latent_15_fair_ontology_v3"))
    ap.add_argument("--json-root", type=Path, default=Path(r"C:\Users\Administrator\Desktop\Data-HNU\LLM_annotation"))
    ap.add_argument("--python", type=Path, default=Path(r"C:\Users\Administrator\Desktop\Data-HNU\.venv\Scripts\python.exe"))
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--device", choices=("cpu","cuda"), default="cpu")
    args=ap.parse_args()
    validator=Path(__file__).with_name("validate_latents.py")
    # validate_latents imports shared schema from the repository Data package.
    # Pass the repository root on PYTHONPATH for subprocesses launched from any cwd.
    env=dict(os.environ)
    repo_root=Path(__file__).resolve().parents[1]
    env["PYTHONPATH"]=str(repo_root / "Data") + os.pathsep + str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")
    rows=[]; failures=[]
    for action in ACTIONS:
        action_root=args.latent_root/action
        if not action_root.is_dir(): failures.append(f"missing action: {action_root}"); continue
        for fold in sorted(p for p in action_root.iterdir() if p.is_dir() and p.name.startswith("subject_")):
            for profile in PROFILES:
                latent_dir=fold/profile
                required_outputs=("best_motion_latent.pt","train_latents.npz","val_latents.npz","test_latents.npz")
                if not latent_dir.is_dir(): failures.append(f"missing latent: {latent_dir}"); continue
                if any(not (latent_dir/name).is_file() for name in required_outputs):
                    failures.append({"action":action,"fold":fold.name,"profile":profile,"error":"incomplete_training_outputs"})
                    continue
                report=latent_dir/("latent_validation_report_" + profile + ".json")
                cmd=[str(args.python),str(validator),"--latent-dir",str(latent_dir.resolve()),"--profile",profile,"--json-dir",str((args.json_root/action).resolve()),"--report",str(report.resolve())]
                proc=subprocess.run(cmd,capture_output=True,text=True,env=env)
                if proc.returncode:
                    failures.append({"action":action,"fold":fold.name,"profile":profile,"returncode":proc.returncode,"stderr":proc.stderr[-2000:]})
                    continue
                try: data=json.loads(report.read_text(encoding="utf-8-sig"))
                except Exception as exc:
                    failures.append({"action":action,"fold":fold.name,"profile":profile,"error":f"bad report: {exc}"}); continue
                row={"action":action,"fold":fold.name,"profile":profile,"latent_dir":str(latent_dir),"splits":{},"probes":data.get("probes",{}),"command_interface":data.get("command_interface",{})}
                for split, info in data.get("splits",{}).items():
                    row["splits"][split]={"rows":info.get("rows"),"latent_health":info.get("latents",{}),"alignment":info.get("alignment"),"decorrelation":info.get("decorrelation"),"shift_from_train":info.get("shift_from_train")}
                rows.append(row)
    out=args.output or args.latent_root.parent/"loso_validation_summary.json"
    out.write_text(json.dumps({"format":"loso_validation_summary_v1","actions":ACTIONS,"profiles":PROFILES,"results":rows,"failures":failures},indent=2,ensure_ascii=False,allow_nan=True),encoding="utf-8")
    print(json.dumps({"validated":len(rows),"failures":len(failures),"output":str(out)},ensure_ascii=False))
    if failures:
        print(json.dumps(failures[:10],indent=2,ensure_ascii=False),file=sys.stderr)
        raise SystemExit(1)
if __name__ == "__main__": main()






