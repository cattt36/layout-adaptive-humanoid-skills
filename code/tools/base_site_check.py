"""Which modules does the student pipeline load from the BASE environment (venv has include-system-site-packages)?
Imports what distill_train.py / eval_student.py / the task package import, then lists modules from the base site-packages."""
import importlib
import sys

sys.argv = ["x"]
for m in ("torch", "numpy", "mujoco", "warp", "mujoco_warp", "mjlab", "rsl_rl", "tensorboard", "torch.utils.tensorboard",
          "scipy", "yaml", "hnu_tasks", "hnu_tasks.distill", "hnu_tasks.multi_motion", "hnu_tasks.perception", "tyro", "gymnasium"):
    try:
        importlib.import_module(m)
    except Exception as e:  # noqa: BLE001
        print("import failed:", m, type(e).__name__, str(e)[:100])
import runpy  # noqa: E402
for script in ("distill_train.py", "eval_student.py"):
    src = open(script).read()
    head = src.split("\ndef ")[0]  # module-level imports only
    try:
        exec(compile(head, script, "exec"), {"__name__": "_probe"})
    except SystemExit:
        pass
    except Exception as e:  # noqa: BLE001
        print("module-level of", script, "failed:", type(e).__name__, str(e)[:120])
base = "/opt/conda/envs/isaaclab/lib/python3.11/site-packages"
hits = sorted({n.split(".")[0] for n, mod in list(sys.modules.items()) if getattr(mod, "__file__", None) and mod.__file__.startswith(base)})
print("modules from base site-packages:", hits)
print("n modules:", len(sys.modules))
