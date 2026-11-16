"""Print the latest text entries of a monitor run (check of tools/monitor_tb.py)."""
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ea = EventAccumulator(sys.argv[1], size_guidance={"tensors": 0})
ea.Reload()
for tag in ea.Tags()["tensors"]:
    ev = ea.Tensors(tag)[-1]
    print(f"### {tag} (step {ev.step})")
    print(ev.tensor_proto.string_val[0].decode())
print("scalars:", ea.Tags()["scalars"])
