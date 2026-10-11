"""What is the resolution of each constraint on the 14-episode design set? Measured, not chosen."""
import json, glob, os
import numpy as np
fs = sorted(glob.glob(os.path.expanduser("~/wtrl/exp/roscosearch/cache/eval_*.json")))
print(len(fs), "cached evaluations")
d = json.load(open(fs[0]))
pe = d.get("per_episode", [])
print("per_episode keys:", sorted(pe[0]) if pe else "none")
