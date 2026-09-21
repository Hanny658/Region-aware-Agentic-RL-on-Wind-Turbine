"""Print the MPC keyword JSON (for evaluate.py --base_mpc_json) of a best.json written by scripts/mpc_design_search.py."""
import json
import sys

print(json.dumps(json.load(open(sys.argv[1]))["mpc_kw"]))
