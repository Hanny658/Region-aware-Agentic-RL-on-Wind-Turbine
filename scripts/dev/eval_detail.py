"""All evaluation points of one or more runs with per-wind detail (from decisions.jsonl)."""
import json
import os
import sys

for r in sys.argv[1:]:
    d = os.path.expanduser(r if "/" in r else f"~/wtrl/exp/{r}")
    print(f"== {os.path.basename(d)}")
    s = os.path.join(d, "summary.json")
    if os.path.exists(s):
        js = json.load(open(s))
        print(f"   best_F {js.get('best_F', float('nan')):.2f} @ep {js.get('best_episode')}  final knobs "
              f"{ {k: round(v, 4) for k, v in js.get('final_knobs', {}).items()} }")
    for ln in open(os.path.join(d, "decisions.jsonl"), encoding="utf-8"):
        rec = json.loads(ln)
        pe = rec.get("per_episode", [])
        parts = []
        for e in pe:
            spd = e["gen_speed_std_R3"]
            spd = f"{spd:.3f}" if spd == spd else "  -  "
            parts.append(f"U{e['mean_wind']:g}: DEL{e['del_red_pct']:+5.1f} spd {spd} |db| {e['dbeta_abs_mean_deg']:.2f} "
                         f"trav {e['pitch_travel_deg']:.0f}/{e['pitch_travel_base_deg']:.0f} "
                         f"E {100 * (1 - e['energy_MWh'] / e['energy_base_MWh']):+.2f}%")
        rb = " ROLLBACK" if "rollback" in rec else ""
        f = rec["fit"]
        ex = ""
        if "TwrBsMyt_DEL_red_pct" in f:
            ex = f" [outb: blade {f['RootMyc1_DEL_red_pct']:+.1f} tower {f['TwrBsMyt_DEL_red_pct']:+.1f}] Eloss {f['energy_loss_pct']:+.2f}"
        print(f"  ep {rec['episode']:4d} F {f['F']:7.2f}{rb}{ex} | " + " | ".join(parts))
