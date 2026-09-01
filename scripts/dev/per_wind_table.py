"""Per-wind DEL reduction at the best evaluation (by F) of each run, from decisions.jsonl."""
import glob
import json
import os
import sys

rows = []
for pat in sys.argv[1:]:
    for d in sorted(glob.glob(os.path.expanduser(pat))):
        p = os.path.join(d, "decisions.jsonl")
        if not os.path.exists(p):
            continue
        recs = [json.loads(l) for l in open(p, encoding="utf-8")]
        recs = [r for r in recs if "per_episode" in r]
        if not recs:
            continue
        best = max(recs, key=lambda r: r["fit"]["F"])
        last = recs[-1]
        def fmt(r):
            pe = {f"U{e['mean_wind']:g}": e["del_red_pct"] for e in r["per_episode"]}
            spd = {f"U{e['mean_wind']:g}": e["gen_speed_std_R3"] for e in r["per_episode"]}
            return pe, spd
        pe, spd = fmt(best)
        rows.append((os.path.basename(d), best["episode"], best["fit"]["F"], pe, best["fit"]["energy_loss_pct"],
                     best["fit"]["speed_std_ratio"]))
print(f"{'run':>24} {'ep':>4} {'F':>7} | DEL reduction % at U8 / U12.5 / U15 | Eloss% | spd ratio")
for name, ep, F, pe, el, sr in rows:
    print(f"{name:>24} {ep:4d} {F:7.2f} | {pe.get('U8', float('nan')):7.2f} {pe.get('U12.5', float('nan')):7.2f} "
          f"{pe.get('U15', float('nan')):7.2f} | {el:5.2f} | {sr:.3f}")
