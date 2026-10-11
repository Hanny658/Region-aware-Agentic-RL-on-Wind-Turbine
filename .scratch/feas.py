import json, glob, os
import numpy as np
rows = []
for h in sorted(glob.glob(os.path.expanduser("~/wtrl/exp/roscosearch/*/history.jsonl"))):
    for line in open(h):
        r = json.loads(line)["result"]
        rows.append(r)
e = np.array([r.get("energy_loss_pct", np.nan) for r in rows], float)
tw = np.array([r.get("tower", np.nan) for r in rows], float)
bl = np.array([r.get("blade", np.nan) for r in rows], float)
sp = np.array([r.get("speed", np.nan) for r in rows], float)
ok = np.isfinite(e) & np.isfinite(tw)
print(f"{len(rows)} candidates, {ok.sum()} with usable numbers")
print(f"energy_loss_pct: min {np.nanmin(e):+.3f}  median {np.nanmedian(e):+.3f}  max {np.nanmax(e):+.3f}"
      f"   (positive = energy LOST)")
print(f"  candidates that lose any energy at all: {(e > 0).sum()} of {ok.sum()}")
print(f"  candidates that lose more than 0.1 %:   {(e > 0.1).sum()}")
feas = (tw >= 0) & (bl >= 0) & (e <= 0)
print(f"\nfeasible under 'no load or energy term worse than shipped': {feas.sum()} of {ok.sum()} "
      f"({100*feas.sum()/ok.sum():.0f} %)")
print(f"  of those, speed-MSE reduction: min {sp[feas].min():+.1f}  median {np.median(sp[feas]):+.1f}  max {sp[feas].max():+.1f}")
print(f"  best speed reduction among feasible {sp[feas].max():+.1f} vs among all {np.nanmax(sp):+.1f}")
# per arm
print("\nper arm (the finished 16-candidate runs):")
for h in sorted(glob.glob(os.path.expanduser("~/wtrl/exp/roscosearch/*/history.jsonl"))):
    rs = [json.loads(l)["result"] for l in open(h)][1:]
    f = [r for r in rs if r.get("tower", -9) >= 0 and r.get("blade", -9) >= 0 and r.get("energy_loss_pct", 9) <= 0]
    nm = os.path.basename(os.path.dirname(h))
    best = max((r["speed"] for r in f), default=float("nan"))
    print(f"  {nm:<10} {len(f):2d}/{len(rs):2d} feasible, best feasible speed reduction {best:+6.1f}")
