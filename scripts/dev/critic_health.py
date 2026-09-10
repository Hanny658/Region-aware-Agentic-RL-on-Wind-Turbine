"""Is a trained critic still a function of the state, or has it collapsed to a constant?

The networks are tanh MLPs (`agents/ppo.py: mlp()` = Linear-Tanh-Linear-Tanh-Linear), so a literal
dead-ReLU cannot occur. The equivalent pathology is **tanh saturation**: units pinned at |tanh| ~ 1
receive no gradient and stop carrying state information; if enough of them saturate the value head
degenerates to a constant and GAE advantages become pure reward noise.

Reported per critic:
  V spread        std / min / max of V(s) over a realistic batch. A collapsed critic has std ~ 0.
  saturated       fraction of hidden units with |tanh| > 0.99 on > 99 % of the batch (no gradient).
  frozen          fraction of hidden units whose activation std over the batch is < 1e-3
                  (constant output = the direct analogue of a dead ReLU).
  |h| mean        mean absolute activation per layer; ~1.0 means the layer lives in saturation.
  corr(V, -|dw|)  R3 sanity check: the R3 reward is w_speed * exp(-|dw|/tau), so a healthy R3
                  value function should rise as the speed error shrinks.

States come from the paired GSPI baseline logs, rebuilt into the exact observation vector the agent
sees (`envs/base_env.py: _obs`) and normalised with the checkpoint's own running statistics. That is
the zero-cost option and covers the right region of state space (U15 is ~100 % R3, U8 ~100 % R2);
it is the GSPI state distribution rather than the policy's own, which is noted where it matters.

    python scripts/dev/critic_health.py                                  # main arms, U15 + U8
    python scripts/dev/critic_health.py --runs "~/wtrl/exp/n1_llmfork_s*"
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import torch
import yaml

from agents.ppo import Critic
from controllers.router import R2, R3
from envs.base_env import PROJ
from envs.factory import WIND_DIR, baseline_dir


def build_obs(npz_path: str, cfg_run: dict, tb: dict) -> tuple[np.ndarray, np.ndarray]:
    """Rebuild the agent's observation vector from a baseline log. Returns (obs, region)."""
    d = np.load(npz_path)
    dt = float(tb["dt_ctrl_s"])
    wg_rated = float(tb["rated_gen_speed_rads"])
    d_wg = (d["gen_speed"] - wg_rated) / wg_rated
    d_wg_dot = np.concatenate([[0.0], np.diff(d_wg) / dt])
    m_scale = float(cfg_run["reward"].get("M_ref_nm", 8.0e6))
    cols = [d_wg, d_wg_dot / 0.05, d["beta_meas"], d["v_hub"] / 25.0, d["M_oop"] / m_scale]
    if cfg_run.get("obs_fa_acc", False):
        cols.append(d["fa_acc"] if "fa_acc" in d.files else np.zeros_like(d_wg))
    obs = np.stack(cols, 1).astype(np.float32)
    act = d["warmup"] == 0
    return obs[act], d["region"][act]


def health(critic: Critic, obs_n: torch.Tensor) -> dict:
    acts = []
    hooks = [m.register_forward_hook(lambda _m, _i, o: acts.append(o.detach()))
             for m in critic.v if isinstance(m, torch.nn.Tanh)]
    with torch.no_grad():
        v = critic(obs_n)
    for h in hooks:
        h.remove()
    out = {"V_std": float(v.std()), "V_mean": float(v.mean()),
           "V_min": float(v.min()), "V_max": float(v.max()), "layers": []}
    for a in acts:
        sat = (a.abs() > 0.99).float().mean(0)          # per unit: fraction of batch saturated
        std = a.std(0)
        out["layers"].append({
            "n_units": a.shape[1],
            "saturated": float((sat > 0.99).float().mean()),
            "frozen": float((std < 1e-3).float().mean()),
            "abs_mean": float(a.abs().mean()),
            "unit_std_median": float(std.median()),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["~/wtrl/exp/tq_off_s*", "~/wtrl/exp/n1_llmfork_s*",
                                                  "~/wtrl/exp/n1_randfork_s*", "~/wtrl/exp/n1_mono_s*"])
    ap.add_argument("--winds", nargs="+", default=["U15_TI8_S3", "U8_TI8_S3"])
    ap.add_argument("--ckpt", default="ckpt_best.pt")
    args = ap.parse_args()
    tb = yaml.safe_load(open(PROJ / "configs" / "turbine" / "nrel5mw.yaml"))
    base = Path(os.path.expanduser(baseline_dir("openfast")))

    print(f"critic health — networks are tanh MLPs (no ReLU); the pathology checked is saturation\n"
          f"states rebuilt from the paired GSPI logs in {base}\n")
    for pat in args.runs:
        for run in sorted(glob.glob(os.path.expanduser(pat))):
            ck = Path(run) / args.ckpt
            if not ck.exists():
                continue
            cfg_run = json.load(open(Path(run) / "config.json", encoding="utf-8"))
            st = torch.load(ck, weights_only=False)
            st = st["state"] if "state" in st else st
            for wind in args.winds:
                f = base / f"{wind}.npz"
                if not f.exists():
                    continue
                obs, region = build_obs(str(f), cfg_run, tb)
                for reg, tag in ((R3, "R3"), (R2, "R2")):
                    sub = st.get(reg) if isinstance(st, dict) else None
                    if not sub or "critic" not in sub:
                        continue
                    sel = region == reg
                    if sel.sum() < 500:                 # not enough states of this region here
                        continue
                    hid = (sub["critic"]["v.0.weight"].shape[0], sub["critic"]["v.2.weight"].shape[0])
                    c = Critic(obs.shape[1], hid)
                    c.load_state_dict(sub["critic"])
                    c.eval()
                    rms = sub["obs_rms"]
                    x = (obs[sel] - rms["mean"]) / np.sqrt(rms["var"] + 1e-8)
                    h = health(c, torch.as_tensor(x, dtype=torch.float32))
                    dwg = obs[sel][:, 0]
                    with torch.no_grad():
                        v = c(torch.as_tensor(x, dtype=torch.float32)).numpy()
                    corr = float(np.corrcoef(v, -np.abs(dwg))[0, 1]) if v.std() > 1e-9 else float("nan")
                    L = h["layers"]
                    print(f"{Path(run).name:>18} {wind:>12} {tag} n={int(sel.sum()):6d} | "
                          f"V {h['V_mean']:9.1f} ± {h['V_std']:7.2f}  [{h['V_min']:8.1f},{h['V_max']:8.1f}] | "
                          + " | ".join(f"L{i+1} sat {l['saturated']*100:4.0f}% frozen {l['frozen']*100:4.0f}% "
                                       f"|h| {l['abs_mean']:.2f}" for i, l in enumerate(L))
                          + (f" | corr(V,-|dw|) {corr:+.2f}" if tag == "R3" else ""))
    print("\nread: V_std ~ 0 or saturated/frozen near 100 % would mean a collapsed value function.")


if __name__ == "__main__":
    main()
