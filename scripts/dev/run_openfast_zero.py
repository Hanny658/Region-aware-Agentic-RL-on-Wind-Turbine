"""Zero-residual rollout of the OpenFAST env: validates the ZMQ coupling, timing, region rule
and reward magnitudes, and cross-checks ZMQ measurements against the .outb channels."""
import argparse
import os
import time

import numpy as np

from envs.base_env import EpisodeSpec, default_config
from envs.openfast_env import OpenFASTEnv

ap = argparse.ArgumentParser()
ap.add_argument("--wind_dir", default="~/wtrl/wind")
ap.add_argument("--template", default="~/wtrl/runs/template_5mw")
ap.add_argument("--work", default="~/wtrl/runs/work_dev")
ap.add_argument("--episode_s", type=float, default=60.0)
ap.add_argument("--warmup_s", type=float, default=20.0)
ap.add_argument("--means", nargs="+", type=float, default=[8, 12.5, 15])
ap.add_argument("--port", type=int, default=5601)
ap.add_argument("--save", default=None)
args = ap.parse_args()

wd = os.path.expanduser(args.wind_dir)
eps = [EpisodeSpec(wind_file=f"{wd}/U{u:g}_TI8_S1.bts", mean_wind=u, episode_s=args.episode_s,
                   warmup_s=args.warmup_s) for u in args.means]
env = OpenFASTEnv(default_config(), eps, args.template, args.work, port=args.port, keep_outputs=True)
for i, ep in enumerate(eps):
    t0 = time.time()
    obs, _ = env.reset(options={"episode_index": i})
    done, n = False, 0
    while not done:
        obs, r, term, trunc, info = env.step(np.zeros(1, dtype=np.float32))
        done = term or trunc
        n += 1
    L = env.log_arrays()
    act = L["warmup"] == 0
    wall = time.time() - t0
    print(f"\n== U={ep.mean_wind:g} m/s  agent steps={n}  logged={len(L['t'])}  wall={wall:.1f}s  ({ep.episode_s / wall:.1f}x RT)")
    print(f"   t range {L['t'][0]:.2f}..{L['t'][-1]:.2f}   dt median {np.median(np.diff(L['t'])):.4f}")
    print(f"   v_hub  mean {L['v_hub'][act].mean():6.2f}  v_est mean {L['v_est'][act].mean():6.2f}")
    print(f"   gen_spd mean {L['gen_speed'][act].mean():6.1f} rad/s (rated 122.9)  std {L['gen_speed'][act].std():5.2f}")
    print(f"   P mean {L['P'][act].mean() / 1e6:5.2f} MW   pitch mean {np.rad2deg(L['beta_meas'][act].mean()):5.2f} deg  "
          f"native {np.rad2deg(L['beta_native'][act].mean()):5.2f}  applied {np.rad2deg(L['beta_applied'][act].mean()):5.2f}  min_pit {np.rad2deg(L['min_pit'][act].mean()):5.2f}")
    print(f"   M_oop(zmq) mean {L['M_oop'][act].mean() / 1e6:6.2f} MNm  std {L['M_oop'][act].std() / 1e6:5.2f}")
    print(f"   region R3 fraction {L['region'][act].mean():.2f}   reward/step mean {L['reward'][act].mean():7.3f} "
          f"(task {L['r_task'][act].mean():7.3f}, load {L['r_load'][act].mean():7.3f})")
    if env.outb is not None:
        O = env.outb
        k = O["Time"] >= args.warmup_s
        print(f"   outb: RootMyc1 mean {O['RootMyc1'][k].mean() / 1e3:6.2f} MNm  TwrBsMyt mean {O['TwrBsMyt'][k].mean() / 1e3:6.2f} MNm  "
              f"GenPwr mean {O['GenPwr'][k].mean() / 1e3:5.2f} MW  BldPitch1 mean {O['BldPitch1'][k].mean():5.2f} deg")
    print(f"   terminated={term}")
    if args.save:
        np.savez(f"{os.path.expanduser(args.save)}_U{ep.mean_wind:g}.npz", **L)
env.close()
