"""Zero-residual rollout of the toy env on each wind file: checks ROSCO ctypes wiring,
steady-state behaviour per region, region labelling and reward magnitudes."""
import argparse
import os
import time

import numpy as np

from envs.base_env import EpisodeSpec, default_config
from envs.toy_env import ToyTurbineEnv

ap = argparse.ArgumentParser()
ap.add_argument("--wind_dir", default="~/wtrl/wind")
ap.add_argument("--lib", default="~/wtrl/rosco_install/lib/libdiscon.so")
ap.add_argument("--discon_dir", default="~/wtrl/runs/toy_discon")
ap.add_argument("--episode_s", type=float, default=150.0)
ap.add_argument("--means", nargs="+", type=float, default=[8, 12.5, 15])
ap.add_argument("--save", default=None, help="npz prefix to dump logs")
args = ap.parse_args()

wd = os.path.expanduser(args.wind_dir)
eps = [EpisodeSpec(wind_file=f"{wd}/U{u:g}_TI8_S1.bts", mean_wind=u, episode_s=args.episode_s)
       for u in args.means]
env = ToyTurbineEnv(default_config(), eps, args.lib, args.discon_dir)
for i, ep in enumerate(eps):
    t0 = time.time()
    obs, _ = env.reset(options={"episode_index": i})
    done = False
    n = 0
    while not done:
        obs, r, term, trunc, info = env.step(np.zeros(1, dtype=np.float32))
        done = term or trunc
        n += 1
    L = env.log_arrays()
    act = L["warmup"] == 0
    wall = time.time() - t0
    print(f"\n== U={ep.mean_wind:g} m/s  steps={n}  wall={wall:.1f}s  ({ep.episode_s / wall:.0f}x RT)")
    print(f"   v_hub  mean {L['v_hub'][act].mean():6.2f}  min {L['v_hub'][act].min():6.2f}  max {L['v_hub'][act].max():6.2f}")
    print(f"   gen_spd mean {L['gen_speed'][act].mean():6.1f} rad/s (rated 122.9)  std {L['gen_speed'][act].std():5.2f}")
    print(f"   P mean {L['P'][act].mean() / 1e6:5.2f} MW   pitch mean {np.rad2deg(L['beta_meas'][act].mean()):5.2f} deg  max {np.rad2deg(L['beta_meas'][act].max()):5.2f}")
    print(f"   M_oop mean {L['M_oop'][act].mean() / 1e6:5.2f} MNm  std {L['M_oop'][act].std() / 1e6:5.2f}")
    print(f"   region R3 fraction {L['region'][act].mean():.2f}   reward/step mean {L['reward'][act].mean():7.3f} "
          f"(task {L['r_task'][act].mean():7.3f}, load {L['r_load'][act].mean():7.3f})")
    print(f"   terminated={term}")
    if args.save:
        np.savez(f"{os.path.expanduser(args.save)}_U{ep.mean_wind:g}.npz", **L)
env.close()
