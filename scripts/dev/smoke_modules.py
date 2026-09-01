"""Quick sanity checks for the pure-python building blocks (no simulator needed)."""
import numpy as np

from controllers.router import OracleRegionRouter, R3
from envs.reward import RegionReward
from envs.safety import ResidualSafety, SecondOrderDamper
from eval.metrics import del_rainflow

T_rot = 60.0 / 12.1                     # rated rotor period [s]
wn = 6.0 * 4.0 * 9.23 / T_rot           # paper: omega_n = 6 * omega_n,ref, omega_n,ref = 4*9.23/T_rot
d = SecondOrderDamper(omega_n=wn, zeta=1.0, dt=0.01)
ys = [d.step(1.0) for _ in range(300)]
print(f"damper wn={wn:.2f} rad/s; step response @0.1s/0.5s/3s: {ys[9]:.3f} {ys[49]:.3f} {ys[-1]:.3f}")

r = OracleRegionRouter(0.0, np.deg2rad(0.5), 1.0, 0.01)
seq = [r.update(np.deg2rad(2.0), 0.0) for _ in range(150)]
print("router: first R3 index (expect 99):", seq.index(R3))

s = ResidualSafety(0.05, 0.0, 1.5708, damper=None)
print("safety R2 clamp (-0.6 -> 0):", s.apply(-0.6, 0, 0.0, 0.0),
      "| R3 lower bound (native 0.005, min 0.0, unit -1 ->", s.apply(-1.0, 1, 0.005, 0.0), ")")

rw = RegionReward({"w_power": 20, "w_speed": 20, "tau_speed_err": 0.02, "lambda_load": 1.0,
                   "lambda_act": 0.1, "kappa_beta_rad": 0.1, "load_proxy": "range_inc", "dM_ref_nm": 4e3,
                   "R_ref_nm": 1.0e6, "window_s": 10.0}, 122.9, "toy", 0.01)
tot = 0.0
for i, M in enumerate([0.0, 0.2e6, -0.3e6, 0.1e6, 0.5e6, 0.0]):     # window range ends at 0.8 MNm
    tot += rw(1, 5e6, 5e6, 122.9, M, 0.0, 0.0)[1]["r_load"]
print("range_inc: sum of load penalties over 6 steps (expect -0.8*1000/1.0 = -800):", round(tot, 1))
print("DEL sanity (unit sine, m=10, 10 cycles/s):",
      round(del_rainflow(np.sin(np.linspace(0, 20 * np.pi, 2000)), 0.01, 10, neq_per_s=1.0), 3))
print("OK")
