"""Step-by-step trace of the OpenFAST/ZMQ handshake for the first few control steps."""
import os
import time

import numpy as np

from envs.base_env import EpisodeSpec, default_config
from envs.openfast_env import M_STATUS, M_TIME, OpenFASTEnv

wd = os.path.expanduser("~/wtrl/wind")
ep = EpisodeSpec(wind_file=f"{wd}/U11.4_TI8_S1.bts", mean_wind=11.4, episode_s=5.0, warmup_s=1.0)
env = OpenFASTEnv(default_config(), [ep], "~/wtrl/runs/template_5mw", "~/wtrl/runs/work_dbg",
                  port=5602, keep_outputs=True)
env.spec_ep = ep
env._load_baseline(ep)
m = env._sim_reset(ep)
print("case dir:", env.case, "exists:", env.case.exists())
print("first msg: t=%.3f iStatus=%g  sock=%s proc_alive=%s" % (m["t"], env._last_t, env.sock is not None, env.proc.poll() is None))
raw_last = None
for i in range(6):
    env._send(0.0)
    v = env._recv()
    print(f"  exchange {i}: t={v[M_TIME]:.4f} iStatus={v[M_STATUS]:g} P={v[4] / 1e6:.3f}MW genspd={v[5]:.2f} pitch={np.rad2deg(v[17]):.3f} native={np.rad2deg(v[18]):.3f} minpit={np.rad2deg(v[19]):.3f} Vw={v[20]:.2f} M1={v[11] / 1e6:.3f}")
t0 = time.time()
n = 0
while True:
    env._send(0.0)
    v = env._recv()
    n += 1
    if v[M_STATUS] < -0.5 or v[M_TIME] >= ep.episode_s - 1e-6:
        print(f"  ... {n} more exchanges, last t={v[M_TIME]:.3f} iStatus={v[M_STATUS]:g}  ({time.time() - t0:.1f}s)")
        break
env._send(0.0)
try:
    v = env._recv()
    print(f"  after last: t={v[M_TIME]:.3f} iStatus={v[M_STATUS]:g}")
    env._send(0.0)
except TimeoutError as e:
    print("  no further message:", e)
env.proc.wait(timeout=30)
print("openfast exit code:", env.proc.returncode)
print(open(env.case / "openfast.log").read()[-600:])
