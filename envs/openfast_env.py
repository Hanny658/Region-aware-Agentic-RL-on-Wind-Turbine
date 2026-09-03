"""OpenFAST backend: one OpenFAST process per episode, coupled through the patched ROSCO ZeroMQ
interface (22 measurements in, 8 setpoints out, one exchange per control step).

Sequence per control step (ROSCO calls UpdateZeroMQ *after* PitchControl, see DISCON.F90):
    OpenFAST -> ROSCO computes native pitch/torque for step k, sends measurements(k) to us
    we reply with the pitch offset to be applied at step k+1  (one-step delay, 10 ms)
The final ROSCO call (iStatus == -1) also performs an exchange; we answer zeros and the
process exits. Full-fidelity channels (RootMyc1, TwrBsMyt, ...) are read from the .outb
afterwards and attached to the episode log.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import zmq

from envs.base_env import EnvConfig, EpisodeSpec, ResidualPitchEnv
from envs.fast_io import set_param
from eval.fast_out import read_outb

# indices into the 22-float measurement vector (see controllers/rosco_patch/ZeroMQInterface.f90)
M_ID, M_STATUS, M_TIME, M_MECH_PWR, M_GEN_PWR, M_GEN_SPD, M_ROT_SPD, M_GEN_TQ = range(8)
M_NAC_HEAD, M_NAC_VANE, M_WIND, M_OOP1, M_OOP2, M_OOP3, M_FA_ACC, M_NAC_IMU, M_AZI = range(8, 17)
M_BLPITCH, M_PC_PITCOMT, M_PC_MINPIT, M_WE_VW, M_PITCOM1 = range(17, 22)
N_MEAS = 22

TEMPLATE_FILES = {
    "fst": "NREL-5MW.fst",
    "ed": "NRELOffshrBsline5MW_Onshore_ElastoDyn.dat",
    "ifw": "NRELOffshrBsline5MW_InflowWind.dat",
    "svd": "NRELOffshrBsline5MW_Onshore_ServoDyn.dat",
    "discon": "DISCON.IN",
}


class OpenFASTEnv(ResidualPitchEnv):
    BACKEND = "openfast"
    def __init__(self, cfg: EnvConfig, episodes: list[EpisodeSpec], template_dir: str, work_dir: str,
                 port: int = 5555, openfast_bin: str = "openfast", seed: int = 0,
                 keep_outputs: bool = False, zmq_timeout_s: float = 120.0):
        super().__init__(cfg, episodes, seed)
        self.template_dir = Path(os.path.expanduser(template_dir))
        self.work_dir = Path(os.path.expanduser(work_dir))
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.port = port
        self.bin = openfast_bin
        self.keep_outputs = keep_outputs
        self.timeout_ms = int(zmq_timeout_s * 1000)
        self.ctx = zmq.Context.instance()
        self.sock = None
        self.proc = None
        self._n_ep = 0
        self._last_t = None
        self._pending_offset = 0.0
        self.outb: dict[str, np.ndarray] | None = None

    # ---------------------------------------------------------------- case setup
    def _initial_conditions(self, spec: EpisodeSpec) -> tuple[float, float]:
        """(rotor speed rpm, pitch deg) rough steady state for the episode mean wind."""
        tb = self.tb
        v = spec.mean_wind
        omega = min(float(tb["tsr_opt"]) * v / float(tb["rotor_radius_m"]),
                    float(tb["rated_rotor_speed_rpm"]) * 2 * np.pi / 60)
        pitch = 0.0 if v <= tb["rated_wind_ms"] else 2.0 * (v - tb["rated_wind_ms"]) + 1.0
        return omega * 60 / (2 * np.pi), pitch

    def _make_case(self, spec: EpisodeSpec) -> Path:
        case = self.work_dir / f"ep{self._n_ep:05d}_p{self.port}"
        if case.exists():
            shutil.rmtree(case)
        case.mkdir(parents=True)
        # symlink everything from the template, then materialise the files we edit
        for item in self.template_dir.iterdir():
            os.symlink(item, case / item.name)
        for f in TEMPLATE_FILES.values():
            os.unlink(case / f)
            shutil.copy(self.template_dir / f, case / f)
        rpm, pitch = self._initial_conditions(spec)
        set_param(case / TEMPLATE_FILES["fst"], "TMax", f"{spec.episode_s:g}")
        ed = case / TEMPLATE_FILES["ed"]
        set_param(ed, "RotSpeed", f"{rpm:.3f}")
        for k in (1, 2, 3):
            set_param(ed, f"BlPitch({k})", f"{pitch:.3f}")
        set_param(case / TEMPLATE_FILES["ifw"], "FileName_BTS", f'"{os.path.expanduser(spec.wind_file)}"')
        set_param(case / TEMPLATE_FILES["discon"], "ZMQ_CommAddress", f'"tcp://127.0.0.1:{self.port}"')
        return case

    # ---------------------------------------------------------------- zmq plumbing
    def _open_socket(self):
        self._close_socket()
        self.sock = self.ctx.socket(zmq.REP)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.bind(f"tcp://127.0.0.1:{self.port}")

    def _close_socket(self):
        if self.sock is not None:
            self.sock.close()
            self.sock = None

    def _recv(self) -> np.ndarray:
        if self.sock.poll(self.timeout_ms) == 0:
            raise TimeoutError(f"no ZMQ message from OpenFAST within {self.timeout_ms / 1000:.0f}s "
                               f"(process alive: {self.proc is not None and self.proc.poll() is None})")
        msg = self.sock.recv().decode(errors="replace").replace("\x00", "")
        vals = np.array([float(x) for x in msg.split(",")])
        if len(vals) != N_MEAS:
            raise ValueError(f"expected {N_MEAS} measurements, got {len(vals)}: {msg[:80]}")
        return vals

    def _send(self, pitch_offset: float, tq_offset: float = 0.0, ipc3=None):
        p = [pitch_offset] * 3 if ipc3 is None else [pitch_offset + float(x) for x in ipc3]
        sp = [tq_offset, 0.0, p[0], p[1], p[2], 0.0, 0.0, 0.0]
        self.sock.send(", ".join(f"{s:.8e}" for s in sp).encode())

    def _to_meas(self, v: np.ndarray, offset: float) -> dict:
        return {
            "t": float(v[M_TIME]), "P": float(v[M_GEN_PWR]), "gen_speed": float(v[M_GEN_SPD]),
            "rot_speed": float(v[M_ROT_SPD]), "gen_torque": float(v[M_GEN_TQ]),
            "v_hub": float(v[M_WIND]), "v_est": float(v[M_WE_VW]), "M_oop": float(v[M_OOP1]),
            "M_oop2": float(v[M_OOP2]), "M_oop3": float(v[M_OOP3]),
            "beta_meas": float(v[M_BLPITCH]), "beta_native": float(v[M_PC_PITCOMT]),
            "min_pit": float(v[M_PC_MINPIT]), "beta_applied": float(v[M_PITCOM1]), "offset": offset,
            "fa_acc": float(v[M_FA_ACC]), "azimuth": float(v[M_AZI]),
        }

    # ---------------------------------------------------------------- backend hooks
    def _sim_reset(self, spec: EpisodeSpec) -> dict:
        self._sim_close()
        self.outb = None
        self.case = self._make_case(spec)
        self._tmax = float(spec.episode_s)
        self._n_ep += 1
        self._open_socket()
        log = open(self.case / "openfast.log", "w")
        self.proc = subprocess.Popen([self.bin, TEMPLATE_FILES["fst"]], cwd=self.case,
                                     stdout=log, stderr=subprocess.STDOUT)
        self._t_start = time.time()
        v = self._recv()                       # first exchange (ROSCO init call, t = 0)
        self._last_t = float(v[M_TIME])
        return self._to_meas(v, 0.0)

    def _sim_step(self, pitch_offset: float, tq_offset: float = 0.0, ipc3=None) -> dict:
        # answer the pending request with this step's offsets, then wait for the next measurement.
        self._send(pitch_offset, tq_offset, ipc3)
        while True:
            v = self._recv()
            t = float(v[M_TIME])
            if v[M_STATUS] < -0.5:             # final call: reply and let the process finish
                self._send(0.0)
                self._finish()
                return self._to_meas(v, pitch_offset)
            if self._last_t is not None and t <= self._last_t + 1e-9:
                self._send(pitch_offset, tq_offset, ipc3)   # repeated call at the same time (corrector)
                continue
            self._last_t = t
            if t >= self._tmax - 1e-6:         # last regular step: drain the final exchange now
                self._send(pitch_offset, tq_offset, ipc3)
                try:
                    vf = self._recv()
                    self._send(0.0)
                except TimeoutError:
                    pass
                self._finish()
            return self._to_meas(v, pitch_offset)

    def _finish(self):
        if self.proc is not None:
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.wall_s = time.time() - self._t_start
        outb = self.case / (TEMPLATE_FILES["fst"][:-4] + ".outb")
        if outb.exists():
            self.outb, _ = read_outb(outb)
        self._close_socket()
        if not self.keep_outputs:
            shutil.rmtree(self.case, ignore_errors=True)
        self.proc = None

    def _sim_close(self):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait()
        self.proc = None
        self._close_socket()

    def close(self):
        self._sim_close()
