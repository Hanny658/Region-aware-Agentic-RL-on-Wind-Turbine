"""Minimal ctypes wrapper around the compiled ROSCO DISCON library (Bladed interface).

Used by the toy environment so that the *same* libdiscon.so / DISCON.IN that OpenFAST loads
also drives the 1-DOF model. avrSWAP indices below are 0-based (Fortran index - 1).
Modelled on rosco.toolbox.control_interface.ControllerInterface (v2.10.5).
"""
from __future__ import annotations

from ctypes import POINTER, byref, c_char_p, c_float, c_int32, cdll, create_string_buffer
from pathlib import Path

import numpy as np

# avrSWAP layout (0-based)
I_STATUS, I_TIME, I_DT = 0, 1, 2
I_PITCH1, I_PITCH2, I_PITCH3 = 3, 32, 33
I_GEN_POWER = 14
I_GEN_SPEED, I_ROT_SPEED = 19, 20
I_GEN_TQ_MEAS = 22
I_YAW_ERR = 23
I_WIND = 26
I_IPC_FLAG = 27
I_ROOT_OOP = (29, 30, 31)       # blade root out-of-plane bending moments [Nm]
I_YAW_FROM_NORTH = 36
I_PITCH_CMD1 = 41               # avrSWAP(42) pitch command blade 1 [rad]
I_PITCH_CMD_COLL = 44           # avrSWAP(45) collective pitch command [rad]
I_GEN_TQ_CMD = 46               # avrSWAP(47) generator torque demand [Nm]
I_YAW_RATE = 47
I_CHAR_BUF, I_LEN_INFILE, I_LEN_OUTNAME, I_LEN_MSG = 48, 49, 50, 51
I_FA_ACC = 52
I_NUM_BLADES = 60
I_NAC_IMU = 82


class RoscoDiscon:
    def __init__(self, lib_path: str | Path, discon_in: str | Path, dt: float,
                 sim_name: str = "toy", avr_size: int = 500):
        self.lib_path = str(lib_path)
        self.discon_in = str(discon_in)
        self.dt = dt
        self.avr = np.zeros(avr_size, dtype=np.float32)
        self._lib = cdll.LoadLibrary(self.lib_path)
        self._lib.DISCON.argtypes = [POINTER(c_float), POINTER(c_int32), c_char_p, c_char_p, c_char_p]
        self._fail = c_int32(0)
        self._infile = self.discon_in.encode()
        self._outname = sim_name.encode()
        self._msg = create_string_buffer(1000)
        self._initialised = False

    def _call(self):
        p = self.avr.ctypes.data_as(POINTER(c_float))
        self._lib.DISCON(p, byref(self._fail), self._infile, self._outname, self._msg)
        if self._fail.value < 0:
            raise RuntimeError("ROSCO DISCON error: " + self._msg.value.decode(errors="replace"))

    def init(self, pitch: float, gen_speed: float, rot_speed: float, wind: float, gen_torque: float = 0.0):
        a = self.avr
        a[:] = 0.0
        a[I_STATUS] = 0
        a[I_DT] = self.dt
        a[I_NUM_BLADES] = 3
        a[I_PITCH1] = a[I_PITCH2] = a[I_PITCH3] = pitch
        a[I_GEN_SPEED], a[I_ROT_SPEED] = gen_speed, rot_speed
        a[I_WIND] = wind
        a[I_GEN_TQ_MEAS] = gen_torque
        a[I_IPC_FLAG] = 1
        a[I_CHAR_BUF] = 500
        a[I_LEN_INFILE] = len(self._infile)
        a[I_LEN_OUTNAME] = len(self._outname)
        a[I_LEN_MSG] = 500
        self._call()
        a[I_STATUS] = 1
        self._initialised = True

    def step(self, t: float, pitch: float, gen_speed: float, rot_speed: float, gen_torque: float,
             gen_eff: float, wind: float, root_oop=(0.0, 0.0, 0.0), fa_acc: float = 0.0,
             nac_imu: float = 0.0) -> tuple[float, float]:
        """Returns (collective pitch command [rad], generator torque command [Nm])."""
        a = self.avr
        a[I_STATUS] = 1
        a[I_TIME], a[I_DT] = t, self.dt
        a[I_PITCH1] = a[I_PITCH2] = a[I_PITCH3] = pitch
        a[I_GEN_POWER] = gen_speed * gen_torque * gen_eff
        a[I_GEN_TQ_MEAS] = gen_torque
        a[I_GEN_SPEED], a[I_ROT_SPEED] = gen_speed, rot_speed
        a[I_YAW_ERR] = 0.0
        a[I_WIND] = wind
        a[I_YAW_FROM_NORTH] = 0.0
        for i, m in zip(I_ROOT_OOP, root_oop):
            a[i] = m
        a[I_FA_ACC], a[I_NAC_IMU] = fa_acc, nac_imu
        self._call()
        return float(a[I_PITCH_CMD_COLL]), float(a[I_GEN_TQ_CMD])

    def finish(self):
        """Final DISCON call, then really unload the .so: ROSCO refuses a second iStatus==0
        initialisation while the library stays mapped ("already been loaded")."""
        if self._initialised:
            self.avr[I_STATUS] = -1
            try:
                self._call()
            except RuntimeError:
                pass
            self._initialised = False
        if self._lib is not None:
            import _ctypes
            handle = self._lib._handle
            self._lib = None
            try:
                _ctypes.dlclose(handle)
            except OSError:
                pass
