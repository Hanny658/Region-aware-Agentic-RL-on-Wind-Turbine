"""Operating-region routers.

Phase 1 uses only the oracle rule (decision #3, refined):
    R3  <=>  ROSCO-native collective pitch command > pitch_floor + threshold, held for >= hold_s
    R2  <=>  ... < threshold, held for >= hold_s
where pitch_floor is ROSCO's *current* lower pitch limit (fine pitch, or the peak-shaving
minimum when PS_Mode = 1). Using the floor instead of the constant fine pitch keeps
peak-shaving operation below rated (pitch sitting on the PS limit) labelled as R2.
The same rule gates the region-conditional reward, so the label used for the reward and the
label used for routing are identical by construction. Learned routers (MLP/KAN) come later
and must be evaluated against this rule.
"""
from __future__ import annotations

R2, R3 = 0, 1


class OracleRegionRouter:
    def __init__(self, fine_pitch_rad: float, threshold_rad: float, hold_s: float, dt: float,
                 initial_region: int = R2):
        self.fine_pitch = fine_pitch_rad
        self.thr = threshold_rad
        self.hold_steps = max(1, int(round(hold_s / dt)))
        self.region = initial_region
        self._counter = 0

    def reset(self, initial_region: int = R2) -> int:
        self.region = initial_region
        self._counter = 0
        return self.region

    def update(self, pitch_cmd_native_rad: float, pitch_floor_rad: float | None = None) -> int:
        """Advance one control step with ROSCO's own (pre-residual) pitch command and its
        current lower limit (defaults to the fine pitch)."""
        floor = self.fine_pitch if pitch_floor_rad is None else max(pitch_floor_rad, self.fine_pitch)
        candidate = R3 if pitch_cmd_native_rad > floor + self.thr else R2
        if candidate != self.region:
            self._counter += 1
            if self._counter >= self.hold_steps:
                self.region = candidate
                self._counter = 0
        else:
            self._counter = 0
        return self.region
