import numpy as np

LEG_ORDER = ["FL", "FR", "RL", "RR"]


class TrotGaitScheduler:
    """
    Trot simple:
      Fase A: FL + RR en stance, FR + RL en swing
      Fase B: FR + RL en stance, FL + RR en swing
    """

    def __init__(self, period=0.60, duty_factor=0.60):
        self.period = float(period)
        self.duty_factor = float(duty_factor)

        # phase offsets in [0,1)
        self.phase_offset = {
            "FL": 0.0,
            "RR": 0.0,
            "FR": 0.5,
            "RL": 0.5,
        }

    def leg_phase(self, leg: str, t: float) -> float:
        ph = (t / self.period + self.phase_offset[leg]) % 1.0
        return ph

    def is_stance(self, leg: str, t: float) -> bool:
        return self.leg_phase(leg, t) < self.duty_factor

    def is_swing(self, leg: str, t: float) -> bool:
        return not self.is_stance(leg, t)

    def swing_phase(self, leg: str, t: float) -> float:
        """
        Normalized swing phase in [0,1]
        """
        ph = self.leg_phase(leg, t)
        if ph < self.duty_factor:
            return 0.0
        return (ph - self.duty_factor) / max(1e-6, (1.0 - self.duty_factor))

    def stance_phase(self, leg: str, t: float) -> float:
        """
        Normalized stance phase in [0,1]
        """
        ph = self.leg_phase(leg, t)
        if ph >= self.duty_factor:
            return 0.0
        return ph / max(1e-6, self.duty_factor)

    def contact_mask(self, t: float):
        return np.array([self.is_stance(leg, t) for leg in LEG_ORDER], dtype=bool)
