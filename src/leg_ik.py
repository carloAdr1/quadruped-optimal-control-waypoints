import numpy as np


def clip(x, a, b):
    return np.minimum(np.maximum(x, a), b)


class SimpleQuadrupedLegIK:
    """
    IK geométrica genérica de una pata 3DOF.
    Demo / aproximada; puede requerir calibración por robot.
    """

    def __init__(self, l0=0.05, l1=0.20, l2=0.20):
        self.l0 = float(l0)
        self.l1 = float(l1)
        self.l2 = float(l2)

    def solve(self, p_des, side=+1):
        x, y, z = p_des

        r_yz_sq = y**2 + z**2 - self.l0**2
        r_yz_sq = max(r_yz_sq, 1e-8)
        r_yz = np.sqrt(r_yz_sq)

        q1 = np.arctan2(y, -z) - np.arctan2(side * self.l0, r_yz)

        r_sq = x**2 + z**2
        D = (r_sq - self.l1**2 - self.l2**2) / (2.0 * self.l1 * self.l2)
        D = clip(D, -1.0, 1.0)

        q3 = -np.arccos(D)

        alpha = np.arctan2(x, -z)
        beta = np.arctan2(self.l2 * np.sin(-q3), self.l1 + self.l2 * np.cos(q3))
        q2 = alpha - beta

        return np.array([q1, q2, q3], dtype=float)
