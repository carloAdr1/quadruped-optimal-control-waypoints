import numpy as np


def wrap_to_pi(angle: float) -> float:
    return np.arctan2(np.sin(angle), np.cos(angle))


class WaypointTrajectory:
    def __init__(self, waypoints, speed=0.15, dt=0.01, hold_final=True, start_delay=3.0):
        self.waypoints = np.asarray(waypoints, dtype=float)
        self.speed = float(speed)
        self.dt = float(dt)
        self.hold_final = hold_final
        self.start_delay = float(start_delay)

        if self.waypoints.ndim != 2 or self.waypoints.shape[1] != 3:
            raise ValueError("Waypoints deben tener forma (N,3): [x,y,yaw]")

        if len(self.waypoints) < 2:
            raise ValueError("Se requieren al menos 2 waypoints")

        self.segment_times = []
        self.cumulative_times = [self.start_delay]

        for i in range(len(self.waypoints) - 1):
            p0 = self.waypoints[i, :2]
            p1 = self.waypoints[i + 1, :2]
            d = np.linalg.norm(p1 - p0)
            T = max(d / max(self.speed, 1e-6), self.dt)
            self.segment_times.append(T)
            self.cumulative_times.append(self.cumulative_times[-1] + T)

        self.total_time = self.cumulative_times[-1]

    def sample(self, t: float, nominal_height: float = 0.225):
        if t <= self.start_delay:
            wp = self.waypoints[0]
            return (
                np.array([wp[0], wp[1], nominal_height]),
                np.array([0.0, 0.0, 0.0]),
                np.array([0.0, 0.0, wp[2]]),
                np.array([0.0, 0.0, 0.0]),
                False,
            )

        if t >= self.total_time:
            wp = self.waypoints[-1]
            return (
                np.array([wp[0], wp[1], nominal_height]),
                np.array([0.0, 0.0, 0.0]),
                np.array([0.0, 0.0, wp[2]]),
                np.array([0.0, 0.0, 0.0]),
                True,
            )

        seg_idx = 0
        for i in range(len(self.segment_times)):
            if self.cumulative_times[i] <= t < self.cumulative_times[i + 1]:
                seg_idx = i
                break

        t0 = self.cumulative_times[seg_idx]
        T = self.segment_times[seg_idx]
        alpha = np.clip((t - t0) / T, 0.0, 1.0)

        wp0 = self.waypoints[seg_idx]
        wp1 = self.waypoints[seg_idx + 1]

        p0 = wp0[:2]
        p1 = wp1[:2]
        pos_xy = (1.0 - alpha) * p0 + alpha * p1

        delta = p1 - p0
        dist = np.linalg.norm(delta)
        if dist > 1e-9:
            vel_xy = self.speed * delta / dist
        else:
            vel_xy = np.zeros(2)

        yaw0 = wp0[2]
        yaw1 = wp1[2]
        dyaw = wrap_to_pi(yaw1 - yaw0)
        yaw = wrap_to_pi(yaw0 + alpha * dyaw)
        wz = dyaw / T

        pos_ref = np.array([pos_xy[0], pos_xy[1], nominal_height])
        vel_ref = np.array([vel_xy[0], vel_xy[1], 0.0])
        euler_ref = np.array([0.0, 0.0, yaw])
        omega_ref = np.array([0.0, 0.0, wz])

        return pos_ref, vel_ref, euler_ref, omega_ref, False


def build_named_trajectory(name: str):
    name = name.lower()

    if name == "line":
        waypoints = [
            [0.00, 0.00, 0.0],
            [0.08, 0.00, 0.0],
            [0.16, 0.00, 0.0],
        ]
    elif name == "square":
        waypoints = [
            [0.00, 0.00, 0.0],
            [0.08, 0.00, 0.0],
            [0.15, 0.15, np.pi / 2],
            [0.00, 0.15, np.pi],
            [0.00, 0.00, -np.pi / 2],
        ]
    elif name == "zigzag":
        waypoints = [
            [0.00, 0.00, 0.0],
            [0.10, 0.05, 0.1],
            [0.20, -0.05, -0.1],
            [0.30, 0.05, 0.1],
            [0.40, 0.00, 0.0],
        ]
    else:
        raise ValueError(f"Trayectoria desconocida: {name}")

    return np.array(waypoints, dtype=float)
