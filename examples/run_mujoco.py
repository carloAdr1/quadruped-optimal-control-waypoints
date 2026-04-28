#!/usr/bin/env python3
import sys
import os
import argparse
import threading
import select
import numpy as np
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gym_quadruped.quadruped_env import QuadrupedEnv
from src.trajectory_generator import WaypointTrajectory, build_named_trajectory
from src.dynamics import QuadrupedDynamics
from src.estimator_ekf import OrientationEKF
from src.controller_pmp import PontryaginController
from src.controller_lqg import LQGController
from src.controller_mpc import MPCController
from src.gait_scheduler import TrotGaitScheduler, LEG_ORDER
from src.foot_trajectory import JointSpaceTrotPlanner


ROBOT_MASS = 9.0
ROBOT_INERTIA = np.diag([0.107, 0.098, 0.024])
ROBOT_HIP_HEIGHT = 0.225
ROBOT_FOOT_OFFSET = np.array([
    [0.19,  0.111, -0.225],
    [0.19, -0.111, -0.225],
    [-0.19,  0.111, -0.225],
    [-0.19, -0.111, -0.225],
])


@dataclass
class TeleopState:
    vx: float = 0.0
    vy: float = 0.0
    wz: float = 0.0
    step_lin: float = 0.03
    step_ang: float = 0.08
    max_vx: float = 0.20
    max_vy: float = 0.10
    max_wz: float = 0.40
    quit_requested: bool = False

    def clamp(self):
        self.vx = float(np.clip(self.vx, -self.max_vx, self.max_vx))
        self.vy = float(np.clip(self.vy, -self.max_vy, self.max_vy))
        self.wz = float(np.clip(self.wz, -self.max_wz, self.max_wz))

    def zero(self):
        self.vx = 0.0
        self.vy = 0.0
        self.wz = 0.0


def teleop_keyboard_loop(teleop: TeleopState):
    import termios
    import tty

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    print("\n[Teleop enabled]")
    print("  ↑ / ↓ : forward/backward")
    print("  ← / → : yaw left/right")
    print("  z / c : lateral left/right")
    print("  space : zero commands")
    print("  Ctrl+C: quit\n")

    try:
        tty.setcbreak(fd)
        while not teleop.quit_requested:
            if select.select([sys.stdin], [], [], 0.05)[0]:
                ch = sys.stdin.read(1)

                if ch == "\x1b":
                    seq1 = sys.stdin.read(1)
                    seq2 = sys.stdin.read(1)
                    if seq1 == "[":
                        if seq2 == "A":
                            teleop.vx += teleop.step_lin
                        elif seq2 == "B":
                            teleop.vx -= teleop.step_lin
                        elif seq2 == "C":
                            teleop.wz -= teleop.step_ang
                        elif seq2 == "D":
                            teleop.wz += teleop.step_ang
                elif ch == "z":
                    teleop.vy += teleop.step_lin
                elif ch == "c":
                    teleop.vy -= teleop.step_lin
                elif ch == " ":
                    teleop.zero()

                teleop.clamp()
                print(
                    f"\rcmd -> vx={teleop.vx:+.2f}, vy={teleop.vy:+.2f}, wz={teleop.wz:+.2f}   ",
                    end="",
                    flush=True,
                )
    except KeyboardInterrupt:
        teleop.quit_requested = True
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        print()


def wrap_to_pi(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def get_state(env) -> np.ndarray:
    p = env.base_pos.copy()
    v = env.base_lin_vel(frame="world")
    rpy = env.base_ori_euler_xyz.copy()
    omega = env.base_ang_vel(frame="base")
    return np.concatenate([p, v, rpy, omega])


def grf_to_torques(env, grfs: np.ndarray, contact: np.ndarray) -> np.ndarray:
    tau = np.zeros(env.mjModel.nu)
    try:
        jacobians = env.feet_jacobians(frame="world")
    except Exception:
        return tau

    for i, leg in enumerate(["FL", "FR", "RL", "RR"]):
        if not contact[i]:
            continue
        f_leg = grfs[3 * i: 3 * i + 3]
        J_full = jacobians[leg]
        leg_idx = env.legs_qvel_idx[leg]
        J_leg = J_full[:, leg_idx]
        tau_leg = -J_leg.T @ f_leg
        tau_idx = env.legs_tau_idx[leg]
        tau[tau_idx] = tau_leg
    return tau


def get_contacts(env) -> np.ndarray:
    try:
        cs, _ = env.feet_contact_state()
        return np.array([cs.FL, cs.FR, cs.RL, cs.RR], dtype=bool)
    except Exception:
        return np.ones(4, dtype=bool)


def get_feet_world(env) -> np.ndarray:
    try:
        fp = env.feet_pos(frame="world")
        return np.array([fp.FL, fp.FR, fp.RL, fp.RR])
    except Exception:
        return None


def get_joint_state_dict(env):
    """
    Asume 12 joints al final de qpos y qvel.
    Orden esperado: FL, FR, RL, RR (3 por pata).
    """
    qj = env.mjData.qpos[-12:].copy().reshape(4, 3)
    dqj = env.mjData.qvel[-12:].copy().reshape(4, 3)

    q_dict = {leg: qj[i].copy() for i, leg in enumerate(LEG_ORDER)}
    dq_dict = {leg: dqj[i].copy() for i, leg in enumerate(LEG_ORDER)}
    return q_dict, dq_dict


def compute_gait_pd_torque(env, q_des_dict, q_dict, dq_dict, kp=None, kd=None):
    if kp is None:
        kp = np.array([18.0, 22.0, 16.0])
    if kd is None:
        kd = np.array([0.8, 1.0, 0.7])

    tau = np.zeros(env.mjModel.nu)

    for leg in LEG_ORDER:
        q = q_dict[leg]
        dq = dq_dict[leg]
        q_des = q_des_dict[leg]

        tau_leg = kp * (q_des - q) - kd * dq

        tau_idx = env.legs_tau_idx[leg]
        tau[tau_idx] = tau_leg

    return tau


def build_dynamics():
    dyn = QuadrupedDynamics(
        mass=ROBOT_MASS,
        inertia=ROBOT_INERTIA,
        dt=0.002,
    )
    dyn.r_feet_body = ROBOT_FOOT_OFFSET.copy()
    return dyn


def build_cost_matrices():
    Q = np.diag([
        20, 20, 300,
        12, 12, 50,
        120, 120, 20,
        2, 2, 6,
    ])
    R = np.eye(12) * 5e-4
    Q_f = Q * 3
    return Q, R, Q_f


def build_reference_state_from_velocity_guidance(x, height, vx_cmd, vy_cmd, yaw_cmd, wz_cmd):
    """
    Referencia suave:
    - mantiene posición actual en x,y para no pelear contra locomoción inexistente
    - sí sigue velocidades y yaw
    """
    x_ref = x.copy()
    x_ref[2] = height
    x_ref[3:6] = np.array([vx_cmd, vy_cmd, 0.0])
    x_ref[6:9] = np.array([0.0, 0.0, yaw_cmd])
    x_ref[9:12] = np.array([0.0, 0.0, wz_cmd])
    return x_ref


def build_controller(name: str, dyn: QuadrupedDynamics, Q, R, Q_f, x_ref):
    A_d, B_d, g_d = dyn.get_linear_system(x_ref)
    A_c, B_c = dyn.continuous_AB(x_ref)

    if name == "pmp":
        ctrl = PontryaginController(
            A=A_c,
            B=B_c,
            Q_s=Q,
            R_u=R,
            Q_f=Q_f,
            g_aff=dyn.gravity_vector() / dyn.dt,
            dt=dyn.dt,
            horizon=300,
        )
        ctrl.solve_discrete_sweep(x_ref.copy(), x_ref)
        print("  [PMP] Hamiltonian-based controller initialized")
        return ctrl

    if name == "lqg":
        ctrl = LQGController(
            A_d=A_d,
            B_d=B_d,
            g_d=g_d,
            Q=Q * dyn.dt,
            R=R * dyn.dt,
            Q_proc=np.diag([1e-3] * 3 + [5e-3] * 3 + [5e-3] * 3 + [1e-2] * 3),
            R_meas=np.diag([5e-3] * 3 + [2e-2] * 3 + [1e-2] * 3 + [5e-2] * 3),
        )
        ctrl.set_initial_estimate(x_ref)
        print("  [LQG] Controller initialized")
        return ctrl

    if name == "mpc":
        ctrl = MPCController(
            A_d=A_d,
            B_d=B_d,
            g_d=g_d,
            Q=Q * dyn.dt,
            R=R * dyn.dt,
            Q_f=Q_f * dyn.dt,
            N=8,
            mu=0.6,
            fz_max=120.0,
        )
        print("  [MPC] Horizon=8, OSQP-based controller initialized")
        return ctrl

    raise ValueError(f"Unknown controller: {name}")


def make_velocity_command_from_waypoints(x, trajectory, t, speed_cap=0.02, yaw_k=0.4):
    pos_ref, vel_ref, euler_ref, omega_ref, _ = trajectory.sample(t, nominal_height=ROBOT_HIP_HEIGHT)

    dx = pos_ref[0] - x[0]
    dy = pos_ref[1] - x[1]
    desired_yaw = np.arctan2(dy, dx) if (dx * dx + dy * dy) > 1e-8 else euler_ref[2]
    yaw_err = wrap_to_pi(desired_yaw - x[8])

    vx_cmd = float(np.clip(vel_ref[0], -speed_cap, speed_cap))
    vy_cmd = float(np.clip(vel_ref[1], -0.5 * speed_cap, 0.5 * speed_cap))
    wz_cmd = float(np.clip(yaw_k * yaw_err, -0.05, 0.05))

    return vx_cmd, vy_cmd, desired_yaw, wz_cmd, pos_ref


def save_single_run_plot(result, controller_name, robot_name, disturbance_type):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs("results", exist_ok=True)

    log_t = result["time"]
    log_x = result["state"]
    log_pos_ref = result["pos_ref"]
    log_u = result["control"]
    log_dist = result["disturbance"]

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(log_t, log_x[:, 0], label="x")
    axes[0].plot(log_t, log_pos_ref[:, 0], "--", label="x_ref")
    axes[0].plot(log_t, log_x[:, 1], label="y")
    axes[0].plot(log_t, log_pos_ref[:, 1], "--", label="y_ref")
    axes[0].set_ylabel("XY [m]")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(log_t, log_x[:, 3], label="vx")
    axes[1].plot(log_t, log_x[:, 4], label="vy")
    axes[1].set_ylabel("Velocity [m/s]")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(log_t, np.linalg.norm(log_x[:, :2] - log_pos_ref[:, :2], axis=1), label="XY error")
    axes[2].set_ylabel("XY err [m]")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(log_t, np.linalg.norm(log_u, axis=1), label="||GRFs||")
    axes[3].fill_between(log_t, 0, log_dist * 2, alpha=0.25, label="disturbance")
    axes[3].set_ylabel("Force [N]")
    axes[3].set_xlabel("Time [s]")
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)

    path = f"results/mujoco_{controller_name}_{robot_name}_{disturbance_type}.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\n  Plot saved: {path}")


def run(
    controller_name: str,
    robot_name: str = "mini_cheetah",
    teleop_enabled: bool = False,
    render: bool = True,
    duration: float = 10.0,
    disturbance_type: str = "none",
    save_log: bool = True,
    trajectory_name: str = "none",
    waypoint_speed: float = 0.05,
):
    print(f"\n{'=' * 60}")
    print(f"  Controller:   {controller_name.upper()}")
    print(f"  Robot:        {robot_name}")
    print(f"  Teleop:       {teleop_enabled}")
    print(f"  Duration:     {duration}s")
    print(f"  Disturbance:  {disturbance_type}")
    if trajectory_name != "none":
        print(f"  Trajectory:   {trajectory_name}")
        print(f"  WP speed:     {waypoint_speed:.2f} m/s")
    print(f"{'=' * 60}\n")

    state_obs_names = tuple(QuadrupedEnv.ALL_OBS)

    env = QuadrupedEnv(
        robot=robot_name,
        scene="flat",
        sim_dt=0.002,
        base_vel_command_type="human",
        state_obs_names=state_obs_names,
    )

    _ = env.reset(random=False)
    if render:
        env.render()

    teleop = TeleopState()
    if teleop_enabled and trajectory_name == "none":
        teleop_thread = threading.Thread(target=teleop_keyboard_loop, args=(teleop,), daemon=True)
        teleop_thread.start()

    dyn = build_dynamics()
    Q, R, Q_f = build_cost_matrices()

    x0 = get_state(env)

    q_dict0, dq_dict0 = get_joint_state_dict(env)
    gait = TrotGaitScheduler(period=0.55, duty_factor=0.60)
    gait_planner = JointSpaceTrotPlanner(q_dict0)

    trajectory = None
    if trajectory_name != "none":
        wps = build_named_trajectory(trajectory_name)
        wps[:, 0] += x0[0]
        wps[:, 1] += x0[1]
        trajectory = WaypointTrajectory(wps, speed=waypoint_speed, dt=0.01, start_delay=3.0)

    x_ref0 = build_reference_state_from_velocity_guidance(
        x=x0, height=ROBOT_HIP_HEIGHT, vx_cmd=0.0, vy_cmd=0.0, yaw_cmd=x0[8], wz_cmd=0.0
    )
    u_ref = dyn.standing_control()
    controller = build_controller(controller_name, dyn, Q, R, Q_f, x_ref0)

    ori_ekf = OrientationEKF(dt=env.mjModel.opt.timestep)

    sim_dt = env.mjModel.opt.timestep
    ctrl_dt = 0.01
    ctrl_steps = max(1, int(ctrl_dt / sim_dt))
    n_steps = int(duration / sim_dt)

    log_t, log_x, log_pos_ref, log_u, log_err, log_dist = [], [], [], [], [], []
    current_grfs = u_ref.copy()
    stable_resets = 0

    print(f"  Sim dt: {sim_dt}s, Ctrl rate: {1 / ctrl_dt:.0f} Hz, Total steps: {n_steps}")
    print("  Starting simulation...\n")

    try:
        for step in range(n_steps):
            t = step * sim_dt
            x = get_state(env)
            contact = get_contacts(env)
            r_feet = get_feet_world(env)

            if trajectory is not None:
                vx_cmd, vy_cmd, yaw_cmd, wz_cmd, pos_ref = make_velocity_command_from_waypoints(
                    x, trajectory, t, speed_cap=waypoint_speed, yaw_k=0.4
                )
            else:
                vx_cmd = teleop.vx if teleop_enabled else 0.0
                vy_cmd = teleop.vy if teleop_enabled else 0.0
                wz_cmd = teleop.wz if teleop_enabled else 0.0
                yaw_cmd = x[8]
                pos_ref = np.array([x[0], x[1], ROBOT_HIP_HEIGHT])

            x_ref = build_reference_state_from_velocity_guidance(
                x=x,
                height=ROBOT_HIP_HEIGHT,
                vx_cmd=vx_cmd,
                vy_cmd=vy_cmd,
                yaw_cmd=yaw_cmd,
                wz_cmd=wz_cmd,
            )

            q_dict, dq_dict = get_joint_state_dict(env)
            q_des_dict = gait_planner.get_joint_targets(
                t=t,
                gait=gait,
                vx_cmd=vx_cmd,
                vy_cmd=vy_cmd,
                wz_cmd=wz_cmd,
            )

            dist = np.zeros(6)
            if disturbance_type == "impulse":
                if 2.0 <= t < 2.15:
                    dist = np.array([30.0, 10.0, 0.0, 0.0, 0.0, 2.0])
            elif disturbance_type == "persistent":
                if t >= 2.0:
                    dist = np.array([8.0, 3.0, 0.0, 0.0, 0.0, 1.0])

            env.mjData.qfrc_applied[:6] = dist

            gyro = env.base_ang_vel(frame="base")
            accel_world = env.base_lin_acc(frame="world")
            R_WB = env.base_configuration[0:3, 0:3]
            accel_body = R_WB.T @ (accel_world - np.array([0.0, 0.0, -9.81]))
            ori_ekf.predict(gyro)
            ori_ekf.update_accel(accel_body)

            if step % ctrl_steps == 0:
                try:
                    if controller_name == "lqg":
                        y = x + np.random.randn(12) * np.array(
                            [5e-3] * 3 + [2e-2] * 3 + [1e-2] * 3 + [5e-2] * 3
                        )
                        current_grfs = controller.step(y, x_ref, u_ref)
                    else:
                        current_grfs = controller.compute_control(x=x, x_ref=x_ref, u_ref=u_ref)
                except Exception as e:
                    if step < 5:
                        print(f"  Controller error at t={t:.3f}: {e}")
                    current_grfs = u_ref.copy()

                current_grfs = np.clip(current_grfs, -100.0, 100.0)
                for i in range(4):
                    if not contact[i]:
                        current_grfs[3 * i:3 * i + 3] = 0.0

            tau_body = grf_to_torques(env, current_grfs, contact)
            tau_gait = compute_gait_pd_torque(env, q_des_dict, q_dict, dq_dict)

            tau = tau_body + 0.55 * tau_gait
            tau = np.clip(tau, -22.0, 22.0)

            _, _, terminated, _, _ = env.step(action=tau)

            if render:
                env.render()

            log_t.append(t)
            log_x.append(x.copy())
            log_pos_ref.append(pos_ref.copy())
            log_u.append(current_grfs.copy())
            log_err.append(np.linalg.norm(x[3:6] - x_ref[3:6]))
            log_dist.append(np.linalg.norm(dist))

            if step % int(1.0 / sim_dt) == 0:
                pos_err = np.linalg.norm(x[:2] - pos_ref[:2])
                vel_err = np.linalg.norm(x[3:6] - x_ref[3:6])
                print(
                    f"  t={t:5.1f}s | pos_err={pos_err:.4f}m | "
                    f"vel_err={vel_err:.4f}m/s | "
                    f"height={x[2]:.3f}m | "
                    f"vx={x[3]:+.3f} | vy={x[4]:+.3f} | wz={x[11]:+.3f} | "
                    f"cmd=({vx_cmd:+.2f},{vy_cmd:+.2f},{wz_cmd:+.2f})"
                )

            if terminated:
                print(f"  Terminated at t={t:.2f}s")
                stable_resets += 1
                _ = env.reset(random=False)

                ori_ekf = OrientationEKF(dt=env.mjModel.opt.timestep)
                x_reset = get_state(env)
                x_ref_reset = build_reference_state_from_velocity_guidance(
                    x=x_reset,
                    height=ROBOT_HIP_HEIGHT,
                    vx_cmd=0.0,
                    vy_cmd=0.0,
                    yaw_cmd=x_reset[8],
                    wz_cmd=0.0,
                )
                controller = build_controller(controller_name, dyn, Q, R, Q_f, x_ref_reset)
                current_grfs = u_ref.copy()

                q_dict_reset, dq_dict_reset = get_joint_state_dict(env)
                gait = TrotGaitScheduler(period=0.55, duty_factor=0.60)
                gait_planner = JointSpaceTrotPlanner(q_dict_reset)

                if render:
                    env.render()

    except KeyboardInterrupt:
        print("\n  Interrupted by user.")
    finally:
        teleop.quit_requested = True
        env.close()

    log_t = np.array(log_t)
    log_x = np.array(log_x)
    log_pos_ref = np.array(log_pos_ref)
    log_u = np.array(log_u)
    log_err = np.array(log_err)
    log_dist = np.array(log_dist)

    result = {
        "time": log_t,
        "state": log_x,
        "pos_ref": log_pos_ref,
        "control": log_u,
        "error": log_err,
        "disturbance": log_dist,
    }

    if save_log and len(log_t) > 1:
        save_single_run_plot(result, controller_name, robot_name, disturbance_type)

    xy_rmse = np.sqrt(np.mean(np.sum((log_x[:, :2] - log_pos_ref[:, :2]) ** 2, axis=1)))
    xy_max = np.max(np.linalg.norm(log_x[:, :2] - log_pos_ref[:, :2], axis=1))

    print(f"\n  --- {controller_name.upper()} Summary ---")
    print(f"  Velocity RMSE: {np.sqrt(np.mean(log_err**2)):.4f}")
    print(f"  XY tracking RMSE: {xy_rmse:.4f} m")
    print(f"  XY max tracking error: {xy_max:.4f} m")
    print(f"  Mean GRF norm: {np.mean(np.linalg.norm(log_u, axis=1)):.1f} N")
    print(f"  Resets: {stable_resets}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quadruped control with MuJoCo rendering")
    parser.add_argument("--controller", default="lqg", choices=["pmp", "lqg", "mpc"])
    parser.add_argument("--robot-name", type=str, default="mini_cheetah")
    parser.add_argument("--teleop", action="store_true")
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--disturbance", default="none", choices=["impulse", "persistent", "none"])
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--trajectory", type=str, default="none", choices=["none", "line", "square", "zigzag"])
    parser.add_argument("--waypoint-speed", type=float, default=0.05)
    args = parser.parse_args()

    run(
        controller_name=args.controller,
        robot_name=args.robot_name,
        teleop_enabled=args.teleop,
        render=not args.no_render,
        duration=args.duration,
        disturbance_type=args.disturbance,
        save_log=True,
        trajectory_name=args.trajectory,
        waypoint_speed=args.waypoint_speed,
    )
