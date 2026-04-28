#!/usr/bin/env python3
import os
import sys
import csv
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gym_quadruped.quadruped_env import QuadrupedEnv
from src.gait_scheduler import TrotGaitScheduler, LEG_ORDER
from src.foot_trajectory import JointSpaceTrotPlanner

# These are imported from the professor-style MuJoCo optimal-control demo.
# They are used only when --controller is lqg, pmp, or mpc.
try:
    from run_mujoco import (
        build_dynamics,
        build_cost_matrices,
        build_controller,
        build_reference_state_from_velocity_guidance,
        get_state,
        get_contacts,
        grf_to_torques,
        ROBOT_HIP_HEIGHT,
    )
    HAS_OPT_CONTROL = True
except Exception as e:
    HAS_OPT_CONTROL = False
    OPT_IMPORT_ERROR = e
    ROBOT_HIP_HEIGHT = 0.225


def get_joint_state_dict(env):
    qj = env.mjData.qpos[-12:].copy().reshape(4, 3)
    dqj = env.mjData.qvel[-12:].copy().reshape(4, 3)

    q_dict = {leg: qj[i].copy() for i, leg in enumerate(LEG_ORDER)}
    dq_dict = {leg: dqj[i].copy() for i, leg in enumerate(LEG_ORDER)}

    return q_dict, dq_dict


def pd_leg_torque(env, q_des_dict, q_dict, dq_dict, kp_dict, kd_dict):
    tau = np.zeros(env.mjModel.nu)

    for leg in LEG_ORDER:
        tau_leg = (
            kp_dict[leg] * (q_des_dict[leg] - q_dict[leg])
            - kd_dict[leg] * dq_dict[leg]
        )
        tau[env.legs_tau_idx[leg]] = tau_leg

    return tau


def save_csv(filename, rows):
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)

    fieldnames = [
        "t",
        "controller",
        "opt_alpha",
        "phase",
        "target_waypoint",
        "x_wp",
        "y_wp",
        "yaw_wp",
        "x_ref",
        "y_ref",
        "x_real",
        "y_real",
        "z_real",
        "ex",
        "ey",
        "e_pos",
        "vx_real",
        "vx_ref",
        "vx_cmd_eff",
        "vy_cmd_eff",
        "wz_cmd_eff",
        "tau_gait_norm",
        "tau_opt_norm",
        "tau_total_norm",
        "opt_failed",
    ]

    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class PositionPI:
    def __init__(self, kp=0.25, ki=0.025, vmax=0.12):
        self.kp = kp
        self.ki = ki
        self.vmax = vmax
        self._ix = 0.0
        self._iy = 0.0

    def reset(self):
        self._ix = 0.0
        self._iy = 0.0

    def update(self, ex, ey, dt):
        self._ix = np.clip(self._ix + ex * dt, -0.5, 0.5)
        self._iy = np.clip(self._iy + ey * dt, -0.5, 0.5)

        dvx = self.kp * ex + self.ki * self._ix
        dvy = self.kp * ey + self.ki * self._iy

        dvx = np.clip(dvx, -self.vmax * 0.5, self.vmax)
        dvy = np.clip(dvy, -self.vmax * 0.4, self.vmax * 0.4)

        return dvx, dvy


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def make_gain_dict(kp_val, kd_val):
    kp_dict = {leg: kp_val.copy() for leg in LEG_ORDER}
    kd_dict = {leg: kd_val.copy() for leg in LEG_ORDER}
    return kp_dict, kd_dict


def build_waypoints(args):
    wp0 = {
        "name": "WP0_START",
        "x": 0.0,
        "y": 0.0,
        "yaw": 0.0,
    }

    wp1 = {
        "name": "WP1_FORWARD_TARGET",
        "x": args.vref * args.forward_time,
        "y": 0.0,
        "yaw": 0.0,
    }

    wp2 = {
        "name": "WP2_TURN_180_TARGET",
        "x": args.vref * args.forward_time,
        "y": 0.0,
        "yaw": np.pi,
    }

    wp3 = {
        "name": "WP3_RETURN_TARGET",
        "x": None,
        "y": None,
        "yaw": np.pi,
    }

    return [wp0, wp1, wp2, wp3]


def print_waypoints(waypoints):
    print("\nDefined waypoint trajectory:")
    for wp in waypoints:
        x_txt = f"{wp['x']:.3f}" if wp["x"] is not None else "updated_after_turn"
        y_txt = f"{wp['y']:.3f}" if wp["y"] is not None else "updated_after_turn"
        print(
            f"  {wp['name']}: "
            f"x={x_txt}, y={y_txt}, yaw={wp['yaw']:.3f} rad"
        )


def compute_optimal_torque(
    env,
    controller_name,
    opt_controller,
    u_ref,
    vx_cmd,
    vy_cmd,
    wz_cmd,
    grf_clip,
):
    """
    Computes an optimal-control torque correction from LQG/PMP/MPC.

    The stable gait controller still generates the main walking torques.
    This function only adds a small correction based on the selected
    optimal controller.
    """
    tau_opt = np.zeros(env.mjModel.nu)
    grfs = None
    opt_failed = False

    if opt_controller is None or controller_name == "pi":
        return tau_opt, grfs, opt_failed

    try:
        x_state = get_state(env)
        contact = get_contacts(env)

        x_ref_opt = build_reference_state_from_velocity_guidance(
            x=x_state,
            height=ROBOT_HIP_HEIGHT,
            vx_cmd=vx_cmd,
            vy_cmd=vy_cmd,
            yaw_cmd=x_state[8],
            wz_cmd=wz_cmd,
        )

        if controller_name == "lqg":
            # LQG uses measurement + reference + nominal force.
            grfs = opt_controller.step(x_state.copy(), x_ref_opt, u_ref)
        else:
            # PMP and MPC usually expose compute_control.
            try:
                grfs = opt_controller.compute_control(
                    x=x_state,
                    x_ref=x_ref_opt,
                    u_ref=u_ref,
                )
            except TypeError:
                try:
                    grfs = opt_controller.compute_control(x_state, x_ref_opt, u_ref)
                except TypeError:
                    grfs = opt_controller.compute_control(x_state, x_ref_opt)

        grfs = np.asarray(grfs, dtype=float).reshape(-1)

        if grfs.shape[0] != 12:
            raise ValueError(f"Expected 12 GRF values, got shape {grfs.shape}")

        grfs = np.clip(grfs, -grf_clip, grf_clip)

        # Do not apply contact forces on swing legs.
        for i in range(4):
            if not contact[i]:
                grfs[3 * i:3 * i + 3] = 0.0

        tau_opt = grf_to_torques(env, grfs, contact)

    except Exception as e:
        opt_failed = True
        tau_opt = np.zeros(env.mjModel.nu)
        grfs = None

    return tau_opt, grfs, opt_failed


def main():
    parser = argparse.ArgumentParser(
        description="Waypoint walking demo with optional LQG/PMP/MPC torque assistance"
    )

    parser.add_argument("--robot-name", type=str, default="mini_cheetah")
    parser.add_argument("--duration", type=float, default=52.0)

    # Main controller mode.
    parser.add_argument(
        "--controller",
        type=str,
        default="pi",
        choices=["pi", "lqg", "pmp", "mpc"],
        help="pi = stable walking only; lqg/pmp/mpc = stable walking + optimal torque correction",
    )
    parser.add_argument(
        "--opt-alpha",
        type=float,
        default=0.05,
        help="Blend factor for LQG/PMP/MPC torque correction. Start small: 0.03 to 0.10.",
    )
    parser.add_argument(
        "--grf-clip",
        type=float,
        default=80.0,
        help="Absolute clipping limit for each GRF component before torque mapping.",
    )

    # Forward walking.
    parser.add_argument("--cycle", type=float, default=0.95)
    parser.add_argument("--vref", type=float, default=0.03)

    # 180-degree turn.
    parser.add_argument("--turn-cycle", type=float, default=0.50)
    parser.add_argument("--turn-rate", type=float, default=-1.20)
    parser.add_argument("--turn-duration", type=float, default=11.0)

    # Return walking.
    parser.add_argument("--return-cycle", type=float, default=0.70)
    parser.add_argument("--return-vref", type=float, default=0.055)

    parser.add_argument("--forward-time", type=float, default=18.0)

    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--csv", type=str, default=None)

    parser.add_argument("--kp-pos", type=float, default=0.25)
    parser.add_argument("--ki-pos", type=float, default=0.025)

    args = parser.parse_args()

    if args.csv is None:
        args.csv = f"results/waypoint_walk_{args.controller}.csv"

    if args.controller != "pi" and not HAS_OPT_CONTROL:
        print("\nERROR: Could not import optimal-control helpers from examples/run_mujoco.py")
        print(f"Import error: {OPT_IMPORT_ERROR}")
        print("Run with --controller pi or check run_mujoco.py imports.")
        sys.exit(1)

    env = QuadrupedEnv(robot=args.robot_name, scene="flat", sim_dt=0.002)
    env.reset(random=False)

    if not args.no_render:
        env.render()

    sim_dt = env.mjModel.opt.timestep
    n_steps = int(args.duration / sim_dt)

    q_nom, _ = get_joint_state_dict(env)

    gait_walk = TrotGaitScheduler(period=args.cycle, duty_factor=0.55)
    gait_turn = TrotGaitScheduler(period=args.turn_cycle, duty_factor=0.60)
    gait_return = TrotGaitScheduler(period=args.return_cycle, duty_factor=0.55)

    planner = JointSpaceTrotPlanner(q_nom)

    # Walking gains.
    kp_walk_val = np.array([38.0, 48.0, 48.0])
    kd_walk_val = np.array([2.2, 3.0, 3.0])
    kp_walk_dict, kd_walk_dict = make_gain_dict(kp_walk_val, kd_walk_val)

    # Turn gains.
    kp_turn_val = np.array([70.0, 100.0, 100.0])
    kd_turn_val = np.array([3.5, 5.5, 5.5])
    kp_turn_dict, kd_turn_dict = make_gain_dict(kp_turn_val, kd_turn_val)

    # Return gains.
    kp_return_val = np.array([46.0, 58.0, 58.0])
    kd_return_val = np.array([2.6, 3.4, 3.4])
    kp_return_dict, kd_return_dict = make_gain_dict(kp_return_val, kd_return_val)

    pi_ctrl = PositionPI(kp=args.kp_pos, ki=args.ki_pos, vmax=0.12)

    x0 = float(env.base_pos[0])
    y0 = float(env.base_pos[1])

    waypoints = build_waypoints(args)
    print_waypoints(waypoints)

    # Optional optimal controller setup.
    dyn = None
    Q = R = Q_f = None
    u_ref = None
    opt_controller = None

    if args.controller != "pi":
        print(f"\nInitializing optimal-control layer: {args.controller.upper()}")
        dyn = build_dynamics()
        Q, R, Q_f = build_cost_matrices()
        x_init = get_state(env)

        x_ref_init = build_reference_state_from_velocity_guidance(
            x=x_init,
            height=ROBOT_HIP_HEIGHT,
            vx_cmd=0.0,
            vy_cmd=0.0,
            yaw_cmd=x_init[8],
            wz_cmd=0.0,
        )

        u_ref = dyn.standing_control()
        opt_controller = build_controller(args.controller, dyn, Q, R, Q_f, x_ref_init)

        print(
            f"  {args.controller.upper()} correction enabled | "
            f"opt_alpha={args.opt_alpha:.3f} | grf_clip={args.grf_clip:.1f}"
        )
    else:
        print("\nController mode: PI-only stable waypoint walking")

    x_prev = 0.0
    t_prev = 0.0
    vx_real = 0.0

    log_rows = []

    x_after_turn = None
    y_after_turn = None
    after_turn_t0 = None

    tau_opt_current = np.zeros(env.mjModel.nu)
    opt_failed_count = 0

    # Compute optimal correction at 100 Hz instead of every MuJoCo step.
    ctrl_dt = 0.01
    ctrl_steps = max(1, int(ctrl_dt / sim_dt))

    print(
        f"\n[WAYPOINT WALK CONTROL: WP0 -> WP1 -> WP2 -> WP3] "
        f"Robot: {args.robot_name} | controller={args.controller} | "
        f"walk_cycle={args.cycle:.2f}s | "
        f"turn_cycle={args.turn_cycle:.2f}s | "
        f"return_cycle={args.return_cycle:.2f}s | "
        f"vref={args.vref:.3f} m/s | "
        f"return_vref={args.return_vref:.3f} m/s | "
        f"forward_time={args.forward_time:.1f}s | "
        f"turn_duration={args.turn_duration:.1f}s | "
        f"turn_rate={args.turn_rate:.2f} rad/s"
    )

    try:
        for step in range(n_steps):
            t = step * sim_dt

            q_dict, dq_dict = get_joint_state_dict(env)

            x_real = float(env.base_pos[0] - x0)
            y_real = float(env.base_pos[1] - y0)
            z_real = float(env.base_pos[2])

            turn_start = args.forward_time
            turn_end = args.forward_time + args.turn_duration

            if t < turn_start:
                phase_name = "FORWARD"

                target_wp = waypoints[1]
                target_waypoint = target_wp["name"]
                x_wp = target_wp["x"]
                y_wp = target_wp["y"]
                yaw_wp = target_wp["yaw"]

                x_ref = args.vref * t
                y_ref = 0.0

                vref_now = args.vref
                wzref_now = 0.0

            elif t < turn_end:
                phase_name = "TURN_180"

                target_wp = waypoints[2]
                target_waypoint = target_wp["name"]
                x_wp = target_wp["x"]
                y_wp = target_wp["y"]
                yaw_wp = target_wp["yaw"]

                x_ref = args.vref * args.forward_time
                y_ref = 0.0

                turn_elapsed = t - turn_start
                ramp = smoothstep(turn_elapsed / 1.0)

                vref_now = 0.0
                wzref_now = args.turn_rate * ramp

            else:
                phase_name = "FORWARD_AFTER_TURN"

                target_wp = waypoints[3]
                target_waypoint = target_wp["name"]

                if x_after_turn is None:
                    x_after_turn = x_real
                    y_after_turn = y_real
                    after_turn_t0 = t
                    pi_ctrl.reset()

                    remaining_time = max(0.0, args.duration - t)
                    waypoints[3]["x"] = x_after_turn - args.vref * remaining_time
                    waypoints[3]["y"] = y_after_turn

                    print(
                        f"\nWaypoint real después del giro guardado: "
                        f"x={x_after_turn:.3f}, y={y_after_turn:.3f}"
                    )
                    print(
                        f"Updated WP3_RETURN_TARGET: "
                        f"x={waypoints[3]['x']:.3f}, "
                        f"y={waypoints[3]['y']:.3f}, "
                        f"yaw={waypoints[3]['yaw']:.3f} rad\n"
                    )

                x_wp = waypoints[3]["x"]
                y_wp = waypoints[3]["y"]
                yaw_wp = waypoints[3]["yaw"]

                after_turn_time = t - after_turn_t0

                x_ref = x_after_turn - args.vref * after_turn_time
                y_ref = y_after_turn

                ramp_after = smoothstep(after_turn_time / 1.2)

                vref_now = -args.return_vref * ramp_after
                wzref_now = 0.0

            ex = x_ref - x_real
            ey = y_ref - y_real

            dvx, dvy = pi_ctrl.update(ex, ey, sim_dt)

            if phase_name == "FORWARD":
                vx_cmd_eff = float(np.clip(vref_now + dvx, 0.0, 0.12))
                vy_cmd_eff = float(np.clip(dvy, -0.05, 0.05))
                wz_cmd_eff = 0.0

                gait = gait_walk
                gait_t = t

                kp_dict = kp_walk_dict
                kd_dict = kd_walk_dict
                torque_limit = 14.0

            elif phase_name == "TURN_180":
                vx_cmd_eff = 0.0
                vy_cmd_eff = 0.0
                wz_cmd_eff = float(wzref_now)

                gait = gait_turn
                gait_t = t - turn_start

                kp_dict = kp_turn_dict
                kd_dict = kd_turn_dict

                turn_elapsed = t - turn_start
                torque_limit = 12.0 + 6.0 * smoothstep(turn_elapsed / 2.0)

            else:
                after_turn_time = max(0.0, t - after_turn_t0)
                ramp_after = smoothstep(after_turn_time / 1.2)

                vx_cmd_eff = float(
                    -np.clip(args.return_vref * ramp_after, 0.0, args.return_vref)
                )

                vy_cmd_eff = 0.0
                wz_cmd_eff = 0.0

                gait = gait_return
                gait_t = after_turn_time

                kp_dict = kp_return_dict
                kd_dict = kd_return_dict
                torque_limit = 18.0

            q_des_dict = planner.get_joint_targets(
                t=gait_t,
                gait=gait,
                vx_cmd=vx_cmd_eff,
                vy_cmd=vy_cmd_eff,
                wz_cmd=wz_cmd_eff,
            )

            tau_gait = pd_leg_torque(env, q_des_dict, q_dict, dq_dict, kp_dict, kd_dict)

            opt_failed = False
            if args.controller != "pi" and step % ctrl_steps == 0:
                tau_opt_current, _, opt_failed = compute_optimal_torque(
                    env=env,
                    controller_name=args.controller,
                    opt_controller=opt_controller,
                    u_ref=u_ref,
                    vx_cmd=vx_cmd_eff,
                    vy_cmd=vy_cmd_eff,
                    wz_cmd=wz_cmd_eff,
                    grf_clip=args.grf_clip,
                )
                if opt_failed:
                    opt_failed_count += 1

            tau = tau_gait + args.opt_alpha * tau_opt_current
            tau = np.clip(tau, -torque_limit, torque_limit)

            env.step(action=tau)

            if (not args.no_render) and step % 15 == 0:
                env.render()

            if step > 0:
                dt_step = t - t_prev
                vx_inst = (x_real - x_prev) / max(dt_step, 1e-9)
                vx_real = 0.95 * vx_real + 0.05 * vx_inst

            x_prev = x_real
            t_prev = t

            e_pos = float(np.sqrt(ex**2 + ey**2))
            active_swing = [leg for leg in LEG_ORDER if gait.is_swing(leg, gait_t)]

            tau_gait_norm = float(np.linalg.norm(tau_gait))
            tau_opt_norm = float(np.linalg.norm(tau_opt_current))
            tau_total_norm = float(np.linalg.norm(tau))

            log_rows.append({
                "t": t,
                "controller": args.controller,
                "opt_alpha": args.opt_alpha,
                "phase": phase_name,
                "target_waypoint": target_waypoint,
                "x_wp": x_wp,
                "y_wp": y_wp,
                "yaw_wp": yaw_wp,
                "x_ref": x_ref,
                "y_ref": y_ref,
                "x_real": x_real,
                "y_real": y_real,
                "z_real": z_real,
                "ex": ex,
                "ey": ey,
                "e_pos": e_pos,
                "vx_real": vx_real,
                "vx_ref": vref_now,
                "vx_cmd_eff": vx_cmd_eff,
                "vy_cmd_eff": vy_cmd_eff,
                "wz_cmd_eff": wz_cmd_eff,
                "tau_gait_norm": tau_gait_norm,
                "tau_opt_norm": tau_opt_norm,
                "tau_total_norm": tau_total_norm,
                "opt_failed": int(opt_failed),
            })

            if step % int(1.0 / sim_dt) == 0:
                print(
                    f"t={t:4.1f}s | "
                    f"{phase_name:18s} -> {target_waypoint:22s} | "
                    f"Swing:{'+'.join(active_swing) or '-':9s} | "
                    f"x_ref={x_ref:7.3f} y_ref={y_ref:7.3f} | "
                    f"x_real={x_real:7.3f} y_real={y_real:7.3f} | "
                    f"err={e_pos:6.3f} | "
                    f"vx_eff={vx_cmd_eff:+.3f} "
                    f"vy_eff={vy_cmd_eff:+.3f} "
                    f"wz_eff={wz_cmd_eff:+.3f} | "
                    f"tau_opt={tau_opt_norm:7.2f} | "
                    f"z={z_real:.3f}"
                )

    except KeyboardInterrupt:
        print("\nInterrumpido por usuario.")

    finally:
        try:
            save_csv(args.csv, log_rows)
        except Exception as e:
            print(f"No se pudo guardar CSV: {e}")

        if log_rows:
            x_final_ref = log_rows[-1]["x_ref"]
            y_final_ref = log_rows[-1]["y_ref"]
            x_final_real = log_rows[-1]["x_real"]
            y_final_real = log_rows[-1]["y_real"]
            e_final = log_rows[-1]["e_pos"]

            rmse = np.sqrt(np.mean([r["e_pos"] ** 2 for r in log_rows]))

            dist_real = np.sqrt(
                (log_rows[-1]["x_real"] - log_rows[0]["x_real"]) ** 2 +
                (log_rows[-1]["y_real"] - log_rows[0]["y_real"]) ** 2
            )

            drift_y_max = max(abs(r["y_real"]) for r in log_rows)
            vx_promedio = np.mean([r["vx_real"] for r in log_rows[50:]])
            mean_tau_gait = np.mean([r["tau_gait_norm"] for r in log_rows])
            mean_tau_opt = np.mean([r["tau_opt_norm"] for r in log_rows])
            mean_tau_total = np.mean([r["tau_total_norm"] for r in log_rows])

            print("\n=== MÉTRICAS ===")
            print(f"Controller mode         : {args.controller}")
            print(f"Optimal blend alpha     : {args.opt_alpha:.4f}")
            print(f"Referencia final X      : {x_final_ref:.4f} m")
            print(f"Referencia final Y      : {y_final_ref:.4f} m")
            print(f"Posición real final X   : {x_final_real:.4f} m")
            print(f"Posición real final Y   : {y_final_real:.4f} m")
            print(f"Distancia real recorrida: {dist_real:.4f} m")
            print(f"Error final de posición : {e_final:.4f} m")
            print(f"RMSE de trayectoria     : {rmse:.4f} m")
            print(f"Velocidad real promedio : {vx_promedio:.4f} m/s")
            print(f"Drift lateral máximo    : {drift_y_max:.4f} m")
            print(f"Mean tau gait norm      : {mean_tau_gait:.4f}")
            print(f"Mean tau opt norm       : {mean_tau_opt:.4f}")
            print(f"Mean tau total norm     : {mean_tau_total:.4f}")
            print(f"Optimal failures        : {opt_failed_count}")
            print(f"CSV guardado en         : {args.csv}")

        try:
            env.close()
        except Exception:
            pass
        finally:
            os._exit(0)


if __name__ == "__main__":
    main()
