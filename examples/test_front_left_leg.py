#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gym_quadruped.quadruped_env import QuadrupedEnv


LEG_ORDER = ["FL", "FR", "RL", "RR"]


def get_joint_state_dict(env):
    qj = env.mjData.qpos[-12:].copy().reshape(4, 3)
    dqj = env.mjData.qvel[-12:].copy().reshape(4, 3)
    q_dict = {leg: qj[i].copy() for i, leg in enumerate(LEG_ORDER)}
    dq_dict = {leg: dqj[i].copy() for i, leg in enumerate(LEG_ORDER)}
    return q_dict, dq_dict


def get_feet_pos_base(env):
    fp = env.feet_pos(frame="base")
    return {
        "FL": np.array(fp.FL, dtype=float),
        "FR": np.array(fp.FR, dtype=float),
        "RL": np.array(fp.RL, dtype=float),
        "RR": np.array(fp.RR, dtype=float),
    }


def pd_leg_torque(env, q_des_dict, q_dict, dq_dict, kp_dict, kd_dict):
    tau = np.zeros(env.mjModel.nu)

    for leg in LEG_ORDER:
        q = q_dict[leg]
        dq = dq_dict[leg]
        q_des = q_des_dict[leg]

        kp = kp_dict[leg]
        kd = kd_dict[leg]

        tau_leg = kp * (q_des - q) - kd * dq
        tau_idx = env.legs_tau_idx[leg]
        tau[tau_idx] = tau_leg

    return tau


def lerp(a, b, s):
    return (1.0 - s) * a + s * b


def fl_leg_sequence_from_diagnostics(q0, t, cycle=8.0):
    """
    Secuencia basada en lo observado en tus pruebas:
    - KFE positivo -> ayuda a recoger/subir
    - HFE positivo -> mueve hacia adelante
    - HAA fijo
    """
    phase = (t % cycle) / cycle

    # q = [HAA, HFE, KFE]
    q_home = q0.copy()

    # recoger / subir un poco
    q_up = q0 + np.array([0.00, 0.00, +0.08])

    # avanzar manteniendo la rodilla algo recogida
    q_fwd = q0 + np.array([0.00, +0.08, +0.06])

    # volver apoyando
    q_down = q0 + np.array([0.00, +0.02, +0.00])

    if phase < 0.25:
        s = phase / 0.25
        q = lerp(q_home, q_up, s)
    elif phase < 0.50:
        s = (phase - 0.25) / 0.25
        q = lerp(q_up, q_fwd, s)
    elif phase < 0.75:
        s = (phase - 0.50) / 0.25
        q = lerp(q_fwd, q_down, s)
    else:
        s = (phase - 0.75) / 0.25
        q = lerp(q_down, q_home, s)

    return q


def main():
    parser = argparse.ArgumentParser(description="FL single-leg test from diagnostics")
    parser.add_argument("--robot-name", type=str, default="mini_cheetah")
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    env = QuadrupedEnv(
        robot=args.robot_name,
        scene="flat",
        sim_dt=0.002,
        base_vel_command_type="human",
        state_obs_names=tuple(QuadrupedEnv.ALL_OBS),
    )

    _ = env.reset(random=False)

    if not args.no_render:
        env.render()

    sim_dt = env.mjModel.opt.timestep
    n_steps = int(args.duration / sim_dt)

    q_nom, _ = get_joint_state_dict(env)
    feet0 = get_feet_pos_base(env)

    print("\n=== FL Single-Leg Test (diagnostics-based) ===")
    print(f"Robot: {args.robot_name}")
    print(f"Duration: {args.duration:.1f}s")
    print("FL should: lift a bit with KFE, then move forward with HFE, then return.\n")
    print(f"q0_FL = {np.round(q_nom['FL'], 4)}")
    print(f"p0_FL = {np.round(feet0['FL'], 4)}\n")

    kp_dict = {
        "FL": np.array([8.0, 10.0, 10.0]),
        "FR": np.array([44.0, 50.0, 44.0]),
        "RL": np.array([44.0, 50.0, 44.0]),
        "RR": np.array([44.0, 50.0, 44.0]),
    }
    kd_dict = {
        "FL": np.array([0.4, 0.5, 0.5]),
        "FR": np.array([2.0, 2.4, 2.0]),
        "RL": np.array([2.0, 2.4, 2.0]),
        "RR": np.array([2.0, 2.4, 2.0]),
    }

    try:
        for step in range(n_steps):
            t = step * sim_dt

            q_dict, dq_dict = get_joint_state_dict(env)
            feet = get_feet_pos_base(env)

            q_des_dict = {leg: q_nom[leg].copy() for leg in LEG_ORDER}
            q_des_dict["FL"] = fl_leg_sequence_from_diagnostics(q_nom["FL"], t, cycle=8.0)

            tau = pd_leg_torque(
                env,
                q_des_dict=q_des_dict,
                q_dict=q_dict,
                dq_dict=dq_dict,
                kp_dict=kp_dict,
                kd_dict=kd_dict,
            )

            tau = np.clip(tau, -5.0, 5.0)

            _, _, terminated, _, _ = env.step(action=tau)

            if not args.no_render:
                env.render()

            if step % int(0.5 / sim_dt) == 0:
                print(
                    f"t={t:4.1f}s | "
                    f"z={env.base_pos[2]:.3f} | "
                    f"q_FL={np.round(q_dict['FL'], 3)} | "
                    f"q_FL_des={np.round(q_des_dict['FL'], 3)} | "
                    f"p_FL={np.round(feet['FL'], 3)}"
                )

            if terminated:
                print(f"Terminated at t={t:.2f}s")
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
