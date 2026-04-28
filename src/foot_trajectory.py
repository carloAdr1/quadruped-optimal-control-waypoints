import numpy as np
from src.gait_scheduler import LEG_ORDER


class JointSpaceTrotPlanner:
    def __init__(self, q_nominal_dict):
        self.q_nom = {
            leg: np.array(q_nominal_dict[leg], dtype=np.float32).copy()
            for leg in LEG_ORDER
        }

        self.side_sign = {
            "FL": +1.0,
            "RL": +1.0,
            "FR": -1.0,
            "RR": -1.0,
        }

    @staticmethod
    def _swing_knee_profile(s):
        if s < 0.15:
            return np.sin(np.pi * s / 0.15) * 0.3
        return np.sin(np.pi * s)

    def _walk_targets(self, t, gait, vx_cmd, vy_cmd, wz_cmd):
        """
        Modo caminata.
        Ahora soporta vx positivo y vx negativo.
        vx positivo = caminar hacia adelante.
        vx negativo = caminar hacia atrás / regreso.
        """
        vmag = np.sqrt(vx_cmd**2 + vy_cmd**2)

        # Signo del avance.
        # Esto invierte el barrido de cadera cuando queremos regresar.
        walk_sign = 1.0 if vx_cmd >= 0.0 else -1.0

        hip_amp = np.clip(0.17 + 2.3 * vmag, 0.17, 0.36)
        knee_amp = np.clip(0.16 + 2.5 * vmag, 0.16, 0.32)
        ab_amp = np.clip(0.012 + 0.12 * abs(wz_cmd), 0.008, 0.065)

        turn_amp = np.clip(0.24 * abs(wz_cmd), 0.0, 0.14)
        turn_sign = np.sign(wz_cmd)

        q_des = {}

        for leg in LEG_ORDER:
            q0 = self.q_nom[leg].copy()
            hip_dir = -np.sign(q0[1]) if abs(q0[1]) > 1e-6 else 1.0
            is_left = leg in ["FL", "RL"]

            if gait.is_swing(leg, t):
                s = gait.swing_phase(leg, t)

                q_haa = q0[0] + (
                    self.side_sign[leg]
                    * ab_amp
                    * np.sin(np.pi * s)
                    * np.sign(wz_cmd + 1e-9)
                )

                # Caminata normal, pero con signo para permitir regreso.
                q_hfe = q0[1] + hip_dir * hip_amp * walk_sign * (0.78 - 1.56 * s)

                if abs(wz_cmd) > 1e-3:
                    if is_left:
                        q_hfe += turn_sign * turn_amp
                    else:
                        q_hfe -= turn_sign * turn_amp

                q_kfe = q0[2] + knee_amp * self._swing_knee_profile(s)

            else:
                s = gait.stance_phase(leg, t)

                q_haa = q0[0] + (
                    self.side_sign[leg]
                    * 0.3
                    * ab_amp
                    * (
                        vy_cmd / max(abs(vy_cmd), 0.01)
                        if abs(vy_cmd) > 1e-3
                        else 0.0
                    )
                    * (1.0 - s)
                )

                # Stance también se invierte cuando vx_cmd es negativo.
                q_hfe = q0[1] + hip_dir * hip_amp * walk_sign * (0.5 - s)

                if abs(wz_cmd) > 1e-3:
                    if is_left:
                        q_hfe -= turn_sign * turn_amp
                    else:
                        q_hfe += turn_sign * turn_amp

                toe_off = knee_amp * 0.12 * max(0.0, (s - 0.80) / 0.20)
                q_kfe = q0[2] + 0.06 * knee_amp * np.cos(np.pi * s) + toe_off

            q_des[leg] = np.array([q_haa, q_hfe, q_kfe], dtype=np.float32)

        return q_des

    def _turn_targets(self, t, gait, vx_cmd, vy_cmd, wz_cmd):
        """
        Modo giro: basado en el código que gira bien con:
        cycle=0.50, vref=0.0, turn-rate=-1.2.
        """
        q_des = {}

        h_amp = 0.25
        k_amp = 0.32
        a_amp = 0.18

        turn_sign = np.sign(wz_cmd + 1e-9)

        for leg in LEG_ORDER:
            q0 = self.q_nom[leg].copy()

            is_swing = gait.is_swing(leg, t)
            s = gait.swing_phase(leg, t) if is_swing else gait.stance_phase(leg, t)

            # HAA: base ancha para giro estable.
            q_haa = q0[0] + (
                self.side_sign[leg]
                * a_amp
                * np.sin(np.pi * s)
                * turn_sign
            )

            hip_dir = -np.sign(q0[1]) if abs(q0[1]) > 1e-6 else 1.0

            # Patrón diagonal para giro: FL+RR contra FR+RL.
            turn_dir = 1.0 if leg in ["FL", "RR"] else -1.0

            if is_swing:
                sweep = -turn_dir * h_amp * (1.0 - 2.0 * s)
                q_hfe = q0[1] + hip_dir * sweep
                q_kfe = q0[2] + k_amp * np.sin(np.pi * s)
            else:
                sweep = turn_dir * h_amp * (0.8 - 1.6 * s)
                q_hfe = q0[1] + hip_dir * sweep
                q_kfe = q0[2] + 0.18 * np.sin(np.pi * s)

            q_des[leg] = np.array([q_haa, q_hfe, q_kfe], dtype=np.float32)

        return q_des

    def get_joint_targets(self, t, gait, vx_cmd, vy_cmd, wz_cmd):
        is_turning = abs(wz_cmd) > 0.1

        if is_turning:
            return self._turn_targets(t, gait, vx_cmd, vy_cmd, wz_cmd)

        return self._walk_targets(t, gait, vx_cmd, vy_cmd, wz_cmd)