# Control Óptimo y Seguimiento de Waypoints en MuJoCo

Este repositorio contiene mi implementación para la tarea **"Applying Optimal Control in MuJoCo"**.

El objetivo de la tarea fue implementar generación de trayectorias basadas en waypoints para que un robot cuadrúpedo en simulación pueda seguir caminos predefinidos. Además, se pidió probar diferentes controladores, comparar su comportamiento y publicar tanto la implementación como los resultados en GitHub.

Este proyecto parte del repositorio base proporcionado por el profesor, pero fue extendido con una demo estable de caminata por waypoints, generación de marcha tipo trot, control PI/PD, integración de controladores LQG/PMP/MPC, registro de métricas en CSV y comparación de resultados.

---

## 1. Qué pidió la tarea

La tarea pidió lo siguiente:

> Implementar generación adecuada de trayectorias basadas en waypoints para que el robot en simulación pueda seguir caminos predefinidos con precisión. Probar el desempeño de diferentes controladores en esas trayectorias, comparar su comportamiento y publicar la implementación y los resultados en GitHub.

Criterios de evaluación:

| Criterio | Descripción | Puntos |
|---|---|---:|
| Implementación de trayectoria por waypoints | Genera y sigue correctamente trayectorias basadas en waypoints | 10 |
| Integración y funcionamiento de controladores | Los controladores están adaptados, ejecutados y probados dentro de la simulación | 10 |
| Evaluación y comparación de desempeño | Incluye métricas claras, análisis y comparación del seguimiento de trayectoria | 10 |
| Documentación y presentación en GitHub | Código, resultados y explicaciones organizadas claramente | 10 |
| Total | | 40 |

---

## 2. Qué implementé

Implementé una simulación de caminata por waypoints para el robot Mini Cheetah en MuJoCo.

La trayectoria principal está definida por los siguientes waypoints:

| Waypoint | Descripción | Objetivo |
|---|---|---|
| `WP0_START` | Pose inicial del robot | Iniciar desde la posición actual |
| `WP1_FORWARD_TARGET` | Avance hacia enfrente | Caminar en dirección X |
| `WP2_TURN_180_TARGET` | Giro en el mismo lugar | Girar aproximadamente 180 grados |
| `WP3_RETURN_TARGET` | Trayectoria de regreso | Caminar de regreso después del giro |

El comportamiento implementado es:

```
WP0_START -> WP1_FORWARD_TARGET -> WP2_TURN_180_TARGET -> WP3_RETURN_TARGET
```

El robot primero avanza, después realiza un giro de 180 grados y finalmente camina de regreso. Para el waypoint de regreso, se usa la posición real medida después del giro, de manera que la trayectoria de retorno se adapta a la posición alcanzada por el robot.

---

## 3. Archivos agregados o modificados

### Demo estable de caminata por waypoints — `examples/waypoint_walk_demo.py`

Este archivo es la demo estable principal. Usa:

- waypoints explícitos
- gait scheduler tipo trot
- generación de trayectorias articulares
- control PD en articulaciones
- corrección PI de posición
- registro en CSV
- métricas finales de seguimiento

Esta demo muestra que el robot puede completar una trayectoria:

```
avanzar -> girar 180 grados -> regresar
```

### Demo con comparación de controladores PI/LQG/PMP/MPC — `examples/waypoint_walk_control.py`

Este archivo compara diferentes modos de control sobre la misma trayectoria por waypoints.

Modos disponibles: `pi`, `lqg`, `pmp`, `mpc`

El modo `pi` usa solamente la caminata estable con control PI/PD.

Los modos `lqg`, `pmp` y `mpc` usan la misma caminata estable, pero agregan una pequeña corrección de torque calculada por el controlador óptimo correspondiente.

La estructura general es:

```
waypoints
  -> comandos de velocidad
  -> gait planner tipo trot
  -> torque PD articular
  -> corrección opcional LQG/PMP/MPC
  -> simulación en MuJoCo
```

Los controladores LQG/PMP/MPC no reemplazan la marcha base. Se integran como una capa de asistencia sobre la caminata estable.

### Comparación formal de controladores — `examples/run_mujoco.py`

Este archivo mantiene la comparación formal del framework original, pero fue extendido para guardar métricas en CSV.

Soporta los controladores: `LQG`, `PMP`, `MPC`

Y las trayectorias: `line`, `square`, `zigzag`

### Archivos de soporte

- `src/trajectory_generator.py`
- `src/gait_scheduler.py`
- `src/foot_trajectory.py`
- `src/leg_ik.py`

Estos archivos apoyan la generación de trayectorias, planificación de marcha tipo trot, generación de objetivos articulares y cinemática de las patas.

### Scripts de experimentos

- `run_experiments.sh` — ejecuta la comparación formal con LQG, PMP y MPC
- `summarize_results.py` — combina los archivos CSV de métricas en una tabla resumen

---

## 4. Cómo correr la demo waypoint estable

Desde la raíz del repositorio:

```bash
cd ~/quadruped-optimal-control
```

Ejecutar:

```bash
python examples/waypoint_walk_demo.py \
  --duration 52 \
  --cycle 0.95 \
  --turn-cycle 0.50 \
  --return-cycle 0.70 \
  --vref 0.03 \
  --return-vref 0.055 \
  --forward-time 18 \
  --turn-duration 11.0 \
  --turn-rate -1.2 \
  --csv results/waypoint_walk_demo.csv \
  --no-render
```

Este comando genera: `results/waypoint_walk_demo.csv`

Columnas importantes del CSV:

| Columna | Significado |
|---|---|
| `phase` | Fase actual del movimiento |
| `target_waypoint` | Waypoint activo |
| `x_wp, y_wp, yaw_wp` | Valores objetivo del waypoint |
| `x_ref, y_ref` | Referencia de trayectoria |
| `x_real, y_real` | Posición real del robot |
| `e_pos` | Error de seguimiento XY |
| `vx_cmd_eff, vy_cmd_eff, wz_cmd_eff` | Comandos efectivos de velocidad |

---

## 5. Cómo correr la demo con PI/LQG/PMP/MPC

### PI base

```bash
python examples/waypoint_walk_control.py \
  --controller pi \
  --csv results/waypoint_control_pi.csv \
  --no-render
```

### LQG asistido

```bash
python examples/waypoint_walk_control.py \
  --controller lqg \
  --opt-alpha 0.03 \
  --csv results/waypoint_control_lqg.csv \
  --no-render
```

### PMP asistido

```bash
python examples/waypoint_walk_control.py \
  --controller pmp \
  --opt-alpha 0.03 \
  --csv results/waypoint_control_pmp.csv \
  --no-render
```

### MPC asistido

```bash
python examples/waypoint_walk_control.py \
  --controller mpc \
  --opt-alpha 0.03 \
  --csv results/waypoint_control_mpc.csv \
  --no-render
```

---

## 6. Cómo medir los controladores antes de evaluarlos

La demo `waypoint_walk_control.py` guarda columnas adicionales para comprobar que los controladores se están aplicando correctamente.

| Columna | Significado |
|---|---|
| `controller` | Controlador usado: `pi`, `lqg`, `pmp` o `mpc` |
| `opt_alpha` | Factor de mezcla para la corrección óptima |
| `tau_gait_norm` | Norma del torque generado por la marcha base |
| `tau_opt_norm` | Norma de la corrección generada por LQG/PMP/MPC |
| `tau_total_norm` | Norma del torque total aplicado |
| `opt_failed` | Indica si el controlador óptimo falló en ese instante |

Comportamiento esperado:

| Controlador | `tau_opt_norm` esperado | `opt_failed` esperado |
|---|---|---|
| PI | 0 | 0 |
| LQG | Mayor que 0 | 0 o muy bajo |
| PMP | Mayor que 0 | 0 o muy bajo |
| MPC | Mayor que 0 | 0 o muy bajo |

En mis pruebas finales, los controladores asistidos se ejecutaron sin fallas de optimización.

---

## 7. Cómo correr la comparación formal LQG/PMP/MPC

```bash
python examples/run_mujoco.py \
  --controller lqg \
  --trajectory line \
  --duration 10 \
  --waypoint-speed 0.03 \
  --no-render

python examples/run_mujoco.py \
  --controller pmp \
  --trajectory line \
  --duration 10 \
  --waypoint-speed 0.03 \
  --no-render

python examples/run_mujoco.py \
  --controller mpc \
  --trajectory line \
  --duration 10 \
  --waypoint-speed 0.03 \
  --no-render
```

También se pueden correr los tres controladores con:

```bash
./run_experiments.sh
```

Y resumir los resultados con:

```bash
python summarize_results.py
```

Esto genera:
- `results/summary_metrics.csv`
- `results/summary_metrics.md`

---

## 8. Resultados

### Demo waypoint estable

La demo estable completó la trayectoria de avance, giro y regreso.

| Métrica | Resultado |
|---|---|
| Referencia final X | -0.5268 m |
| Referencia final Y | -0.0814 m |
| Posición real final X | -0.5658 m |
| Posición real final Y | 0.0045 m |
| Error final de posición | 0.0943 m |
| RMSE de trayectoria | 0.1349 m |
| Drift lateral máximo | 0.2501 m |
| Distancia real recorrida | 0.5658 m |

El robot completó la trayectoria con un error final menor a 10 cm.

### Comparación PI/LQG/PMP/MPC en la demo waypoint

| Controlador | RMSE trayectoria | Error final | Drift lateral máximo | Media tau_opt_norm | Fallas óptimas |
|---|---|---|---|---|---|
| PI | 0.1349 m | 0.0943 m | 0.2501 m | 0.0000 | 0 |
| LQG asistido | 0.1370 m | 0.1479 m | 0.2521 m | 4.8331 | 0 |
| PMP asistido | 0.1925 m | 0.2995 m | 0.2295 m | 17.4013 | 0 |
| MPC asistido | 0.1323 m | 0.1093 m | 0.2437 m | 7.2410 | 0 |

Observaciones:

- PI obtuvo el menor error final.
- MPC asistido obtuvo el menor RMSE de trayectoria.
- PMP generó las correcciones de torque más grandes y empeoró el error final.
- LQG fue estable, pero no mejoró el desempeño de PI.
- Ningún controlador asistido tuvo fallas de optimización.
- La parte más difícil fue el giro de 180 grados, donde apareció mayor drift lateral.

### Comparación formal usando `run_mujoco.py`

| Controlador | Trayectoria | Velocity RMSE | XY RMSE | XY Max Error | Final XY Error | Mean GRF Norm | Resets |
|---|---|---|---|---|---|---|---|
| PMP | line | 0.4208 | 0.1523 m | 0.5759 m | 0.1563 m | 130.8950 N | 3 |
| MPC | line | 0.4028 | 0.1603 m | 0.5501 m | 0.1784 m | 58.2157 N | 3 |
| LQG | line | 0.1968 | 0.2102 m | 0.5557 m | 0.5557 m | 29.5566 N | 0 |

Observaciones:

- PMP tuvo el menor XY RMSE en la comparación formal, pero requirió el mayor esfuerzo de control y causó resets.
- MPC tuvo un seguimiento parecido a PMP, pero con menor norma media de fuerza.
- LQG fue el más estable en términos de resets, pero tuvo el mayor error final.
- Esto muestra que menor error promedio no siempre significa mayor estabilidad.

---

## 9. Discusión

El proyecto incluye dos modos complementarios de evaluación.

El primer modo, `waypoint_walk_demo.py`, se enfoca en una caminata estable por waypoints. Esta demo demuestra que el robot puede completar una trayectoria de avance, giro y regreso usando referencias explícitas.

El segundo modo, `waypoint_walk_control.py`, prueba PI, LQG, PMP y MPC sobre la misma trayectoria. En este caso, LQG/PMP/MPC se integran como capas pequeñas de corrección de torque sobre la marcha estable.

El experimento formal con `run_mujoco.py` se conserva como comparación adicional de los controladores originales LQG, PMP y MPC.

En general, PI obtuvo el menor error final, mientras que MPC asistido obtuvo el menor RMSE de trayectoria.

---

## 10. Conclusión

Este repositorio implementa una tarea de locomoción por waypoints para un robot cuadrúpedo en MuJoCo.

La implementación incluye:

- Definición explícita de waypoints
- Caminata estable tipo trot
- Control PD articular
- Corrección PI de posición
- Asistencia con LQG/PMP/MPC
- Generación de CSV
- Métricas de seguimiento
- Comparación formal de controladores
- Documentación en GitHub

El robot logró completar la trayectoria con un error final de posición de **0.0943 m**. Además, el modo MPC asistido logró el menor RMSE de trayectoria con **0.1323 m**.
