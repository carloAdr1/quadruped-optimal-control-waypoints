#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

mkdir -p results

DURATION=10
SPEED=0.03
TRAJECTORY=line
DISTURBANCE=none

echo "Running waypoint trajectory controller comparison"
echo "Trajectory: ${TRAJECTORY}"
echo "Duration: ${DURATION}s"
echo "Speed: ${SPEED} m/s"
echo "Disturbance: ${DISTURBANCE}"

for CTRL in lqg pmp mpc
do
    echo ""
    echo "========================================"
    echo "Controller: ${CTRL}"
    echo "========================================"

    python examples/run_mujoco.py \
        --controller "${CTRL}" \
        --trajectory "${TRAJECTORY}" \
        --duration "${DURATION}" \
        --waypoint-speed "${SPEED}" \
        --disturbance "${DISTURBANCE}" \
        --no-render
done

echo ""
echo "Experiments completed."
echo "Metrics saved in results/metrics_*_${TRAJECTORY}_${DISTURBANCE}.csv"
