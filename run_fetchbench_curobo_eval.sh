#!/usr/bin/env bash
set -euo pipefail

# Wrapper to run the original FetchBench CuRobo evaluator as-is.
# It executes:
#   python isaacgymenvs/eval.py task=FetchMeshCurobo scene=<scene-config>
# in the FetchBench conda environment.
#
# Usage:
#   ./corl2025/run_fetchbench_curobo_eval.sh benchmark_eval/RigidObjDesk_0
#   ./corl2025/run_fetchbench_curobo_eval.sh benchmark_eval/RigidObjDesk_0 --headless=True
#
# Environment overrides:
#   CONDA_ENV   (default: FetchBench)
#   CORL_ROOT   (default: /media/shoon/DISK1/seunghoonjeong/1_research/01_corl2025/FetchBench-CORL2024)

SCENE_CONFIG="${1:-benchmark_eval/RigidObjDesk_0}"
shift || true

CONDA_ENV="${CONDA_ENV:-FetchBench}"
CORL_ROOT="${CORL_ROOT:-/media/shoon/DISK1/seunghoonjeong/1_research/01_corl2025/FetchBench-CORL2024}"
INFINIGYM_DIR="${CORL_ROOT}/InfiniGym"

if [[ ! -d "${INFINIGYM_DIR}" ]]; then
  echo "[ERROR] InfiniGym directory not found: ${INFINIGYM_DIR}" >&2
  exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda command not found. Activate conda first or add conda to PATH." >&2
  exit 1
fi

cd "${INFINIGYM_DIR}"

echo "[INFO] Running FetchMeshCurobo in env=${CONDA_ENV}, scene=${SCENE_CONFIG}"
# scene.num_tasks=1 makes eval loop run only task 0 (reset_task(0) once).
conda run -n "${CONDA_ENV}" python isaacgymenvs/eval.py \
  task=FetchMeshCurobo \
  scene="${SCENE_CONFIG}" \
  scene.num_tasks=1 \
  "$@"
