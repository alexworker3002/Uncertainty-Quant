#!/usr/bin/env bash
set -euo pipefail

# Reproducible build helper for cpp/dmt_implicit pybind extension.
# Usage:
#   bash scripts/run/build_dmt_implicit.sh
# Optional env:
#   CONDA_ENV=py312
#   CMAKE_BIN=/usr/bin/cmake

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONDA_ENV="${CONDA_ENV:-py312}"
CMAKE_BIN="${CMAKE_BIN:-/usr/bin/cmake}"
BUILD_DIR="${ROOT_DIR}/cpp/dmt_implicit/build"

if ! command -v /usr/local/miniconda3/bin/conda >/dev/null 2>&1; then
  echo "[ERROR] conda not found at /usr/local/miniconda3/bin/conda"
  exit 1
fi

if [ ! -x "${CMAKE_BIN}" ]; then
  echo "[ERROR] CMake not found at ${CMAKE_BIN}"
  echo "Hint: set CMAKE_BIN=/path/to/cmake"
  exit 1
fi

echo "[INFO] ROOT_DIR=${ROOT_DIR}"
echo "[INFO] CONDA_ENV=${CONDA_ENV}"
echo "[INFO] CMAKE_BIN=${CMAKE_BIN}"

/usr/local/miniconda3/bin/conda run -n "${CONDA_ENV}" bash -lc "
set -euo pipefail
cd '${ROOT_DIR}'
PYBIND11_CMAKE_DIR=\$(python -c 'import pybind11; print(pybind11.get_cmake_dir())')
'${CMAKE_BIN}' -S cpp/dmt_implicit -B '${BUILD_DIR}' -Dpybind11_DIR=\"\${PYBIND11_CMAKE_DIR}\"
'${CMAKE_BIN}' --build '${BUILD_DIR}' -j
"

echo "[OK] Build finished."
ls -lh "${BUILD_DIR}"/dmt_implicit_ext*.so
