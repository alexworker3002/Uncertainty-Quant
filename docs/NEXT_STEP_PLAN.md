# Next-Step Execution Plan (Agent Handoff)

Last updated: 2026-03-24
Owner: Ice + Cursor agents

## 1) Current project status

### Completed in this round
- Added 3D-ready cubical filtration scaffold in C++:
  - `build_filtration_3d_upper_star(...)` with upper-star min-rule + deterministic tie-breaking.
  - 3D cell blocks exposed: vertex / edge_x / edge_y / edge_z / face_xy / face_xz / face_yz / cube.
- Added 3D gradient storage scaffold:
  - `build_gradient_field_3d_scaffold(...)` with `v2e/e2v/e2f/f2e/f2c/c2f` map allocation.
- Added pybind debug APIs for staged verification:
  - `debug_build_filtration_3d`, `debug_build_gradient_3d`
  - existing 2D debug APIs retained.
- Added alignment script scaffold:
  - `scripts/eval/test_alignment.py` (prototype compare path for dmt_cpp vs GUDHI).
- Implemented bit-array pairing state scaffold in `cpp/dmt_implicit/src/dmt_implicit.cpp`:
  - introduced compact bit flags (`BitFlags`) for pair-consumed cells,
  - integrated into H0 proxy pairing pass.
- Added initial H1 proxy detection attempt (3x3 ring heuristic) in C++ backend (prototype-only).
- Re-validated backend comparison on demo NPZ:
  - `gudhi`: THE mean ~0.010916, pairs mean ~1236
  - `dmt_cpp`: THE mean ~0.001058, pairs mean ~853
  - gap remains expected because backend is still heuristic/prototype.
- Added prototype implicit DMT backend scaffold under `cpp/dmt_implicit/`:
  - `CMakeLists.txt`
  - `include/dmt_implicit.hpp`
  - `src/dmt_implicit.cpp`
  - `src/bindings.cpp`
- Added reproducible build helper:
  - `scripts/run/build_dmt_implicit.sh` (conda env + pybind11 CMake dir + system cmake path).
- Added pybind module interface `dmt_implicit_ext.extract_persistence_2d(...)` (prototype shape/API ready).
- Hooked runtime backend switch into Phase-1 persistence extraction:
  - `02_phase1_diagnostic/core/persistence_homology.py`
  - New `extract_persistence(..., backend="gudhi"|"dmt_cpp")`
- Extended THE bridge and CLI to pass backend selection:
  - `02_phase1_diagnostic/baselines/struct_uncertainty_bridge.py`
  - `scripts/eval/eval_the.py` (`--backend {gudhi,dmt_cpp}`)
- Added topology config backend field:
  - `configs/topology/the_metric.yaml` now includes `backend: gudhi`.
- Sanity checks passed:
  - Python syntax compile on updated files succeeded.
  - No linter errors found on updated Python files.

### Failed / blocked in this round
- During refactor to 3D scaffold + stricter 2D foundation, `extract_persistence_2d` temporary proxy currently returns empty pairs on random inputs; PH-equivalent persistence extraction is still pending.
- `dmt_cpp` backend is **not** full PH-equivalent Morse pairing yet.
- Initial apt-based toolchain install path was unstable due to archive rename/cache errors.
- Build was unblocked by using conda env + system cmake (`/usr/bin/cmake`) and pybind11 from `py312` environment.
- End-to-end smoke test succeeded on synthetic NPZ (`outputs/uq_maps/demo_uq_for_the.npz`).
- Backend comparison executed on synthetic NPZ:
  - GUDHI CSV: `outputs/reports/the_demo_gudhi.csv` (mean pairs ~1236, includes H0/H1).
  - DMT prototype CSV: `outputs/reports/the_demo_dmt_cpp.csv` (mean pairs ~853, H0-only proxy).
- Real-dataset parity vs GUDHI is not validated yet.
- Bit-array Morse pairing and critical-cell extraction are not implemented yet.

## 2) Immediate next objectives (priority ordered)

1. **P0: Replace heuristic pairing with PH-consistent Morse pairing**
   - Current bit-array state exists but pairing logic is still heuristic.
   - Implement true discrete Morse pairing transitions and critical-cell extraction.

2. **P0: Improve H1 extraction quality**
   - Current ring heuristic often yields zero H1 in realistic maps.
   - Introduce robust cycle candidate tracking (edge/face-level, not pixel-ring shortcut).

3. **P0: Validate and harden C++ build path**
   - Add local build instructions + optional editable install notes.
   - Verify pybind module import path (`dmt_implicit_ext`) under target Python env.

3. **P0: Wire config-driven backend selection end-to-end**
   - Ensure THE runner reads `configs/topology/the_metric.yaml` backend field directly.
   - Keep fallback to `gudhi` when extension is unavailable.

4. **P1: Hook backend into THE/TTTGF mainline behavior checks**
   - Compare output PD counts/statistics between `gudhi` and `dmt_cpp`.
   - Preserve existing THE API and sparse-routing compatibility.

5. **P1: Benchmarks and acceptance gates**
   - Add memory/runtime benchmark script for 2D/3D tensors.
   - Define threshold for PD consistency + no TTTGF convergence regression.

## 3) Suggested execution commands (server)

```bash
# 0) dependency sync
pip install -r requirements.txt

# 1) build extension with reproducible helper (recommended)
bash scripts/run/build_dmt_implicit.sh

# 2) smoke test using synthetic npz (already validated once)
python - <<'PY'
import numpy as np
from pathlib import Path
p = Path('outputs/uq_maps'); p.mkdir(parents=True, exist_ok=True)
np.savez(p/'demo_uq_for_the.npz', pred_prob=np.random.rand(2,1,64,64).astype('float32'), names=np.array(['a.png','b.png']))
print(p/'demo_uq_for_the.npz')
PY

PYTHONPATH="cpp/dmt_implicit/build:${PYTHONPATH}" \
python scripts/eval/eval_the.py \
  --uq_npz outputs/uq_maps/demo_uq_for_the.npz \
  --backend dmt_cpp \
  --max_samples 2 --max_hw 64

# 3) real-data compare (when real uq npz exists)
PYTHONPATH="cpp/dmt_implicit/build:${PYTHONPATH}" \
python scripts/eval/eval_the.py --uq_npz outputs/uq_maps/mc_dropout_stats.npz --backend gudhi

PYTHONPATH="cpp/dmt_implicit/build:${PYTHONPATH}" \
python scripts/eval/eval_the.py --uq_npz outputs/uq_maps/mc_dropout_stats.npz --backend dmt_cpp
```

## 4) Handoff checklist for next agent

- [x] Create `cpp/dmt_implicit/` prototype with on-the-fly neighborhood evaluation scaffold.
- [x] Expose C++ API through pybind prototype (`dmt_implicit_ext`).
- [x] Add runtime backend switch (`gudhi` / `dmt_cpp`) in Phase-1 persistence extraction path.
- [x] Add THE CLI backend argument (`scripts/eval/eval_the.py --backend ...`).
- [ ] Implement bit-array Morse pairing state and critical-cell extraction (real outputs).
- [ ] Validate C++ extension build/import in target runtime and add troubleshooting notes.
- [ ] Add memory/runtime benchmark script and write comparison report.
- [ ] Define acceptance threshold for PD consistency and TTTGF behavior.

## 5) Process rule (important)

For every completed execution round, update this file before ending work:
1. What was completed.
2. What failed/blocked and why.
3. Concrete next-step tasks with priority.
4. Exact runnable commands for the next agent.
