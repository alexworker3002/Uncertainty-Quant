# Next-Step Execution Plan (Agent Handoff)

Last updated: 2026-03-20
Owner: Ice + Cursor agents

## 1) Current project status

### Completed in this round
- Upgraded and stabilized data acquisition/preparation pipeline:
  - `scripts/data/download_drive.py` now supports:
    - online URL attempts,
    - local zip inputs,
    - Kaggle CLI fallback,
    - KaggleHub fallback (python API).
  - Added support for extracted DRIVE tree fallback (works even when zips are absent but `data/raw/drive/DRIVE/...` exists).
  - Added compatibility fallback for test split when `1st_manual` is unavailable (uses `mask/`).
- Real DRIVE data is now prepared in project layout:
  - `data/processed/drive/train/images`, `data/processed/drive/train/mask`
  - `data/processed/drive/test/images`, `data/processed/drive/test/mask`
  - Pair counts: train=20, test=20.
- Deterministic split generation completed:
  - `data/splits/drive_train.txt` (16)
  - `data/splits/drive_val.txt` (4)
  - `data/splits/drive_test.txt` (20)
- Training pipeline improvements:
  - `scripts/train/train_baseline.py`
    - full-epoch logic fixed (`--smoke_steps 0` now runs configured full epochs),
    - test predictions exported for full test set,
    - test inference uses `best.pt` checkpoint.
- Import robustness fixed in scripts:
  - `scripts/train/train_baseline.py`
  - `scripts/eval/eval_seg.py`
  - `scripts/infer/infer_uq.py`
  - All support local execution via `src` bootstrap + `uce.*` imports.
- Full baseline training completed on real DRIVE data (50 epochs).
- Segmentation evaluation completed:
  - Dice: **0.2410**
  - IoU: **0.1374**
- UQ path executed end-to-end (scaffold level):
  - `scripts/infer/infer_uq.py` generated `outputs/uq_maps/mc_dropout_stats.npz`.
  - `scripts/eval/eval_uq.py` executed successfully and printed summary:
    - `mean_prob shape: (2, 1, 512, 512)`
    - `variance mean: 0.003914`
    - `entropy mean: 0.581950`

### Failed / blocked in this round
- Official direct DRIVE URLs returned 404 in this environment; resolved by using KaggleHub/extracted-tree fallback.
- UQ evaluation remains scaffold-level only (ECE/NLL/Brier/Risk-Coverage/AURC not implemented yet).

## 2) Immediate next objectives (priority ordered)

1. **P0: DMT implementation planning and boundary alignment**
   - Keep paper/docs explicit: current code uses GUDHI path, implicit DMT backend not integrated yet.
   - Ensure all THE/TTTGF docs avoid claiming completed C++/CUDA DMT acceleration.

2. **P0: Build implicit DMT backend prototype (C++ first, CUDA optional)**
   - New module target: `cpp/dmt_implicit/`.
   - Implement on-the-fly local boundary evaluation (no explicit 8N simplex objects).
   - Implement bit-array pairing state for discrete Morse matching.
   - Output only compact critical-cell representation.

3. **P0: Python binding integration**
   - Add pybind bridge to expose:
     - `critical_cells`
     - `pair_indices`
     - `filtration_values`
   - Add fallback switch: `backend = {"gudhi", "dmt_cpp"}` in Phase-1 config.

4. **P1: Hook backend into THE/TTTGF mainline**
   - Replace direct cubical-complex build path when `backend=dmt_cpp`.
   - Preserve existing THE API and sparse-routing API compatibility.

5. **P1: Benchmarks and acceptance gates**
   - Add benchmark script for memory/time:
     - compare GUDHI vs DMT backend on 2D and 3D tensors.
   - Acceptance criteria:
     - same/similar PD statistics,
     - peak memory reduction,
     - no regression in TTTGF convergence behavior.

## 3) Suggested execution commands (server)

```bash
# 0) dependency sync (if needed)
pip install -r requirements.txt

# 1) prepare DRIVE (auto-fallback enabled)
python scripts/data/download_drive.py --kagglehub_dataset andrewmvd/drive-digital-retinal-images-for-vessel-extraction

# 2) generate splits
python scripts/data/preprocess.py --root data/processed/drive --seed 42 --val_ratio 0.2

# 3) full baseline train (50 epochs from config)
python scripts/train/train_baseline.py --config configs/experiments/exp_01_baseline_uq.yaml --smoke_steps 0

# 4) segmentation evaluation
python scripts/eval/eval_seg.py --pred_dir outputs/predictions --gt_dir data/processed/drive/test/mask

# 5) current scaffold UQ run
python scripts/infer/infer_uq.py \
  --config configs/experiments/exp_01_baseline_uq.yaml \
  --method mc_dropout \
  --ckpt outputs/checkpoints/best.pt \
  --output outputs/uq_maps/mc_dropout_stats.npz \
  --num_samples 10

python scripts/eval/eval_uq.py --uq_npz outputs/uq_maps/mc_dropout_stats.npz --gt_dir data/processed/drive/test/mask
```

## 4) Handoff checklist for next agent

- [x] Keep docs aligned with current status: THE/TTTGF implemented, implicit DMT backend not yet integrated.
- [ ] Create `cpp/dmt_implicit/` prototype with on-the-fly neighborhood evaluation.
- [ ] Implement bit-array Morse pairing state and critical-cell extraction.
- [ ] Expose C++ APIs through pybind and add runtime backend switch.
- [ ] Integrate backend option into Phase-1 persistence extraction path.
- [ ] Add memory/runtime benchmark script and write comparison report.
- [ ] Define acceptance threshold for PD consistency and TTTGF behavior.

## 5) Process rule (important)

For every completed execution round, update this file before ending work:
1. What was completed.
2. What failed/blocked and why.
3. Concrete next-step tasks with priority.
4. Exact runnable commands for the next agent.
