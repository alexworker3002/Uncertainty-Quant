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

1. **P0: Make UQ inference use real test loader instead of random tensor**
   - Update `scripts/infer/infer_uq.py` to iterate over `DriveDataset(split="test")`.
   - Save per-image aligned outputs and names.

2. **P0: Implement formal UQ metrics**
   - Extend `scripts/eval/eval_uq.py` with:
     - ECE
     - NLL
     - Brier Score
     - Risk-Coverage / AURC

3. **P1: Export reproducible report tables**
   - Save segmentation + UQ summaries into CSV:
     - `reports/tables/exp01_metrics.csv`

4. **P1: Optional training quality uplift**
   - Add stronger augmentation and/or more stable validation protocol.
   - Track best epoch and metric curves to diagnose low Dice.

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

- [x] Acquire/prepare DRIVE data in project layout.
- [x] Generate deterministic splits.
- [x] Run full baseline training on real data.
- [x] Run segmentation evaluation and record metrics.
- [x] Run scaffold UQ inference/evaluation end-to-end.
- [ ] Replace random-tensor UQ inference with real test dataloader.
- [ ] Implement formal UQ metrics (ECE/NLL/Brier/Risk-Coverage/AURC).
- [ ] Export first reproducible metrics CSV table.

## 5) Process rule (important)

For every completed execution round, update this file before ending work:
1. What was completed.
2. What failed/blocked and why.
3. Concrete next-step tasks with priority.
4. Exact runnable commands for the next agent.
