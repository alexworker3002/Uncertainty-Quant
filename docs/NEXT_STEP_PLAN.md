# Next-Step Execution Plan (Agent Handoff)

Last updated: 2026-03-20
Owner: Ice + Cursor agents

## 1) Current project status

### Completed in this round
- Repository scaffold aligned with `profile.md` two-phase narrative:
  - Phase 1 Diagnostic (THE)
  - Phase 2 Actionability (TTTGF)
- Baseline vessel segmentation + UQ comparison structure created:
  - Config system (`configs/*`)
  - Entry scripts (`scripts/train`, `scripts/eval`, `scripts/run`)
  - Core package skeleton (`src/uce/*`)
- Minimal U-Net 2D model scaffold implemented.
- Basic segmentation metrics scaffold implemented (Dice/IoU).
- Basic UQ output container + summary ops scaffold implemented.
- Baseline experiment config `exp_01_baseline_uq` prepared.

### Not yet completed
- Real DRIVE dataset loader + preprocessing pipeline.
- Real training loop (optimizer/loss/scheduler/amp/checkpoint-resume).
- Full UQ inference pipeline for deterministic / MC Dropout / ensemble / TTA.
- Formal UQ metrics (ECE/NLL/Brier/Risk-Coverage/AURC).
- Experiment report auto-export (CSV + plots).

## 2) Immediate next objectives (priority ordered)

1. **Data pipeline first (P0)**
   - Implement `src/uce/data/dataset.py` and `src/uce/data/transforms.py` for DRIVE.
   - Add `scripts/data/preprocess.py` to normalize and materialize train/val/test folders.
   - Ensure deterministic split file under `data/splits/`.

2. **Train baseline end-to-end (P0)**
   - Upgrade `scripts/train/train_baseline.py` to full trainer:
     - BCE+Dice loss
     - AdamW
     - mixed precision
     - model checkpointing and best-val selection
   - Save predictions for test set.

3. **UQ evaluation path (P1)**
   - Add `scripts/infer/infer_uq.py`.
   - Implement method adapters in `src/uce/uq_baselines/` for:
     - deterministic
     - mc_dropout
     - deep_ensemble
     - tta
   - Store `mean_prob/variance/entropy` as `npz`.

4. **Metrics & reporting (P1)**
   - Extend `scripts/eval/eval_uq.py` with ECE/NLL/Brier and risk-coverage metrics.
   - Save summary table to `reports/tables/exp01_metrics.csv`.

## 3) Suggested execution commands (server)

```bash
# 1) baseline train
python scripts/train/train_baseline.py --config configs/experiments/exp_01_baseline_uq.yaml

# 2) segmentation evaluation
python scripts/eval/eval_seg.py --pred_dir outputs/predictions --gt_dir data/processed/drive/test/mask

# 3) uq evaluation (after infer_uq is completed)
python scripts/eval/eval_uq.py --uq_npz outputs/uq_maps/mc_dropout_stats.npz --gt_dir data/processed/drive/test/mask
```

## 4) Handoff checklist for next agent

- [ ] Confirm dependencies and CUDA-specific torch build on target server.
- [ ] Confirm DRIVE license/access and data placement.
- [ ] Implement and test data loader with one mini-batch visual sanity check.
- [ ] Replace scaffold training placeholder with full training loop.
- [ ] Run a smoke test on 5-10 samples before full training.
- [ ] Export first reproducible metrics table.

## 5) Process rule (important)

For every completed execution round, update this file before ending work:
1. What was completed.
2. What failed/blocked and why.
3. Concrete next-step tasks with priority.
4. Exact runnable commands for the next agent.
