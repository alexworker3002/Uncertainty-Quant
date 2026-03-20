# UCE: Uncertainty & Topology for Vessel Segmentation

This repository is organized to support a two-phase research workflow:

1. **Phase 1 (Diagnostic):** Frozen segmentation model + structural uncertainty diagnostics (THE).
2. **Phase 2 (Actionability):** Test-time topological gradient flow (TTTGF) for safe structural repair.

For now, we provide a minimal, practical baseline pipeline for vessel segmentation and uncertainty comparison.

## Quick start (baseline)

1. Prepare environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Configure experiment:

- `configs/experiments/exp_01_baseline_uq.yaml`

3. Train baseline:

```bash
python scripts/train/train_baseline.py --config configs/experiments/exp_01_baseline_uq.yaml
```

4. Evaluate segmentation:

```bash
python scripts/eval/eval_seg.py --pred_dir outputs/predictions --gt_dir data/processed/drive/test/mask
```

5. Evaluate uncertainty:

```bash
python scripts/eval/eval_uq.py --uq_npz outputs/uq_maps/mc_dropout_stats.npz --gt_dir data/processed/drive/test/mask
```

## Repository layout

- `configs/`: all experiment configs (dataset/model/training/uq/topology).
- `scripts/`: executable entrypoints for data, training, inference, and evaluation.
- `src/uce/`: reusable Python package code.
- `reports/`: generated tables/figures/logs for paper writing.
- `docs/`: experiment protocols and reviewer-defense notes.

## Notes

- Local machine can be used for scaffold/dry-runs.
- Full training/comparison should be run on your 40-series GPU server.
- **Agent handoff rule:** after each execution round, update `docs/NEXT_STEP_PLAN.md` with status, blockers, next tasks, and runnable commands.
