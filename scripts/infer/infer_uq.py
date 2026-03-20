from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np
import torch

from uce.data.dataset import DriveDataset, build_dataloader
from uce.models.unet2d import UNet2D
from uce.uq_baselines import (
    deep_ensemble_predict,
    deterministic_predict,
    mc_dropout_predict,
    temperature_scaled_predict,
    tta_predict,
)
from uce.utils.config import load_yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument(
        "--method",
        type=str,
        default="mc_dropout",
        choices=["deterministic", "mc_dropout", "deep_ensemble", "tta", "temp_scaling"],
    )
    p.add_argument("--ckpt", type=str, default="outputs/checkpoints/best.pt")
    p.add_argument("--output", type=str, default="outputs/uq_maps/mc_dropout_stats.npz")
    p.add_argument("--num_samples", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--temperature_file", type=str, default="")
    return p.parse_args()


def build_model(exp_cfg: dict, ckpt_path: Path, device: torch.device) -> UNet2D:
    model_cfg = load_yaml(exp_cfg["model_config"])
    m = UNet2D(
        in_channels=model_cfg["in_channels"],
        out_channels=model_cfg["out_channels"],
        init_features=model_cfg["init_features"],
        dropout=model_cfg.get("dropout", 0.1),
    ).to(device)
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device)
        m.load_state_dict(state, strict=False)
    return m


def resolve_temperature(args: argparse.Namespace) -> float:
    if args.temperature_file:
        t_path = Path(args.temperature_file)
        if not t_path.exists():
            raise FileNotFoundError(f"temperature_file not found: {t_path}")
        data = json.loads(t_path.read_text(encoding="utf-8"))
        return float(data.get("temperature", 1.0))
    return float(args.temperature)


def main() -> None:
    args = parse_args()
    exp_cfg = load_yaml(args.config)
    dataset_cfg = load_yaml(exp_cfg["dataset_config"])
    training_cfg = load_yaml(exp_cfg["training_config"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = Path(args.ckpt)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    input_h, input_w = dataset_cfg.get("input_size", [512, 512])
    test_ds = DriveDataset(
        root=dataset_cfg["root"],
        split="test",
        image_subdir=dataset_cfg.get("image_subdir", "images"),
        mask_subdir=dataset_cfg.get("mask_subdir", "mask"),
        input_size=(input_h, input_w),
    )
    test_loader = build_dataloader(
        test_ds,
        batch_size=args.batch_size,
        num_workers=training_cfg.get("num_workers", 0),
        shuffle=False,
        drop_last=False,
    )

    all_mean_prob: list[np.ndarray] = []
    all_variance: list[np.ndarray] = []
    all_entropy: list[np.ndarray] = []
    all_names: list[str] = []

    if args.method == "deep_ensemble":
        models = [build_model(exp_cfg, ckpt, device) for _ in range(3)]
        for batch in test_loader:
            x = batch["image"].to(device, non_blocking=True)
            uq = deep_ensemble_predict(models, x)
            all_mean_prob.append(uq.mean_prob.detach().cpu().numpy())
            all_variance.append(uq.variance.detach().cpu().numpy())
            all_entropy.append(uq.entropy.detach().cpu().numpy())
            all_names.extend([str(n) for n in batch["name"]])
    else:
        model = build_model(exp_cfg, ckpt, device)
        temperature = resolve_temperature(args) if args.method == "temp_scaling" else 1.0

        for batch in test_loader:
            x = batch["image"].to(device, non_blocking=True)
            if args.method == "deterministic":
                uq = deterministic_predict(model, x)
            elif args.method == "mc_dropout":
                uq = mc_dropout_predict(model, x, num_samples=args.num_samples)
            elif args.method == "temp_scaling":
                uq = temperature_scaled_predict(model, x, temperature=temperature)
            else:
                uq = tta_predict(model, x)

            all_mean_prob.append(uq.mean_prob.detach().cpu().numpy())
            all_variance.append(uq.variance.detach().cpu().numpy())
            all_entropy.append(uq.entropy.detach().cpu().numpy())
            all_names.extend([str(n) for n in batch["name"]])

    mean_prob = np.concatenate(all_mean_prob, axis=0)
    variance = np.concatenate(all_variance, axis=0)
    entropy = np.concatenate(all_entropy, axis=0)

    np.savez_compressed(
        out,
        mean_prob=mean_prob,
        variance=variance,
        entropy=entropy,
        names=np.array(all_names, dtype=object),
    )
    print(f"Saved UQ stats to {out}")
    print(f"num_samples={len(all_names)} shape={mean_prob.shape}")


if __name__ == "__main__":
    main()
