from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.uce.models.unet2d import UNet2D
from src.uce.utils.config import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exp_cfg = load_yaml(args.config)
    model_cfg = load_yaml(exp_cfg["model_config"])

    model = UNet2D(
        in_channels=model_cfg["in_channels"],
        out_channels=model_cfg["out_channels"],
        init_features=model_cfg["init_features"],
        dropout=model_cfg.get("dropout", 0.1),
    )

    output_root = Path(exp_cfg.get("output_root", "outputs"))
    ckpt_dir = output_root / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Placeholder training loop for scaffold stage.
    # Replace with real dataloader/optimizer/loss in next step.
    dummy_x = torch.randn(2, model_cfg["in_channels"], 512, 512)
    dummy_logits = model(dummy_x)
    print(f"[Scaffold] Forward OK. logits shape={tuple(dummy_logits.shape)}")

    torch.save(model.state_dict(), ckpt_dir / "baseline_scaffold.pt")
    print(f"[Scaffold] Saved checkpoint to {ckpt_dir / 'baseline_scaffold.pt'}")


if __name__ == "__main__":
    main()
