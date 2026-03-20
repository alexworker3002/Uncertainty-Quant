from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch
import torch.nn.functional as F

from uce.data.dataset import DriveDataset, build_dataloader
from uce.models.unet2d import UNet2D
from uce.utils.config import load_yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fit scalar temperature on validation set by minimizing NLL.")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--ckpt", type=str, default="outputs/checkpoints/best.pt")
    p.add_argument("--output", type=str, default="outputs/uq_maps/temperature.json")
    p.add_argument("--max_iter", type=int, default=100)
    p.add_argument("--lr", type=float, default=0.05)
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
    m.eval()
    return m


def main() -> None:
    args = parse_args()
    exp_cfg = load_yaml(args.config)
    dataset_cfg = load_yaml(exp_cfg["dataset_config"])
    training_cfg = load_yaml(exp_cfg["training_config"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(exp_cfg, Path(args.ckpt), device)

    input_h, input_w = dataset_cfg.get("input_size", [512, 512])
    val_split_file = Path("data/splits/drive_val.txt")

    val_ds = DriveDataset(
        root=dataset_cfg["root"],
        split="train",
        image_subdir=dataset_cfg.get("image_subdir", "images"),
        mask_subdir=dataset_cfg.get("mask_subdir", "mask"),
        input_size=(input_h, input_w),
        split_file=val_split_file if val_split_file.exists() else None,
    )

    val_loader = build_dataloader(
        val_ds,
        batch_size=training_cfg.get("batch_size", 1),
        num_workers=training_cfg.get("num_workers", 0),
        shuffle=False,
        drop_last=False,
    )

    logits_list = []
    targets_list = []
    with torch.no_grad():
        for batch in val_loader:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["mask"].to(device, non_blocking=True)
            logits = model(x)
            logits_list.append(logits)
            targets_list.append(y)

    logits_all = torch.cat(logits_list, dim=0)
    targets_all = torch.cat(targets_list, dim=0)

    log_t = torch.nn.Parameter(torch.zeros(1, device=device))
    optimizer = torch.optim.Adam([log_t], lr=args.lr)

    best_loss = float("inf")
    best_t = 1.0

    for _ in range(args.max_iter):
        optimizer.zero_grad(set_to_none=True)
        t = torch.exp(log_t).clamp_min(1e-6)
        scaled_logits = logits_all / t
        loss = F.binary_cross_entropy_with_logits(scaled_logits, targets_all)
        loss.backward()
        optimizer.step()

        l = float(loss.item())
        if l < best_loss:
            best_loss = l
            best_t = float(t.detach().item())

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"temperature": best_t, "val_bce": best_loss, "max_iter": args.max_iter}, indent=2),
        encoding="utf-8",
    )

    print(f"Saved temperature to {out_path}")
    print(f"temperature={best_t:.6f}, val_bce={best_loss:.6f}")


if __name__ == "__main__":
    main()
