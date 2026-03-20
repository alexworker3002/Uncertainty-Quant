from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from uce.data.dataset import DriveDataset, build_dataloader
from uce.models.unet2d import UNet2D
from uce.utils.config import load_yaml


class SyntheticSegDataset(Dataset):
    def __init__(self, n: int, in_channels: int, h: int, w: int) -> None:
        self.n = n
        self.in_channels = in_channels
        self.h = h
        self.w = w

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        x = torch.rand(self.in_channels, self.h, self.w)
        y = (torch.rand(1, self.h, self.w) > 0.5).float()
        return {"image": x, "mask": y, "name": f"synthetic_{idx:04d}.png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--smoke_steps", type=int, default=2, help="Limit train/val iterations for quick local checks.")
    return parser.parse_args()


def dice_loss_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    inter = (probs * targets).sum(dim=(1, 2, 3))
    union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2 * inter + eps) / (union + eps)
    return 1.0 - dice.mean()


def bce_dice_loss(logits: torch.Tensor, targets: torch.Tensor, bce_w: float, dice_w: float) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    dloss = dice_loss_from_logits(logits, targets)
    return bce_w * bce + dice_w * dloss


def build_dataloaders(exp_cfg: dict, training_cfg: dict, model_cfg: dict) -> tuple[DataLoader, DataLoader, DataLoader]:
    dataset_cfg = load_yaml(exp_cfg["dataset_config"])
    input_h, input_w = dataset_cfg.get("input_size", [512, 512])

    try:
        split_dir = Path("data/splits")
        train_split_file = split_dir / "drive_train.txt"
        val_split_file = split_dir / "drive_val.txt"

        train_ds = DriveDataset(
            root=dataset_cfg["root"],
            split="train",
            image_subdir=dataset_cfg.get("image_subdir", "images"),
            mask_subdir=dataset_cfg.get("mask_subdir", "mask"),
            input_size=(input_h, input_w),
            split_file=train_split_file if train_split_file.exists() else None,
        )
        val_ds = DriveDataset(
            root=dataset_cfg["root"],
            split="train",
            image_subdir=dataset_cfg.get("image_subdir", "images"),
            mask_subdir=dataset_cfg.get("mask_subdir", "mask"),
            input_size=(input_h, input_w),
            split_file=val_split_file if val_split_file.exists() else None,
        )
        test_ds = DriveDataset(
            root=dataset_cfg["root"],
            split="test",
            image_subdir=dataset_cfg.get("image_subdir", "images"),
            mask_subdir=dataset_cfg.get("mask_subdir", "mask"),
            input_size=(input_h, input_w),
            split_file=None,
        )
        print(f"[Data] Using DRIVE dataset. train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    except Exception as e:
        print(f"[Data] DRIVE data unavailable, falling back to synthetic smoke dataset. reason={e}")
        train_ds = SyntheticSegDataset(n=8, in_channels=model_cfg["in_channels"], h=input_h, w=input_w)
        val_ds = SyntheticSegDataset(n=4, in_channels=model_cfg["in_channels"], h=input_h, w=input_w)
        test_ds = SyntheticSegDataset(n=4, in_channels=model_cfg["in_channels"], h=input_h, w=input_w)

    train_loader = build_dataloader(
        train_ds,
        batch_size=training_cfg["batch_size"],
        num_workers=training_cfg["num_workers"],
        shuffle=True,
        drop_last=False,
    )
    val_loader = build_dataloader(
        val_ds,
        batch_size=training_cfg["batch_size"],
        num_workers=training_cfg["num_workers"],
        shuffle=False,
        drop_last=False,
    )
    test_loader = build_dataloader(
        test_ds,
        batch_size=training_cfg["batch_size"],
        num_workers=training_cfg["num_workers"],
        shuffle=False,
        drop_last=False,
    )
    return train_loader, val_loader, test_loader


def main() -> None:
    args = parse_args()
    exp_cfg = load_yaml(args.config)
    model_cfg = load_yaml(exp_cfg["model_config"])
    training_cfg = load_yaml(exp_cfg["training_config"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet2D(
        in_channels=model_cfg["in_channels"],
        out_channels=model_cfg["out_channels"],
        init_features=model_cfg["init_features"],
        dropout=model_cfg.get("dropout", 0.1),
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=training_cfg["optimizer"]["lr"],
        weight_decay=training_cfg["optimizer"].get("weight_decay", 1e-5),
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, training_cfg["epochs"]),
        eta_min=training_cfg.get("scheduler", {}).get("min_lr", 1e-6),
    )

    amp_enabled = bool(training_cfg.get("amp", True) and torch.cuda.is_available())
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    train_loader, val_loader, test_loader = build_dataloaders(exp_cfg, training_cfg, model_cfg)

    bce_w = training_cfg["loss"].get("bce_weight", 0.5)
    dice_w = training_cfg["loss"].get("dice_weight", 0.5)

    output_root = Path(exp_cfg.get("output_root", "outputs"))
    ckpt_dir = output_root / "checkpoints"
    pred_dir = output_root / "predictions"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")

    epochs = 1 if args.smoke_steps > 0 else int(training_cfg["epochs"])
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for it, batch in enumerate(train_loader):
            if args.smoke_steps > 0 and it >= args.smoke_steps:
                break
            x = batch["image"].to(device, non_blocking=True)
            y = batch["mask"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                logits = model(x)
                loss = bce_dice_loss(logits, y, bce_w=bce_w, dice_w=dice_w)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += float(loss.item())

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for it, batch in enumerate(val_loader):
                if args.smoke_steps > 0 and it >= args.smoke_steps:
                    break
                x = batch["image"].to(device, non_blocking=True)
                y = batch["mask"].to(device, non_blocking=True)
                logits = model(x)
                loss = bce_dice_loss(logits, y, bce_w=bce_w, dice_w=dice_w)
                val_loss += float(loss.item())

                # Save one prediction sample for pipeline validation
                probs = torch.sigmoid(logits)
                pred = (probs[0, 0] > 0.5).to(torch.uint8).cpu().numpy() * 255
                from PIL import Image

                Image.fromarray(pred).save(pred_dir / str(batch["name"][0]))
                break

        scheduler.step()

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), ckpt_dir / "best.pt")

        print(f"epoch={epoch+1} train_loss={train_loss:.4f} val_loss={val_loss:.4f} best_val={best_val:.4f}")

    torch.save(model.state_dict(), ckpt_dir / "last.pt")

    best_ckpt = ckpt_dir / "best.pt"
    if best_ckpt.exists():
        model.load_state_dict(torch.load(best_ckpt, map_location=device), strict=False)

    # full test-set prediction export for eval_seg.py (from best checkpoint)
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            x = batch["image"].to(device, non_blocking=True)
            logits = model(x)
            probs = torch.sigmoid(logits)
            preds = (probs[:, 0] > 0.5).to(torch.uint8).cpu().numpy() * 255

            from PIL import Image

            names = batch["name"]
            for i in range(len(names)):
                Image.fromarray(preds[i]).save(pred_dir / str(names[i]))

    print(f"Saved checkpoints to {ckpt_dir} and test predictions to {pred_dir}")


if __name__ == "__main__":
    main()
