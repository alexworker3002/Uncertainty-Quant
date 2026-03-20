from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlretrieve

import kagglehub

import numpy as np
from PIL import Image


CANDIDATE_URLS = {
    "training": [
        "https://drive.grand-challenge.org/DRIVE/training.zip",
        "https://drive.grand-challenge.org/api/v1/cases/images/?archive=training",
    ],
    "test": [
        "https://drive.grand-challenge.org/DRIVE/test.zip",
        "https://drive.grand-challenge.org/api/v1/cases/images/?archive=test",
    ],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download and prepare DRIVE dataset into project layout.")
    p.add_argument("--raw_dir", type=str, default="data/raw/drive", help="Where zip/extracted raw files are stored")
    p.add_argument(
        "--processed_root",
        type=str,
        default="data/processed/drive",
        help="Output folder in DRIVE-style train/test + images/mask structure",
    )
    p.add_argument("--training_zip", type=str, default="", help="Optional local path to DRIVE training.zip")
    p.add_argument("--test_zip", type=str, default="", help="Optional local path to DRIVE test.zip")
    p.add_argument("--training_url", type=str, default="", help="Optional direct URL for training.zip")
    p.add_argument("--test_url", type=str, default="", help="Optional direct URL for test.zip")
    p.add_argument(
        "--kaggle_dataset",
        type=str,
        default="",
        help="Optional Kaggle dataset slug (e.g. user/dataset). If set, tries kaggle CLI fallback.",
    )
    p.add_argument(
        "--kagglehub_dataset",
        type=str,
        default="andrewmvd/drive-digital-retinal-images-for-vessel-extraction",
        help="KaggleHub dataset slug for python-based download fallback.",
    )
    p.add_argument("--force", action="store_true", help="Re-download and re-prepare even if files exist")
    return p.parse_args()


def _download_with_fallback(target_name: str, out_path: Path, override_url: str = "") -> None:
    url_candidates = [override_url] if override_url else CANDIDATE_URLS[target_name]
    errors: list[str] = []
    for url in url_candidates:
        if not url:
            continue
        try:
            print(f"[Download] {target_name}: trying {url}")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            urlretrieve(url, out_path)
            if out_path.exists() and out_path.stat().st_size > 0:
                print(f"[Download] {target_name}: saved -> {out_path}")
                return
        except (URLError, HTTPError, TimeoutError) as e:
            errors.append(f"{url} -> {e}")
        except Exception as e:
            errors.append(f"{url} -> {type(e).__name__}: {e}")

    msg = "\n".join(errors) if errors else "no candidate URLs"
    raise RuntimeError(
        f"Failed to download DRIVE {target_name}.\n"
        f"Tried:\n{msg}\n"
        "If direct download is blocked, provide local zip paths or enable Kaggle fallback."
    )


def _try_kaggle_download(raw_dir: Path, dataset_slug: str) -> None:
    if not dataset_slug:
        return

    kaggle_bin = shutil.which("kaggle")
    if kaggle_bin is None:
        raise RuntimeError("kaggle CLI not found in PATH. Install with `pip install kaggle`.")

    print(f"[Kaggle] trying dataset: {dataset_slug}")
    cmd = [kaggle_bin, "datasets", "download", "-d", dataset_slug, "-p", str(raw_dir), "--force"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "Kaggle download failed. Ensure kaggle is authenticated (kaggle.json).\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    print("[Kaggle] download command finished.")


def _try_kagglehub_download(raw_dir: Path, dataset_slug: str) -> Path:
    if not dataset_slug:
        raise RuntimeError("kagglehub dataset slug is empty")

    print(f"[KaggleHub] trying dataset: {dataset_slug}")
    downloaded_path = Path(kagglehub.dataset_download(dataset_slug))
    if not downloaded_path.exists():
        raise FileNotFoundError(f"KaggleHub returned non-existing path: {downloaded_path}")

    # Sync downloaded tree into raw_dir for consistent downstream handling
    raw_dir.mkdir(parents=True, exist_ok=True)
    if downloaded_path.is_dir():
        for item in downloaded_path.iterdir():
            target = raw_dir / item.name
            if target.exists():
                continue
            if item.is_dir():
                shutil.copytree(item, target)
            else:
                shutil.copy2(item, target)
    else:
        target = raw_dir / downloaded_path.name
        if not target.exists():
            shutil.copy2(downloaded_path, target)

    print(f"[KaggleHub] synced files into: {raw_dir}")
    return downloaded_path


def _extract(zip_path: Path, out_dir: Path) -> Path:
    target = out_dir / zip_path.stem
    if target.exists():
        return target
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(target)
    return target


def _normalize_mask(mask_arr: np.ndarray) -> np.ndarray:
    if mask_arr.ndim == 3:
        mask_arr = mask_arr[..., 0]
    return ((mask_arr > 0).astype(np.uint8) * 255)


def _copy_image_as_png(src: Path, dst: Path) -> None:
    img = Image.open(src).convert("RGB")
    dst.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst)


def _copy_mask_as_png(src: Path, dst: Path) -> None:
    arr = np.array(Image.open(src), dtype=np.uint8)
    arr = _normalize_mask(arr)
    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(dst)


def _resolve_split_root(extracted_root: Path, kind: str) -> Path:
    direct = extracted_root / kind
    if direct.exists():
        return direct

    nested = list(extracted_root.glob(f"**/{kind}"))
    for c in nested:
        if (c / "images").exists() and (c / "1st_manual").exists():
            return c

    # fallback: extracted_root itself may already be split root
    if (extracted_root / "images").exists() and (extracted_root / "1st_manual").exists():
        return extracted_root

    raise FileNotFoundError(f"Could not find split root for '{kind}' under {extracted_root}")


def _collect_paths(split_root: Path) -> tuple[Path, Path]:
    image_dir = split_root / "images"
    manual_dir = split_root / "1st_manual"
    if not image_dir.exists():
        raise FileNotFoundError(f"Missing image dir: {image_dir}")
    if not manual_dir.exists():
        # Some mirrors expose only FOV masks for test split.
        # Fall back to mask directory to keep pipeline runnable.
        fallback_mask_dir = split_root / "mask"
        if fallback_mask_dir.exists():
            print(f"[Warn] 1st_manual not found at {split_root}, falling back to mask/")
            return image_dir, fallback_mask_dir
        raise FileNotFoundError(f"Missing manual mask dir: {manual_dir}")
    return image_dir, manual_dir


def _prepare_split(extracted_root: Path, kind: str, processed_root: Path) -> None:
    split_root = _resolve_split_root(extracted_root, kind)
    src_images, src_masks = _collect_paths(split_root)

    out_img = processed_root / kind / "images"
    out_mask = processed_root / kind / "mask"
    out_img.mkdir(parents=True, exist_ok=True)
    out_mask.mkdir(parents=True, exist_ok=True)

    image_paths = sorted([p for p in src_images.iterdir() if p.is_file()])
    if not image_paths:
        raise RuntimeError(f"No source images found in {src_images}")

    for img_path in image_paths:
        stem_prefix = img_path.stem.split("_")[0]
        mask_candidates = sorted(src_masks.glob(f"{stem_prefix}_*"))
        if not mask_candidates:
            print(f"[Warn] No mask found for {img_path.name}, skipped")
            continue

        out_name = f"{img_path.stem}.png"
        _copy_image_as_png(img_path, out_img / out_name)
        _copy_mask_as_png(mask_candidates[0], out_mask / out_name)


def _count_pairs(processed_root: Path, split: str) -> int:
    image_dir = processed_root / split / "images"
    mask_dir = processed_root / split / "mask"
    if not image_dir.exists() or not mask_dir.exists():
        return 0
    n = 0
    for p in image_dir.glob("*.png"):
        if (mask_dir / p.name).exists():
            n += 1
    return n


def _find_local_zip(raw_dir: Path, name: str) -> Path | None:
    candidates = [raw_dir / f"{name}.zip"] + sorted(raw_dir.glob(f"*{name}*.zip"))
    for p in candidates:
        if p.exists() and p.is_file():
            return p
    return None


def _find_extracted_drive_root(raw_dir: Path) -> Path | None:
    candidates = [raw_dir / "DRIVE", raw_dir]
    for c in candidates:
        if (c / "training" / "images").exists() and (c / "test" / "images").exists():
            return c
    nested = [p for p in raw_dir.glob("**/DRIVE") if p.is_dir()]
    for c in nested:
        if (c / "training" / "images").exists() and (c / "test" / "images").exists():
            return c
    return None


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    processed_root = Path(args.processed_root)

    raw_dir.mkdir(parents=True, exist_ok=True)

    user_train_zip = bool(args.training_zip)
    user_test_zip = bool(args.test_zip)
    train_zip = Path(args.training_zip) if user_train_zip else (raw_dir / "training.zip")
    test_zip = Path(args.test_zip) if user_test_zip else (raw_dir / "test.zip")

    if args.force:
        for p in [train_zip, test_zip]:
            if p.exists():
                p.unlink()
        if processed_root.exists():
            shutil.rmtree(processed_root)

    # 1) explicit local paths
    if user_train_zip and not train_zip.exists():
        raise FileNotFoundError(f"Provided --training_zip not found: {train_zip}")
    if user_test_zip and not test_zip.exists():
        raise FileNotFoundError(f"Provided --test_zip not found: {test_zip}")

    # 2) local auto-discovery
    if not train_zip.exists():
        discovered = _find_local_zip(raw_dir, "training")
        if discovered:
            print(f"[Local] found training zip: {discovered}")
            train_zip = discovered
    if not test_zip.exists():
        discovered = _find_local_zip(raw_dir, "test")
        if discovered:
            print(f"[Local] found test zip: {discovered}")
            test_zip = discovered

    # 3) URL download fallback (non-fatal, so Kaggle can still be attempted)
    download_errors: list[str] = []
    if not train_zip.exists():
        try:
            _download_with_fallback("training", train_zip, override_url=args.training_url)
        except Exception as e:
            download_errors.append(f"training: {e}")
    if not test_zip.exists():
        try:
            _download_with_fallback("test", test_zip, override_url=args.test_url)
        except Exception as e:
            download_errors.append(f"test: {e}")

    # 4) Kaggle CLI fallback (if requested)
    if (not train_zip.exists() or not test_zip.exists()) and args.kaggle_dataset:
        try:
            _try_kaggle_download(raw_dir, args.kaggle_dataset)
        except Exception as e:
            download_errors.append(f"kaggle_cli: {e}")

        if not train_zip.exists():
            discovered = _find_local_zip(raw_dir, "training")
            if discovered:
                train_zip = discovered
        if not test_zip.exists():
            discovered = _find_local_zip(raw_dir, "test")
            if discovered:
                test_zip = discovered

    # 5) KaggleHub python fallback
    if not train_zip.exists() or not test_zip.exists():
        try:
            _try_kagglehub_download(raw_dir, args.kagglehub_dataset)
        except Exception as e:
            download_errors.append(f"kagglehub: {e}")

        if not train_zip.exists():
            discovered = _find_local_zip(raw_dir, "training")
            if discovered:
                train_zip = discovered
        if not test_zip.exists():
            discovered = _find_local_zip(raw_dir, "test")
            if discovered:
                test_zip = discovered

    extracted_drive_root = _find_extracted_drive_root(raw_dir)

    if not train_zip.exists() or not test_zip.exists():
        if extracted_drive_root is None:
            details = "\n".join(download_errors) if download_errors else ""
            raise FileNotFoundError(
                "Both training.zip and test.zip must be available, or an extracted DRIVE tree must exist under raw_dir. "
                "Provide --training_zip/--test_zip, valid --training_url/--test_url, --kaggle_dataset, or --kagglehub_dataset."
                + (f"\nDownload attempts:\n{details}" if details else "")
            )
        train_extracted = extracted_drive_root
        test_extracted = extracted_drive_root
    else:
        train_extracted = _extract(train_zip, raw_dir)
        test_extracted = _extract(test_zip, raw_dir)

    _prepare_split(train_extracted, "training", processed_root.parent / "_tmp_training")
    _prepare_split(test_extracted, "test", processed_root.parent / "_tmp_test")

    # Normalize final split names to project conventions: train/test
    final_train = processed_root / "train"
    final_test = processed_root / "test"
    final_train.mkdir(parents=True, exist_ok=True)
    final_test.mkdir(parents=True, exist_ok=True)

    tmp_train = processed_root.parent / "_tmp_training" / "training"
    tmp_test = processed_root.parent / "_tmp_test" / "test"

    if final_train.exists():
        shutil.rmtree(final_train)
    if final_test.exists():
        shutil.rmtree(final_test)

    shutil.move(str(tmp_train), str(final_train))
    shutil.move(str(tmp_test), str(final_test))

    # cleanup temp staging dirs
    for p in [processed_root.parent / "_tmp_training", processed_root.parent / "_tmp_test"]:
        if p.exists():
            shutil.rmtree(p)

    train_n = _count_pairs(processed_root, "train")
    test_n = _count_pairs(processed_root, "test")

    print("[Done] DRIVE prepared.")
    print(f"train pairs: {train_n}")
    print(f"test pairs : {test_n}")
    print(f"output root : {processed_root}")

    if train_n == 0 or test_n == 0:
        print("[Warn] Zero pairs detected. Please inspect raw extraction/layout.")
        sys.exit(2)


if __name__ == "__main__":
    main()
