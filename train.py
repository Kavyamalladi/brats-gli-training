"""Training script for Swin UNETR on BraTS 2023 GLI dataset.

Uses MONAI's SwinUNETR model with the hyper‑parameters defined in
`configs/base_config.yaml`.  The script:

* Loads the dataset via :class:`datasets.brats_dataset.BraTSDataset`
  (passing ``dataset_version='2023'``).
* Splits the data into a training and validation set (80/20 split).
* Builds a SwinUNETR model with ``in_channels=4`` and ``out_channels=3``.
* Optimises with AdamW and a cosine‑annealing learning‑rate schedule.
* Computes a compound loss ``0.5 * DiceLoss + 0.5 * CrossEntropyLoss``
  (MONAI's ``DiceCELoss`` with ``softmax=True``).
* Logs training loss and per‑class validation DSC to ``outputs/training_log.csv``.
* Saves the best checkpoint (highest mean DSC) to ``checkpoints/best_model.pth``.
* Shows progress bars via ``tqdm`` and prints a concise per‑epoch summary.
"""

import argparse
import argparse
import argparse
import csv
import os
from pathlib import Path
from typing import Tuple

import torch
import yaml
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# MONAI imports
from monai.networks.nets import SwinUNETR
from monai.losses import DiceCELoss

# No additional MONAI transforms are required for this script

# Local dataset class
from datasets.brats_dataset import BraTSDataset

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Priority order (highest → lowest):
    1. ``--smoke-test`` overrides everything.
    2. ``--epochs`` / ``--batch-size`` override the YAML config.
    3. YAML config values are used as the baseline.
    """
    parser = argparse.ArgumentParser(
        description="Train Swin UNETR on BraTS 2023 GLI dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--max-subjects", type=int, default=None, help="Optional cap on subjects for a quicker run.")
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        dest="epochs",
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        dest="batch_size",
        help="Mini-batch size for training and validation.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        default=False,
        dest="smoke_test",
        help=(
            "Quick sanity-check mode: forces epochs=2, batch-size=1, "
            "and limits the dataset to 10 subjects."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def load_config(config_path: Path) -> dict:
    """Load a simple YAML config file.

    The provided ``base_config.yaml`` contains only scalar values, so a plain
    ``yaml.safe_load`` is sufficient.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_dataloaders(
    data_dir: Path,
    batch_size: int,
    patch_size: Tuple[int, int, int],
    max_subjects: int | None = None,
) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation dataloaders.

    The dataset returns a dictionary with ``"image"`` (4‑channel tensor) and
    ``"label"`` (3‑channel tensor).  No additional transforms are required because
    the dataset already performs normalisation and optional patch extraction.

    Args:
        data_dir: Root directory of the BraTS dataset.
        batch_size: Mini-batch size.
        patch_size: Spatial patch dimensions ``(D, H, W)``.
        max_subjects: If given, cap the dataset to this many subjects (used by
            ``--smoke-test`` to keep runs short).
    """
    full_dataset = BraTSDataset(
        data_dir=data_dir,
        patch_size=patch_size,
        dataset_version="2023",
        augment=True,
    )

    # Optionally limit dataset size for smoke-test runs
    if max_subjects is not None and max_subjects < len(full_dataset):
        full_dataset, _ = random_split(
            full_dataset,
            [max_subjects, len(full_dataset) - max_subjects],
            generator=torch.Generator().manual_seed(42),
        )

    # 80/20 split – deterministic for reproducibility
    val_len = max(1, int(0.2 * len(full_dataset)))
    train_len = len(full_dataset) - val_len
    train_set, val_set = random_split(full_dataset, [train_len, val_len], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, val_loader


def labels_to_class_indices(nested_labels: torch.Tensor) -> torch.Tensor:
    """Convert nested ET/TC/WT masks to exclusive classes: 0 background, 1 NCR/NET, 2 ED, 3 ET."""
    et = nested_labels[:, 0] > 0.5
    tc = nested_labels[:, 1] > 0.5
    wt = nested_labels[:, 2] > 0.5
    classes = torch.zeros_like(nested_labels[:, 0], dtype=torch.long)
    classes[wt & ~tc] = 2
    classes[tc & ~et] = 1
    classes[et] = 3
    return classes.unsqueeze(1)


def compute_metrics(pred_classes: torch.Tensor, nested_labels: torch.Tensor) -> Tuple[float, float, float]:
    """Compute ET, TC, and WT Dice from exclusive class predictions."""
    pred_et = pred_classes == 3
    pred_tc = (pred_classes == 1) | (pred_classes == 3)
    pred_wt = pred_classes != 0
    true_et = nested_labels[:, 0] > 0.5
    true_tc = nested_labels[:, 1] > 0.5
    true_wt = nested_labels[:, 2] > 0.5

    scores = []
    for pred, truth in ((pred_et, true_et), (pred_tc, true_tc), (pred_wt, true_wt)):
        intersection = (pred & truth).sum().float()
        dice = (2.0 * intersection + 1e-5) / (pred.sum() + truth.sum() + 1e-5)
        scores.append(dice.item())
    return tuple(scores)
# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def main() -> None:
    # -------------------------------------------------------------------
    # Parse CLI arguments first, then load YAML config
    # -------------------------------------------------------------------
    args = parse_args()

    cfg = load_config(Path(__file__).parent / "configs" / "base_config.yaml")
    # 96^3 patches reduce compute and memory use on 4 GB GPUs. Must be divisible by 32.
    patch_size = (64, 64, 64)
    lr = cfg["lr"]
    in_channels = cfg["in_channels"]
    out_channels = 4
    data_root = Path("data") / "BraTS-GLI" / "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"

    # CLI flags override YAML values; --smoke-test overrides everything
    if args.smoke_test:
        max_epochs = 2
        batch_size = 1
        max_subjects: int | None = 10
        print("[smoke-test] epochs=2, batch_size=1, max_subjects=10")
    else:
        max_epochs = args.epochs
        batch_size = args.batch_size
        max_subjects = args.max_subjects

    # -------------------------------------------------------------------
    # Prepare output directories
    # -------------------------------------------------------------------
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    log_path = Path("outputs") / "training_log.csv"

    # -------------------------------------------------------------------
    # Dataloaders
    # -------------------------------------------------------------------
    train_loader, val_loader = get_dataloaders(data_root, batch_size, patch_size, max_subjects=max_subjects)

    # -------------------------------------------------------------------
    # Model, loss, optimizer, scheduler, metric
    # -------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SwinUNETR(
        in_channels=in_channels,
        out_channels=out_channels,
        feature_size=24,
        spatial_dims=3,
        use_checkpoint=False,
    ).to(device)

    criterion = DiceCELoss(to_onehot_y=True, softmax=True, include_background=False, lambda_dice=0.5, lambda_ce=0.5)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)


    # -------------------------------------------------------------------
    # CSV header
    # -------------------------------------------------------------------
    with open(log_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["epoch", "train_loss", "val_DSC_ET", "val_DSC_TC", "val_DSC_WT"])

    best_mean_dsc = -1.0

    # -------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------
    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{max_epochs} [train]", leave=False)
        for batch in train_bar:
            images = batch["image"].to(device)
            nested_labels = batch["label"].to(device)
            labels = labels_to_class_indices(nested_labels)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item() * images.size(0)
            train_bar.set_postfix(loss=loss.item())

        epoch_loss /= len(train_loader.dataset)

        # ----------------------------------------------------------------
        # Validation
        # ----------------------------------------------------------------
        model.eval()
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch}/{max_epochs} [val]", leave=False)
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in val_bar:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                    outputs = model(images)
                all_preds.append(outputs.argmax(dim=1))
                all_labels.append(labels)
        preds_tensor = torch.cat(all_preds)
        labels_tensor = torch.cat(all_labels)
        dsc_et, dsc_tc, dsc_wt = compute_metrics(preds_tensor, labels_tensor)
        mean_dsc = (dsc_et + dsc_tc + dsc_wt) / 3.0

        # ----------------------------------------------------------------
        # Logging & checkpointing
        # ----------------------------------------------------------------
        with open(log_path, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([epoch, f"{epoch_loss:.4f}", f"{dsc_et:.4f}", f"{dsc_tc:.4f}", f"{dsc_wt:.4f}"])

        if mean_dsc > best_mean_dsc:
            best_mean_dsc = mean_dsc
            torch.save(model.state_dict(), "checkpoints/best_model.pth")

        # ----------------------------------------------------------------
        # Per‑epoch summary (printed to stdout)
        # ----------------------------------------------------------------
        print(
            f"Epoch {epoch:03d} | Train Loss: {epoch_loss:.4f} | "
            f"Val DSC ET: {dsc_et:.4f} | TC: {dsc_tc:.4f} | WT: {dsc_wt:.4f}"
        )

        scheduler.step()

    print(f"Training complete – best mean DSC: {best_mean_dsc:.4f} (saved to checkpoints/best_model.pth)")


if __name__ == "__main__":
    main()








