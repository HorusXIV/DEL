from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import ttest_rel
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = Path(__file__).resolve().parent / "results"
BASELINE_RUNS_CSV_PATH = RESULTS_DIR / "baseline_runs.csv"
BASELINE_SUMMARY_JSON_PATH = RESULTS_DIR / "baseline_summary.json"


def resolve_data_dir(data_dir: str) -> Path:
    path = Path(data_dir)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_channel_stats(dataset, num_workers: int) -> tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    channel_sum = torch.zeros(3, dtype=torch.float64)
    channel_sq_sum = torch.zeros(3, dtype=torch.float64)
    total_pixels = 0

    for images, _ in loader:
        images = images.to(dtype=torch.float64)
        channel_sum += images.sum(dim=(0, 2, 3))
        channel_sq_sum += (images ** 2).sum(dim=(0, 2, 3))
        total_pixels += images.size(0) * images.size(2) * images.size(3)

    mean = (channel_sum / total_pixels).to(dtype=torch.float32)
    std = ((channel_sq_sum / total_pixels) - mean.to(dtype=torch.float64) ** 2).sqrt()
    return mean, std.to(dtype=torch.float32)


def build_train_loader(
    data_dir: Path,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> tuple[DataLoader, torch.Tensor, torch.Tensor]:
    base_dataset = datasets.SVHN(
        root=str(data_dir),
        split="train",
        download=True,
        transform=transforms.ToTensor(),
    )
    mean, std = compute_channel_stats(base_dataset, num_workers=num_workers)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean.tolist(), std=std.tolist()),
        ]
    )
    train_dataset = datasets.SVHN(
        root=str(data_dir),
        split="train",
        download=True,
        transform=transform,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, mean, std


def split_indices(
    num_samples: int,
    val_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1.")

    val_size = max(1, int(num_samples * val_fraction))
    if val_size >= num_samples:
        raise ValueError("val_fraction leaves no samples for training.")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(num_samples)
    val_indices = indices[:val_size].tolist()
    train_indices = indices[val_size:].tolist()
    return train_indices, val_indices


def build_augmented_train_val_loaders(
    data_dir: Path,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    val_fraction: float,
    seed: int,
    train_transform,
    val_transform,
) -> tuple[DataLoader, DataLoader, torch.Tensor, torch.Tensor]:
    base_dataset = datasets.SVHN(
        root=str(data_dir),
        split="train",
        download=True,
        transform=transforms.ToTensor(),
    )
    mean, std = compute_channel_stats(base_dataset, num_workers=num_workers)

    train_dataset = datasets.SVHN(
        root=str(data_dir),
        split="train",
        download=True,
        transform=train_transform(mean, std),
    )
    val_dataset = datasets.SVHN(
        root=str(data_dir),
        split="train",
        download=True,
        transform=val_transform(mean, std),
    )

    train_indices, val_indices = split_indices(
        num_samples=len(train_dataset),
        val_fraction=val_fraction,
        seed=seed,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        Subset(train_dataset, train_indices),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator,
    )
    val_loader = DataLoader(
        Subset(val_dataset, val_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, mean, std


def final_metrics(result: dict) -> dict[str, float]:
    history = result["history"]
    return {key: values[-1] for key, values in history.items()}


def load_shared_baseline() -> tuple[pd.DataFrame, dict]:
    if not BASELINE_RUNS_CSV_PATH.exists() or not BASELINE_SUMMARY_JSON_PATH.exists():
        raise FileNotFoundError(
            "Bitte zuerst src/hyp_tests/Baseline.py ausfuehren, damit die geteilte Baseline gespeichert ist."
        )

    baseline_runs_df = pd.read_csv(BASELINE_RUNS_CSV_PATH).sort_values("repeat_idx")
    baseline_summary = json.loads(
        BASELINE_SUMMARY_JSON_PATH.read_text(encoding="utf-8")
    )
    return baseline_runs_df, baseline_summary


def build_paired_summary(
    runs_df: pd.DataFrame,
    *,
    reference_label: str = "baseline",
    variant_label: str,
    alpha: float,
) -> dict[str, float | int | bool]:
    paired = (
        runs_df.pivot(index="repeat_idx", columns="model_variant", values="val_macro_f1")
        .sort_index()
        .dropna()
    )
    deltas = paired[variant_label] - paired[reference_label]
    test_result = ttest_rel(
        paired[variant_label].to_numpy(),
        paired[reference_label].to_numpy(),
        alternative="greater",
    )
    return {
        "repeats": int(len(paired)),
        "alpha": float(alpha),
        f"{reference_label}_mean_val_macro_f1": float(paired[reference_label].mean()),
        f"{reference_label}_std_val_macro_f1": float(
            paired[reference_label].std(ddof=1) if len(paired) > 1 else 0.0
        ),
        f"{variant_label}_mean_val_macro_f1": float(paired[variant_label].mean()),
        f"{variant_label}_std_val_macro_f1": float(
            paired[variant_label].std(ddof=1) if len(paired) > 1 else 0.0
        ),
        "delta_mean_val_macro_f1": float(deltas.mean()),
        "delta_std_val_macro_f1": float(deltas.std(ddof=1) if len(deltas) > 1 else 0.0),
        "t_statistic": float(test_result.statistic),
        "p_value": float(test_result.pvalue),
        "reject_h0": bool(test_result.pvalue < alpha),
    }
