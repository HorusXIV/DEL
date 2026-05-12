from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.hyp_tests.common import (
    BASELINE_RUNS_CSV_PATH,
    BASELINE_SUMMARY_JSON_PATH,
    RESULTS_DIR,
    final_metrics,
    load_shared_baseline,
    resolve_data_dir,
    set_seed,
    split_indices,
)
from src.model import TransferLearningCNN
from src.training import FitConfig, RunConfig, run_model


RUNS_CSV_PATH = RESULTS_DIR / "hypothesis_10_runs.csv"
SUMMARY_JSON_PATH = RESULTS_DIR / "hypothesis_10_summary.json"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
INPUT_SIZE = 224


def build_transfer_learning_loaders(
    data_dir: Path,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    val_fraction: float,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose(
        [
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    train_dataset = datasets.SVHN(
        root=str(data_dir),
        split="train",
        download=True,
        transform=transform,
    )
    val_dataset = datasets.SVHN(
        root=str(data_dir),
        split="train",
        download=True,
        transform=transform,
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
    return train_loader, val_loader


def run_single_experiment(
    repeat_idx: int,
    seed: int,
    train_loader,
    val_loader,
    baseline_config: dict,
    device: str,
) -> dict[str, float | int | str]:
    run_name = f"Hypothese_10-transfer-learning-repeat-{repeat_idx:02d}"
    config = RunConfig(
        learning_rate=float(baseline_config["learning_rate"]),
        batch_size=int(baseline_config["batch_size"]),
        epochs=int(baseline_config["epochs"]),
        data_dir=str(resolve_data_dir(str(baseline_config["data_dir"]))),
        val_fraction=float(baseline_config["val_fraction"]),
        random_seed=seed,
        num_workers=int(baseline_config["num_workers"]),
        pin_memory=torch.cuda.is_available(),
        use_wandb=bool(baseline_config["use_wandb"]),
        wandb_project=str(baseline_config["wandb_project"]),
        wandb_entity=baseline_config["wandb_entity"],
        run_name=run_name,
        device=device,
        tags=["hypothesis-10", "transfer-learning", "resnet18"],
    )
    result = run_model(
        model_factory=TransferLearningCNN,
        model_name="TransferLearningCNN",
        config=config,
        fit_config=FitConfig(epochs=int(baseline_config["epochs"])),
        train_loader=train_loader,
        val_loader=val_loader,
        extra_wandb_config={
            "experiment_name": "Hypothese_10",
            "hypothesis_id": "10",
            "model_variant": "transfer-learning",
            "repeat_idx": repeat_idx,
            "is_summary_run": False,
            "baseline_source_experiment": "Baseline_HypTests",
            "backbone": "resnet18",
            "pretrained": True,
            "freeze_backbone": True,
            "input_size": INPUT_SIZE,
            "normalization_mean": IMAGENET_MEAN,
            "normalization_std": IMAGENET_STD,
        },
    )
    return {
        "repeat_idx": repeat_idx,
        "seed": seed,
        "model_variant": "transfer-learning",
        "model_name": "TransferLearningCNN",
        **final_metrics(result),
    }


def main() -> None:
    baseline_runs_df, baseline_summary = load_shared_baseline()
    baseline_config = baseline_summary["config"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = resolve_data_dir(str(baseline_config["data_dir"]))
    pin_memory = torch.cuda.is_available()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    train_val_cache: dict[int, tuple[DataLoader, DataLoader]] = {}
    normalization_stats = {
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
        "input_size": INPUT_SIZE,
    }

    experiment_records: list[dict[str, float | int | str]] = []
    for row in baseline_runs_df.itertuples(index=False):
        seed = int(row.seed)
        set_seed(seed)

        if seed not in train_val_cache:
            train_val_cache[seed] = build_transfer_learning_loaders(
                data_dir=data_dir,
                batch_size=int(baseline_config["batch_size"]),
                num_workers=int(baseline_config["num_workers"]),
                pin_memory=pin_memory,
                val_fraction=float(baseline_config["val_fraction"]),
                seed=seed,
            )

        train_loader, val_loader = train_val_cache[seed]
        experiment_records.append(
            run_single_experiment(
                repeat_idx=int(row.repeat_idx),
                seed=seed,
                train_loader=train_loader,
                val_loader=val_loader,
                baseline_config=baseline_config,
                device=device,
            )
        )

    experiment_df = pd.DataFrame(experiment_records)
    runs_df = (
        pd.concat([baseline_runs_df, experiment_df], ignore_index=True)
        .sort_values(["repeat_idx", "model_variant"])
        .reset_index(drop=True)
    )

    metric_columns = [
        "train_loss",
        "train_accuracy",
        "train_macro_precision",
        "train_macro_recall",
        "train_macro_f1",
        "val_loss",
        "val_accuracy",
        "val_macro_precision",
        "val_macro_recall",
        "val_macro_f1",
    ]
    grouped_summary = (
        runs_df.groupby("model_variant")[metric_columns]
        .agg(["mean", "std"])
        .round(4)
    )

    summary_payload = {
        "config": {
            **baseline_config,
            "device": device,
            "experiment_name": "Hypothese_10",
            "comparison_model": "TransferLearningCNN",
            "backbone": "resnet18",
            "pretrained": True,
            "freeze_backbone": True,
            "input_size": INPUT_SIZE,
        },
        "normalization": normalization_stats,
        "baseline_source": {
            "runs_csv": str(BASELINE_RUNS_CSV_PATH),
            "summary_json": str(BASELINE_SUMMARY_JSON_PATH),
        },
    }

    runs_df.to_csv(RUNS_CSV_PATH, index=False)
    SUMMARY_JSON_PATH.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    print("Run-Metriken pro Wiederholung:")
    print(runs_df.to_string(index=False))
    print()
    print("Aggregierte Kennzahlen pro Modell:")
    print(grouped_summary.to_string())
    print()
    print(f"CSV gespeichert unter: {RUNS_CSV_PATH}")
    print(f"JSON gespeichert unter: {SUMMARY_JSON_PATH}")


if __name__ == "__main__":
    main()
