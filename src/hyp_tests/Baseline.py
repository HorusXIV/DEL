from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.hyp_tests.common import (
    RESULTS_DIR,
    build_train_loader,
    final_metrics,
    resolve_data_dir,
    set_seed,
)
from src.model import BaselineCNN
from src.training import FitConfig, RunConfig, run_model


RUNS_CSV_PATH = RESULTS_DIR / "baseline_runs.csv"
SUMMARY_JSON_PATH = RESULTS_DIR / "baseline_summary.json"

# Beste Konfiguration aus der Hyperparametersuche im Notebook:
# learning_rate=0.05, batch_size=128, epochs=30
EPOCHS = 30
REPEATS = 5
LEARNING_RATE = 5e-2
BATCH_SIZE = 128
VAL_FRACTION = 0.1
BASE_SEED = 42
DATA_DIR = "data"
WANDB_PROJECT = "DEL"
WANDB_ENTITY = "lukas-breiter-fachhochschule-nordwestschweiz-fhnw"
NUM_WORKERS = 4
USE_WANDB = True


def run_single_experiment(
    repeat_idx: int,
    seed: int,
    train_loader,
    device: str,
) -> dict[str, float | int | str]:
    run_name = f"Baseline_HypTests-repeat-{repeat_idx:02d}"
    config = RunConfig(
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        data_dir=str(resolve_data_dir(DATA_DIR)),
        val_fraction=VAL_FRACTION,
        random_seed=seed,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        use_wandb=USE_WANDB,
        wandb_project=WANDB_PROJECT,
        wandb_entity=WANDB_ENTITY,
        run_name=run_name,
        device=device,
        tags=["baseline-shared", "hyp-tests", "baseline"],
    )
    result = run_model(
        model_factory=BaselineCNN,
        model_name="BaselineCNN",
        config=config,
        fit_config=FitConfig(epochs=EPOCHS),
        train_loader=train_loader,
        extra_wandb_config={
            "experiment_name": "Baseline_HypTests",
            "is_summary_run": False,
            "model_variant": "baseline",
            "repeat_idx": repeat_idx,
        },
    )
    return {
        "repeat_idx": repeat_idx,
        "seed": seed,
        "model_variant": "baseline",
        "model_name": "BaselineCNN",
        **final_metrics(result),
    }


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = resolve_data_dir(DATA_DIR)
    pin_memory = torch.cuda.is_available()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    run_records: list[dict[str, float | int | str]] = []
    train_loader, mean, std = build_train_loader(
        data_dir=data_dir,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )
    normalization_stats = {
        "mean": [float(value) for value in mean.tolist()],
        "std": [float(value) for value in std.tolist()],
    }

    for repeat_idx in range(1, REPEATS + 1):
        seed = BASE_SEED + repeat_idx - 1
        set_seed(seed)
        run_records.append(
            run_single_experiment(
                repeat_idx=repeat_idx,
                seed=seed,
                train_loader=train_loader,
                device=device,
            )
        )

    runs_df = pd.DataFrame(run_records).sort_values(["repeat_idx", "model_variant"])
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
    grouped_summary = runs_df[metric_columns].agg(["mean", "std"]).round(4)
    summary_metrics = {
        f"baseline_{metric}_mean": float(runs_df[metric].mean())
        for metric in metric_columns
    }
    summary_metrics.update(
        {
            f"baseline_{metric}_std": float(
                runs_df[metric].std(ddof=1) if len(runs_df) > 1 else 0.0
            )
            for metric in metric_columns
        }
    )

    summary_payload = {
        "config": {
            "epochs": EPOCHS,
            "repeats": REPEATS,
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "val_fraction": VAL_FRACTION,
            "base_seed": BASE_SEED,
            "num_workers": NUM_WORKERS,
            "use_wandb": USE_WANDB,
            "device": device,
            "data_dir": str(data_dir),
            "wandb_project": WANDB_PROJECT,
            "wandb_entity": WANDB_ENTITY,
            "experiment_name": "Baseline_HypTests",
        },
        "normalization": normalization_stats,
        "summary_metrics": summary_metrics,
    }

    runs_df.to_csv(RUNS_CSV_PATH, index=False)
    SUMMARY_JSON_PATH.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    print("Geteilte Baseline-Run-Metriken:")
    print(runs_df.to_string(index=False))
    print()
    print("Aggregierte Baseline-Kennzahlen:")
    print(grouped_summary.to_string())
    print()
    print(f"CSV gespeichert unter: {RUNS_CSV_PATH}")
    print(f"JSON gespeichert unter: {SUMMARY_JSON_PATH}")


if __name__ == "__main__":
    main()
