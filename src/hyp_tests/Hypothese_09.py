from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.optim as optim


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.hyp_tests.common import (
    BASELINE_RUNS_CSV_PATH,
    BASELINE_SUMMARY_JSON_PATH,
    RESULTS_DIR,
    build_train_loader,
    final_metrics,
    load_shared_baseline,
    resolve_data_dir,
    set_seed,
)
from src.model import BaselineCNN
from src.training import FitConfig, RunConfig, run_model


RUNS_CSV_PATH = RESULTS_DIR / "hypothesis_09_runs.csv"
SUMMARY_JSON_PATH = RESULTS_DIR / "hypothesis_09_summary.json"


def run_single_experiment(
    repeat_idx: int,
    seed: int,
    train_loader,
    baseline_config: dict,
    device: str,
) -> dict[str, float | int | str]:
    run_name = f"Hypothese_09-adam-repeat-{repeat_idx:02d}"
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
        tags=["hypothesis-09", "optimizer", "adam"],
    )
    result = run_model(
        model_factory=BaselineCNN,
        model_name="BaselineCNN",
        config=config,
        fit_config=FitConfig(epochs=int(baseline_config["epochs"])),
        train_loader=train_loader,
        optimizer_factory=lambda params: optim.Adam(
            params,
            lr=float(baseline_config["learning_rate"]),
        ),
        extra_wandb_config={
            "experiment_name": "Hypothese_09",
            "hypothesis_id": "09",
            "model_variant": "adam",
            "repeat_idx": repeat_idx,
            "is_summary_run": False,
            "baseline_source_experiment": "Baseline_HypTests",
            "reference_variant": "baseline",
        },
    )
    return {
        "repeat_idx": repeat_idx,
        "seed": seed,
        "model_variant": "adam",
        "model_name": "BaselineCNN",
        **final_metrics(result),
    }


def main() -> None:
    baseline_runs_df, baseline_summary = load_shared_baseline()
    baseline_config = baseline_summary["config"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = resolve_data_dir(str(baseline_config["data_dir"]))
    pin_memory = torch.cuda.is_available()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    train_loader, mean, std = build_train_loader(
        data_dir=data_dir,
        batch_size=int(baseline_config["batch_size"]),
        num_workers=int(baseline_config["num_workers"]),
        pin_memory=pin_memory,
    )
    normalization_stats = {
        "mean": [float(value) for value in mean.tolist()],
        "std": [float(value) for value in std.tolist()],
    }

    experiment_records: list[dict[str, float | int | str]] = []
    for row in baseline_runs_df.itertuples(index=False):
        set_seed(int(row.seed))
        experiment_records.append(
            run_single_experiment(
                repeat_idx=int(row.repeat_idx),
                seed=int(row.seed),
                train_loader=train_loader,
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
            "experiment_name": "Hypothese_09",
            "reference_model": "BaselineCNN+SGD",
            "comparison_model": "BaselineCNN+Adam",
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
