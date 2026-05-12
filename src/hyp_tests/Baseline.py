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
    METRIC_COLUMNS,
    RESULTS_DIR,
    build_experiment_config,
    build_normalization_summary,
    build_runs_dataframe,
    build_standard_run_config,
    build_train_loader,
    final_metrics,
    get_shared_settings,
    iter_repeat_seeds,
    resolve_data_dir,
    save_experiment_outputs,
    set_seed,
)
from src.model import BaselineCNN
from src.training import FitConfig, run_model


RUNS_CSV_PATH = RESULTS_DIR / "baseline_runs.csv"
SUMMARY_JSON_PATH = RESULTS_DIR / "baseline_summary.json"

SETTINGS = get_shared_settings()


def run_single_experiment(
    repeat_idx: int,
    seed: int,
    train_loader,
    settings,
    device: str,
) -> dict[str, float | int | str]:
    run_name = f"Baseline_HypTests-repeat-{repeat_idx:02d}"
    config = build_standard_run_config(
        settings,
        seed=seed,
        run_name=run_name,
        device=device,
        tags=["baseline-shared", "hyp-tests", "baseline"],
    )
    result = run_model(
        model_factory=BaselineCNN,
        model_name="BaselineCNN",
        config=config,
        fit_config=FitConfig(epochs=settings.epochs),
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
    settings = SETTINGS
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = resolve_data_dir(settings.data_dir)
    pin_memory = torch.cuda.is_available()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    run_records: list[dict[str, float | int | str]] = []
    train_loader, mean, std = build_train_loader(
        data_dir=data_dir,
        batch_size=settings.batch_size,
        num_workers=settings.num_workers,
        pin_memory=pin_memory,
    )
    normalization_stats = build_normalization_summary(mean, std)

    for repeat_idx, seed in iter_repeat_seeds(settings):
        set_seed(seed)
        run_records.append(
            run_single_experiment(
                repeat_idx=repeat_idx,
                seed=seed,
                train_loader=train_loader,
                settings=settings,
                device=device,
            )
        )

    runs_df = build_runs_dataframe(run_records)
    summary_metrics = {
        f"baseline_{metric}_mean": float(runs_df[metric].mean()) for metric in METRIC_COLUMNS
    }
    summary_metrics.update(
        {
            f"baseline_{metric}_std": float(
                runs_df[metric].std(ddof=1) if len(runs_df) > 1 else 0.0
            )
            for metric in METRIC_COLUMNS
        }
    )

    summary_payload = {
        "config": build_experiment_config(
            settings,
            device=device,
            experiment_name="Baseline_HypTests",
        ),
        "normalization": normalization_stats,
        "variants": ["baseline"],
        "summary_metrics": summary_metrics,
    }

    save_experiment_outputs(
        runs_df,
        runs_csv_path=RUNS_CSV_PATH,
        summary_json_path=SUMMARY_JSON_PATH,
        summary_payload=summary_payload,
    )


if __name__ == "__main__":
    main()
