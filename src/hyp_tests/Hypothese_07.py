from __future__ import annotations

import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.hyp_tests.common import (
    RESULTS_DIR,
    build_experiment_config,
    build_normalization_summary,
    build_runs_dataframe,
    build_standard_run_config,
    build_test_loader,
    build_train_loader,
    final_metrics,
    get_shared_settings,
    iter_repeat_seeds,
    resolve_data_dir,
    save_experiment_outputs,
    set_seed,
)
from src.model import BatchNormCNN
from src.training import FitConfig, run_model


RUNS_CSV_PATH = RESULTS_DIR / "hypothesis_07_runs.csv"
SUMMARY_JSON_PATH = RESULTS_DIR / "hypothesis_07_summary.json"
SETTINGS = get_shared_settings()


def run_single_experiment(
    repeat_idx: int,
    seed: int,
    train_loader,
    test_loader,
    settings,
    device: str,
) -> dict[str, float | int | str]:
    run_name = f"Hypothese_07-batchnorm-repeat-{repeat_idx:02d}"
    config = build_standard_run_config(
        settings,
        seed=seed,
        run_name=run_name,
        device=device,
        tags=["hypothesis-07", "batchnorm", "bn"],
    )
    result = run_model(
        model_factory=BatchNormCNN,
        model_name="BatchNormCNN",
        config=config,
        fit_config=FitConfig(epochs=settings.epochs),
        train_loader=train_loader,
        test_loader=test_loader,
        extra_wandb_config={
            "experiment_name": "Hypothese_07",
            "hypothesis_id": "07",
            "model_variant": "batchnorm",
            "repeat_idx": repeat_idx,
            "is_summary_run": False,
        },
    )
    return {
        "repeat_idx": repeat_idx,
        "seed": seed,
        "model_variant": "batchnorm",
        "model_name": "BatchNormCNN",
        **final_metrics(result),
    }


def main() -> None:
    settings = SETTINGS
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = resolve_data_dir(settings.data_dir)
    pin_memory = torch.cuda.is_available()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    train_loader, mean, std = build_train_loader(
        data_dir=data_dir,
        batch_size=settings.batch_size,
        num_workers=settings.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = build_test_loader(
        data_dir=data_dir,
        batch_size=settings.batch_size,
        num_workers=settings.num_workers,
        pin_memory=pin_memory,
        mean=mean,
        std=std,
    )
    normalization_stats = build_normalization_summary(mean, std)

    experiment_records: list[dict[str, float | int | str]] = []
    for repeat_idx, seed in iter_repeat_seeds(settings):
        set_seed(seed)
        experiment_records.append(
            run_single_experiment(
                repeat_idx=repeat_idx,
                seed=seed,
                train_loader=train_loader,
                test_loader=test_loader,
                settings=settings,
                device=device,
            )
        )

    runs_df = build_runs_dataframe(experiment_records)
    summary_payload = {
        "config": build_experiment_config(
            settings,
            device=device,
            experiment_name="Hypothese_07",
            comparison_model="BatchNormCNN",
            reference_variant="baseline",
        ),
        "normalization": normalization_stats,
        "variants": ["batchnorm"],
    }

    save_experiment_outputs(
        runs_df,
        runs_csv_path=RUNS_CSV_PATH,
        summary_json_path=SUMMARY_JSON_PATH,
        summary_payload=summary_payload,
    )


if __name__ == "__main__":
    main()
