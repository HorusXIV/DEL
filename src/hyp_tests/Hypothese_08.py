from __future__ import annotations

import torch

from common import (
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
from deeplearning.model import HeInitCNN, XavierInitCNN
from deeplearning.training import FitConfig, run_model


RUNS_CSV_PATH = RESULTS_DIR / "hypothesis_08_runs.csv"
SUMMARY_JSON_PATH = RESULTS_DIR / "hypothesis_08_summary.json"
SETTINGS = get_shared_settings()


def run_single_experiment(
    repeat_idx: int,
    seed: int,
    train_loader,
    test_loader,
    settings,
    device: str,
    model_factory,
    model_name: str,
    model_variant: str,
) -> dict[str, float | int | str]:
    run_name = f"Hypothese_08-{model_variant}-repeat-{repeat_idx:02d}"
    config = build_standard_run_config(
        settings,
        seed=seed,
        run_name=run_name,
        device=device,
        tags=["hypothesis-08", "initialization", model_variant],
    )
    result = run_model(
        model_factory=model_factory,
        model_name=model_name,
        config=config,
        fit_config=FitConfig(epochs=settings.epochs),
        train_loader=train_loader,
        test_loader=test_loader,
        extra_wandb_config={
            "experiment_name": "Hypothese_08",
            "hypothesis_id": "08",
            "model_variant": model_variant,
            "repeat_idx": repeat_idx,
            "is_summary_run": False,
            "initialization": model_variant,
        },
    )
    return {
        "repeat_idx": repeat_idx,
        "seed": seed,
        "model_variant": model_variant,
        "model_name": model_name,
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
                model_factory=XavierInitCNN,
                model_name="XavierInitCNN",
                model_variant="xavier",
            )
        )

        set_seed(seed)
        experiment_records.append(
            run_single_experiment(
                repeat_idx=repeat_idx,
                seed=seed,
                train_loader=train_loader,
                test_loader=test_loader,
                settings=settings,
                device=device,
                model_factory=HeInitCNN,
                model_name="HeInitCNN",
                model_variant="he",
            )
        )

    runs_df = build_runs_dataframe(experiment_records)
    summary_payload = {
        "config": build_experiment_config(
            settings,
            device=device,
            experiment_name="Hypothese_08",
            reference_model="XavierInitCNN",
            comparison_model="HeInitCNN",
            reference_variant="xavier",
        ),
        "normalization": normalization_stats,
        "variants": ["xavier", "he"],
    }

    save_experiment_outputs(
        runs_df,
        runs_csv_path=RUNS_CSV_PATH,
        summary_json_path=SUMMARY_JSON_PATH,
        summary_payload=summary_payload,
    )


if __name__ == "__main__":
    main()
