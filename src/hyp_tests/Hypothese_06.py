from __future__ import annotations

import sys
from pathlib import Path

import torch
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.hyp_tests.common import (
    RESULTS_DIR,
    build_augmented_train_val_loaders,
    build_experiment_config,
    build_normalization_summary,
    build_runs_dataframe,
    build_standard_run_config,
    final_metrics,
    get_shared_settings,
    iter_repeat_seeds,
    resolve_data_dir,
    save_experiment_outputs,
    set_seed,
)
from src.model import BaselineCNN
from src.training import FitConfig, run_model


RUNS_CSV_PATH = RESULTS_DIR / "hypothesis_06_runs.csv"
SUMMARY_JSON_PATH = RESULTS_DIR / "hypothesis_06_summary.json"
SETTINGS = get_shared_settings()
AUGMENTATION_CONFIG = {
    "random_affine_degrees": 8,
    "random_affine_translate": [0.08, 0.08],
    "color_jitter_brightness": 0.15,
    "color_jitter_contrast": 0.10,
}


def build_train_transform(mean: torch.Tensor, std: torch.Tensor):
    return transforms.Compose(
        [
            transforms.RandomAffine(
                degrees=AUGMENTATION_CONFIG["random_affine_degrees"],
                translate=tuple(AUGMENTATION_CONFIG["random_affine_translate"]),
            ),
            transforms.ColorJitter(
                brightness=AUGMENTATION_CONFIG["color_jitter_brightness"],
                contrast=AUGMENTATION_CONFIG["color_jitter_contrast"],
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean.tolist(), std=std.tolist()),
        ]
    )


def build_val_transform(mean: torch.Tensor, std: torch.Tensor):
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean.tolist(), std=std.tolist()),
        ]
    )


def run_single_experiment(
    repeat_idx: int,
    seed: int,
    train_loader,
    val_loader,
    settings,
    device: str,
) -> dict[str, float | int | str]:
    run_name = f"Hypothese_06-aug-repeat-{repeat_idx:02d}"
    config = build_standard_run_config(
        settings,
        seed=seed,
        run_name=run_name,
        device=device,
        tags=["hypothesis-06", "augmentation", "aug"],
    )
    result = run_model(
        model_factory=BaselineCNN,
        model_name="BaselineCNN",
        config=config,
        fit_config=FitConfig(epochs=settings.epochs),
        train_loader=train_loader,
        val_loader=val_loader,
        extra_wandb_config={
            "experiment_name": "Hypothese_06",
            "hypothesis_id": "06",
            "model_variant": "aug",
            "repeat_idx": repeat_idx,
            "is_summary_run": False,
            "augmentation": AUGMENTATION_CONFIG,
        },
    )
    return {
        "repeat_idx": repeat_idx,
        "seed": seed,
        "model_variant": "aug",
        "model_name": "BaselineCNN",
        **final_metrics(result),
    }


def main() -> None:
    settings = SETTINGS
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = resolve_data_dir(settings.data_dir)
    pin_memory = torch.cuda.is_available()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    train_val_cache: dict[int, tuple] = {}
    normalization_stats: dict[str, object] | None = None

    experiment_records: list[dict[str, float | int | str]] = []
    for repeat_idx, seed in iter_repeat_seeds(settings):
        set_seed(seed)

        if seed not in train_val_cache:
            train_val_cache[seed] = build_augmented_train_val_loaders(
                data_dir=data_dir,
                batch_size=settings.batch_size,
                num_workers=settings.num_workers,
                pin_memory=pin_memory,
                val_fraction=settings.val_fraction,
                seed=seed,
                train_transform=build_train_transform,
                val_transform=build_val_transform,
            )

        train_loader, val_loader, mean, std = train_val_cache[seed]
        normalization_stats = build_normalization_summary(mean, std)

        experiment_records.append(
            run_single_experiment(
                repeat_idx=repeat_idx,
                seed=seed,
                train_loader=train_loader,
                val_loader=val_loader,
                settings=settings,
                device=device,
            )
        )

    runs_df = build_runs_dataframe(experiment_records)
    summary_payload = {
        "config": build_experiment_config(
            settings,
            device=device,
            experiment_name="Hypothese_06",
            comparison_model="BaselineCNN+Augmentation",
            reference_variant="baseline",
            augmentation=AUGMENTATION_CONFIG,
        ),
        "normalization": normalization_stats,
        "variants": ["aug"],
    }

    save_experiment_outputs(
        runs_df,
        runs_csv_path=RUNS_CSV_PATH,
        summary_json_path=SUMMARY_JSON_PATH,
        summary_payload=summary_payload,
    )


if __name__ == "__main__":
    main()
