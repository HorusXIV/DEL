from __future__ import annotations

from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from common import (
    RESULTS_DIR,
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
    split_indices,
)
from deeplearning.model import TransferLearningCNN
from deeplearning.training import FitConfig, run_model


RUNS_CSV_PATH = RESULTS_DIR / "hypothesis_10_runs.csv"
SUMMARY_JSON_PATH = RESULTS_DIR / "hypothesis_10_summary.json"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
INPUT_SIZE = 224
SETTINGS = get_shared_settings()


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


def build_transfer_learning_test_loader(
    data_dir: Path,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    transform = transforms.Compose(
        [
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    test_dataset = datasets.SVHN(
        root=str(data_dir),
        split="test",
        download=True,
        transform=transform,
    )
    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def run_single_experiment(
    repeat_idx: int,
    seed: int,
    train_loader,
    val_loader,
    test_loader,
    settings,
    device: str,
) -> dict[str, float | int | str]:
    run_name = f"Hypothese_10-transfer-learning-repeat-{repeat_idx:02d}"
    config = build_standard_run_config(
        settings,
        seed=seed,
        run_name=run_name,
        device=device,
        tags=["hypothesis-10", "transfer-learning", "resnet18"],
    )
    result = run_model(
        model_factory=TransferLearningCNN,
        model_name="TransferLearningCNN",
        config=config,
        fit_config=FitConfig(epochs=settings.epochs),
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        extra_wandb_config={
            "experiment_name": "Hypothese_10",
            "hypothesis_id": "10",
            "model_variant": "transfer-learning",
            "repeat_idx": repeat_idx,
            "is_summary_run": False,
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
    settings = SETTINGS
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = resolve_data_dir(settings.data_dir)
    pin_memory = torch.cuda.is_available()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    train_val_cache: dict[int, tuple[DataLoader, DataLoader]] = {}
    test_loader = build_transfer_learning_test_loader(
        data_dir=data_dir,
        batch_size=settings.batch_size,
        num_workers=settings.num_workers,
        pin_memory=pin_memory,
    )
    normalization_stats = build_normalization_summary(
        IMAGENET_MEAN,
        IMAGENET_STD,
        input_size=INPUT_SIZE,
    )

    experiment_records: list[dict[str, float | int | str]] = []
    for repeat_idx, seed in iter_repeat_seeds(settings):
        set_seed(seed)

        if seed not in train_val_cache:
            train_val_cache[seed] = build_transfer_learning_loaders(
                data_dir=data_dir,
                batch_size=settings.batch_size,
                num_workers=settings.num_workers,
                pin_memory=pin_memory,
                val_fraction=settings.val_fraction,
                seed=seed,
            )

        train_loader, val_loader = train_val_cache[seed]
        experiment_records.append(
            run_single_experiment(
                repeat_idx=repeat_idx,
                seed=seed,
                train_loader=train_loader,
                val_loader=val_loader,
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
            experiment_name="Hypothese_10",
            comparison_model="TransferLearningCNN",
            reference_variant="baseline",
            backbone="resnet18",
            pretrained=True,
            freeze_backbone=True,
            input_size=INPUT_SIZE,
        ),
        "normalization": normalization_stats,
        "variants": ["transfer-learning"],
    }

    save_experiment_outputs(
        runs_df,
        runs_csv_path=RUNS_CSV_PATH,
        summary_json_path=SUMMARY_JSON_PATH,
        summary_payload=summary_payload,
    )


if __name__ == "__main__":
    main()
