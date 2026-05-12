from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset

from src.metrics import metrics_from_confusion_matrix
from src.model import BaselineCNN
from src.wandb_tools import WandbLogger


@dataclass
class FitConfig:
    epochs: int = 10
    train_batches_per_epoch: int | None = None
    val_batches_per_epoch: int | None = None
    single_batch_mode: bool = False
    single_batch_repeats: int = 1


@dataclass
class RunConfig:
    learning_rate: float = 1e-2
    batch_size: int = 64
    epochs: int = 5
    data_dir: str | Path = "data"
    val_fraction: float = 0.1
    random_seed: int = 42
    num_workers: int = 2
    pin_memory: bool = True
    use_wandb: bool = True
    wandb_project: str = "DEL"
    wandb_entity: str | None = None
    run_name: str | None = None
    device: str | None = None
    tags: list[str] | None = None


def _split_train_dataset(
    dataset: Dataset,
    val_fraction: float,
    seed: int,
) -> tuple[Subset, Subset]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1.")

    num_samples = len(dataset)
    val_size = max(1, int(num_samples * val_fraction))
    if val_size >= num_samples:
        raise ValueError("val_fraction leaves no samples for training.")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(num_samples)
    val_indices = indices[:val_size].tolist()
    train_indices = indices[val_size:].tolist()
    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def _rebuild_loader_from_base(
    base_loader: DataLoader,
    dataset: Dataset,
    *,
    shuffle: bool,
    seed: int | None = None,
) -> DataLoader:
    if base_loader.batch_size is None:
        raise ValueError(
            "train_loader must define batch_size so training.py can rebuild train/validation loaders after splitting."
        )

    generator = None
    if shuffle and seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=base_loader.batch_size,
        shuffle=shuffle,
        num_workers=base_loader.num_workers,
        collate_fn=base_loader.collate_fn,
        pin_memory=base_loader.pin_memory,
        drop_last=base_loader.drop_last if shuffle else False,
        timeout=base_loader.timeout,
        worker_init_fn=base_loader.worker_init_fn,
        multiprocessing_context=base_loader.multiprocessing_context,
        prefetch_factor=base_loader.prefetch_factor
        if base_loader.num_workers > 0
        else None,
        persistent_workers=base_loader.persistent_workers,
        pin_memory_device=base_loader.pin_memory_device,
        generator=generator,
    )


def _split_train_loader(
    train_loader: DataLoader,
    *,
    val_fraction: float,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    train_subset, val_subset = _split_train_dataset(
        dataset=train_loader.dataset,
        val_fraction=val_fraction,
        seed=seed,
    )
    train_split_loader = _rebuild_loader_from_base(
        train_loader,
        train_subset,
        shuffle=True,
        seed=seed,
    )
    val_loader = _rebuild_loader_from_base(
        train_loader,
        val_subset,
        shuffle=False,
    )
    return train_split_loader, val_loader


class Trainer:
    """
    Trainer class to handle training and evaluation of the model.
    Arguments:
    - model: The PyTorch model to train.
    - criterion: The loss function.
    - optimizer: The optimizer for training.
    - device: The device to run the training on (e.g., "cpu" or "cuda").
    - logger: Optional WandbLogger for logging metrics.
    - num_classes: The number of classes in the classification task.

    Methods:
    - train_one_epoch: Trains the model for one epoch on the given data loader.
    - evaluate: Evaluates the model on the given data loader.
    - fit: Trains the model for a specified number of epochs, evaluating on a validation set after each epoch, and logs the metrics.

    Returns:
    - A dictionary containing the training history, run configuration, fit configuration, and device information.
    """

    def __init__(
        self,
        model,
        criterion,
        optimizer,
        device: str,
        logger: WandbLogger | None = None,
        num_classes: int = 10,
    ) -> None:
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.logger = logger
        self.num_classes = num_classes

    def train_one_epoch(
        self,
        loader,
        batches_per_epoch: int | None = None,
        single_batch_mode: bool = False,
        single_batch_repeats: int = 1,
    ) -> dict[str, float]:
        return self._epoch_pass(
            loader=loader,
            optimizer=self.optimizer,
            batches_per_epoch=batches_per_epoch,
            single_batch_mode=single_batch_mode,
            single_batch_repeats=single_batch_repeats,
        )

    def evaluate(
        self, loader, batches_per_epoch: int | None = None
    ) -> dict[str, float]:
        return self._epoch_pass(
            loader=loader,
            optimizer=None,
            batches_per_epoch=batches_per_epoch,
            single_batch_mode=False,
            single_batch_repeats=1,
        )

    def fit(
        self, train_loader, val_loader, config: FitConfig
    ) -> dict[str, list[float]]:
        history = defaultdict(list)

        for epoch in range(1, config.epochs + 1):
            train_metrics = self.train_one_epoch(
                loader=train_loader,
                batches_per_epoch=config.train_batches_per_epoch,
                single_batch_mode=config.single_batch_mode,
                single_batch_repeats=config.single_batch_repeats,
            )
            val_metrics = self.evaluate(
                loader=val_loader,
                batches_per_epoch=config.val_batches_per_epoch,
            )

            for key, value in train_metrics.items():
                history[f"train_{key}"].append(value)
            for key, value in val_metrics.items():
                history[f"val_{key}"].append(value)

            if self.logger is not None:
                self.logger.log_metrics(
                    {
                        "epoch": epoch,
                        **{
                            f"train/{key}": value
                            for key, value in train_metrics.items()
                        },
                        **{f"val/{key}": value for key, value in val_metrics.items()},
                    },
                    step=epoch,
                )

        return dict(history)

    def _epoch_pass(
        self,
        loader,
        optimizer,
        batches_per_epoch: int | None,
        single_batch_mode: bool,
        single_batch_repeats: int,
    ) -> dict[str, float]:
        is_train = optimizer is not None
        if is_train:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        total_samples = 0
        confusion = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

        if single_batch_mode:
            images, labels = next(iter(loader))
            batches = [(images, labels)] * max(1, single_batch_repeats)
        else:
            batches = loader

        max_batches = (
            batches_per_epoch if batches_per_epoch is not None else float("inf")
        )
        batch_count = 0

        context = torch.enable_grad() if is_train else torch.no_grad()
        with context:
            for images, labels in batches:
                if batch_count >= max_batches:
                    break

                images = images.to(self.device)
                labels = labels.to(self.device)

                if is_train:
                    optimizer.zero_grad(set_to_none=True)

                logits = self.model(images)
                loss = self.criterion(logits, labels)

                if is_train:
                    loss.backward()
                    optimizer.step()

                batch_size = labels.size(0)
                total_loss += float(loss.item()) * batch_size
                total_samples += batch_size

                preds = torch.argmax(logits, dim=1)
                labels_np = labels.detach().cpu().numpy()
                preds_np = preds.detach().cpu().numpy()
                for y_true, y_pred in zip(labels_np, preds_np):
                    confusion[y_true, y_pred] += 1

                batch_count += 1

        avg_loss = total_loss / max(total_samples, 1)
        metrics = metrics_from_confusion_matrix(confusion)
        metrics["loss"] = avg_loss
        return metrics


def run_model(
    model_factory: Callable[[int], nn.Module],
    model_name: str,
    config: RunConfig,
    fit_config: FitConfig | None = None,
    train_loader: DataLoader | None = None,
    val_loader: DataLoader | None = None,
    optimizer_factory: Callable[[Any], optim.Optimizer] | None = None,
    extra_wandb_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if train_loader is None:
        raise ValueError("train_loader must be provided.")

    if val_loader is None:
        train_loader, val_loader = _split_train_loader(
            train_loader=train_loader,
            val_fraction=config.val_fraction,
            seed=config.random_seed,
        )

    device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model_factory(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    if optimizer_factory is None:
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=0.0,
            weight_decay=0.0,
        )
    else:
        optimizer = optimizer_factory(model.parameters())

    active_fit_config = fit_config or FitConfig(epochs=config.epochs)
    logger_config = {
        **asdict(config),
        "optimizer": optimizer.__class__.__name__,
        "model": model_name,
    }
    if isinstance(optimizer, optim.SGD):
        logger_config.update(
            {
                "momentum": optimizer.defaults.get("momentum", 0.0),
                "weight_decay": optimizer.defaults.get("weight_decay", 0.0),
            }
        )
    if isinstance(optimizer, optim.Adam):
        logger_config.update(
            {
                "betas": list(optimizer.defaults.get("betas", (0.9, 0.999))),
                "eps": float(optimizer.defaults.get("eps", 1e-8)),
                "weight_decay": optimizer.defaults.get("weight_decay", 0.0),
            }
        )
    if extra_wandb_config:
        logger_config.update(extra_wandb_config)

    logger = WandbLogger(
        project=config.wandb_project,
        entity=config.wandb_entity,
        run_name=config.run_name,
        config=logger_config,
        enabled=config.use_wandb,
        tags=config.tags,
    )
    try:
        logger.watch_model(model)

        trainer = Trainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            logger=logger,
            num_classes=10,
        )
        history = trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            config=active_fit_config,
        )
    finally:
        logger.finish()

    return {
        "model": model,
        "model_name": model_name,
        "history": history,
        "run_config": config,
        "fit_config": active_fit_config,
        "device": device,
        "train_loader": train_loader,
        "val_loader": val_loader,
    }


def run_baseline(
    config: RunConfig,
    fit_config: FitConfig | None = None,
    train_loader: DataLoader | None = None,
    val_loader: DataLoader | None = None,
) -> dict[str, Any]:
    return run_model(
        model_factory=BaselineCNN,
        model_name="BaselineCNN",
        config=config,
        fit_config=fit_config,
        train_loader=train_loader,
        val_loader=val_loader,
    )
