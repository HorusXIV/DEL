from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from collections.abc import Mapping

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import wandb

MPL_CONFIG_DIR = Path("/tmp/matplotlib")
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

class WandbLogger:
    def __init__(
        self,
        project: str,
        entity: str | None = None,
        run_name: str | None = None,
        config: Mapping[str, Any] | None = None,
        enabled: bool = True,
        tags: list[str] | None = None,
        settings: wandb.Settings | None = None,
    ) -> None:
        self.enabled = enabled
        self.run = None

        if self.enabled:
            self.run = wandb.init(
                project=project,
                entity=entity,
                name=run_name,
                config=dict(config) if config else None,
                tags=tags,
                settings=settings,
            )

    def log_metrics(self, metrics: Mapping[str, Any], step: int | None = None) -> None:
        if not self.enabled or self.run is None:
            return

        payload = dict(metrics)
        if step is None:
            self.run.log(payload)
        else:
            self.run.log(payload, step=step)

    def watch_model(self, model, log: str = "all", log_freq: int = 100) -> None:
        if self.enabled and self.run is not None:
            self.run.watch(model, log=log, log_freq=log_freq)

    def finish(self) -> None:
        if self.enabled and self.run is not None:
            self.run.finish()


class WandbRuns:
    def __init__(self, project: str, entity: str | None = None) -> None:
        self.project = project
        self.entity = entity
        self.api = wandb.Api()

    def _project_path(self) -> str:
        if self.entity:
            return f"{self.entity}/{self.project}"
        return self.project

    def list_runs(
        self, filters: Mapping[str, Any] | None = None, order: str = "-created_at"
    ):
        return self.api.runs(
            path=self._project_path(),
            filters=dict(filters) if filters else None,
            order=order,
        )

    def runs_overview(
        self, filters: Mapping[str, Any] | None = None, top_k: int | None = 20
    ) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        runs = self.list_runs(filters=filters)

        for idx, run in enumerate(runs):
            if top_k is not None and idx >= top_k:
                break
            rows.append(
                {
                    "id": run.id,
                    "name": run.name,
                    "state": run.state,
                    "created_at": run.created_at,
                    "val_accuracy": run.summary.get("val/accuracy"),
                    "val_macro_f1": run.summary.get("val/macro_f1"),
                    "val_loss": run.summary.get("val/loss"),
                }
            )

        return pd.DataFrame(rows)

    def history(
        self, run_path: str, keys: list[str], samples: int = 1000
    ) -> pd.DataFrame:
        run = self.api.run(run_path)
        history_df = run.history(keys=keys, samples=samples, pandas=True)
        return history_df.dropna(how="all")

    def compare_metric(
        self,
        run_paths: list[str],
        metric_key: str,
        step_key: str = "_step",
        smooth: int = 1,
        samples: int = 1000,
        title: str | None = None,
    ) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []

        for run_path in run_paths:
            df = self.history(
                run_path=run_path, keys=[step_key, metric_key], samples=samples
            )
            if step_key not in df.columns or metric_key not in df.columns:
                continue

            curve = df[[step_key, metric_key]].copy()
            curve["run"] = run_path.split("/")[-1]
            if smooth > 1:
                curve[metric_key] = (
                    curve[metric_key].rolling(window=smooth, min_periods=1).mean()
                )
            frames.append(curve)

        if not frames:
            return pd.DataFrame(columns=[step_key, metric_key, "run"])

        plot_df = pd.concat(frames, ignore_index=True)
        plt.figure(figsize=(10, 5))
        sns.lineplot(data=plot_df, x=step_key, y=metric_key, hue="run")
        plt.title(title or f"W&B comparison: {metric_key}")
        plt.xlabel(step_key)
        plt.ylabel(metric_key)
        plt.tight_layout()
        plt.show()
        return plot_df
