from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.training import BaselineRunConfig, run_baseline, run_manual_grid, run_overfit_check


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run SVHN baseline training outside the notebook and log results to W&B.",
    )
    parser.add_argument("--mode", choices=["baseline", "overfit", "grid"], default="baseline")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--wandb-project", default="DEL")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument("--grid-learning-rates", nargs="+", type=float, default=[1e-2, 5e-3, 1e-3])
    parser.add_argument("--grid-batch-sizes", nargs="+", type=int, default=[32, 64, 128])
    parser.add_argument("--single-batch-repeats", type=int, default=1)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.mode == "grid":
        results = run_manual_grid(
            learning_rates=args.grid_learning_rates,
            batch_sizes=args.grid_batch_sizes,
            epochs=args.epochs,
            data_dir=args.data_dir,
            use_wandb=not args.disable_wandb,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            num_workers=args.num_workers,
        )
        print(f"Finished {len(results)} grid runs.")
        return

    config = BaselineRunConfig(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        data_dir=args.data_dir,
        num_workers=args.num_workers,
        use_wandb=not args.disable_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        run_name=args.run_name,
    )

    if args.mode == "overfit":
        result = run_overfit_check(
            config=config,
            epochs=args.epochs,
            repeats=args.single_batch_repeats,
        )
    else:
        result = run_baseline(config)

    last_val_accuracy = result["history"]["val_accuracy"][-1]
    last_val_macro_f1 = result["history"]["val_macro_f1"][-1]
    print(f"Finished run on {result['device']}.")
    print(f"Final validation accuracy: {last_val_accuracy:.4f}")
    print(f"Final validation macro F1: {last_val_macro_f1:.4f}")


if __name__ == "__main__":
    main()
