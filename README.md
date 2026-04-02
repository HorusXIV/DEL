# DEL - SVHN Digit Classification

This project builds and evaluates a convolutional neural network (CNN) baseline for classifying house numbers from the SVHN dataset.

## Task

The project follows two phases:

1. Baseline phase
- Explore and preprocess SVHN data.
- Implement a stable training and evaluation pipeline.
- Train a baseline model and report metrics.
- Run an overfitting sanity check on a single batch.
- Estimate metric uncertainty (for example with cross-validation).

2. Hypothesis-driven tuning phase
- Formulate hypotheses about hyperparameters and architecture choices.
- Run controlled experiments.
- Analyze whether results support each hypothesis.

Primary dataset:
- SVHN (Street View House Numbers): http://ufldl.stanford.edu/housenumbers/

## Baseline Model Architecture

Input: 32x32x3 image

Feature extractor:
- Conv2d(3 -> 16, kernel=5, stride=1, padding=2) + ReLU
- Conv2d(16 -> 32, kernel=5, stride=1, padding=2) + ReLU
- MaxPool2d(kernel=2)
- Conv2d(32 -> 128, kernel=5, stride=1, padding=2) + ReLU
- MaxPool2d(kernel=2)

Classifier:
- Flatten
- Linear(128*8*8 -> 512) + ReLU
- Linear(512 -> 96) + ReLU
- Linear(96 -> 10)

Training setup used for the baseline:
- Loss: CrossEntropyLoss
- Optimizer: SGD (momentum=0.0, weight_decay=0.0)
- Validation split from training set
- Logging via Weights & Biases (optional)

## Used Technologies

- Python 3.12+
- PyTorch and TorchVision
- NumPy, SciPy, Pandas
- scikit-learn
- Matplotlib, Seaborn
- Weights & Biases (wandb)
- Jupyter / Notebook tooling
- uv for dependency and environment management

## Quick Start

1. Install dependencies

```bash
uv sync
```

2. Run baseline training

```bash
uv run python scripts/run_baseline.py --mode baseline --epochs 5 --learning-rate 1e-2 --batch-size 64
```

3. Run single-batch overfit check

```bash
uv run python scripts/run_baseline.py --mode overfit --epochs 20 --single-batch-repeats 1
```

4. Run manual learning-rate and batch-size grid

```bash
uv run python scripts/run_baseline.py --mode grid --epochs 5
```

If you want to disable W&B logging, add:

```bash
--disable-wandb
```

## Project Structure

```text
.
|-- data/
|   |-- train_32x32.mat
|   `-- test_32x32.mat
|-- notebooks/
|   |-- del_report.ipynb
|   `-- wandb/
|-- scripts/
|   `-- run_baseline.py
|-- src/
|   |-- __init__.py
|   |-- cross_validation.py
|   |-- metrics.py
|   |-- model.py
|   |-- training.py
|   `-- wandb_tools.py
|-- wandb/
|-- pyproject.toml
`-- README.md
```

## References

- SVHN dataset: http://ufldl.stanford.edu/housenumbers/
- Inspiration: Handwritten Digit Recognition with a Back-Propagation Network (NeurIPS 1989)
  https://proceedings.neurips.cc/paper/1989/file/53c3bce66e43be4f209556518c2fcb54-Paper.pdf