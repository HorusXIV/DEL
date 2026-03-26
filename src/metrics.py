from __future__ import annotations

import numpy as np


def confusion_matrix_from_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 10,
) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_label, pred_label in zip(y_true, y_pred):
        matrix[true_label, pred_label] += 1
    return matrix


def metrics_from_confusion_matrix(matrix: np.ndarray) -> dict[str, float]:
    true_positives = np.diag(matrix).astype(np.float64)
    predicted = matrix.sum(axis=0).astype(np.float64)
    actual = matrix.sum(axis=1).astype(np.float64)

    precision_per_class = np.divide(
        true_positives,
        predicted,
        out=np.zeros_like(true_positives),
        where=predicted > 0,
    )
    recall_per_class = np.divide(
        true_positives,
        actual,
        out=np.zeros_like(true_positives),
        where=actual > 0,
    )
    f1_per_class = np.divide(
        2 * precision_per_class * recall_per_class,
        precision_per_class + recall_per_class,
        out=np.zeros_like(precision_per_class),
        where=(precision_per_class + recall_per_class) > 0,
    )

    accuracy = float(true_positives.sum() / matrix.sum()) if matrix.sum() > 0 else 0.0

    return {
        "accuracy": accuracy,
        "macro_precision": float(precision_per_class.mean()),
        "macro_recall": float(recall_per_class.mean()),
        "macro_f1": float(f1_per_class.mean()),
    }
