"""Statistical evaluation and XAI metric helpers.
Implements AUDC, Spearman rank correlations, Jaccard similarities, and classification metrics.
"""

from typing import Dict, List, Set, Union
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    f1_score,
)


def compute_audc(deletion_steps: List[Union[int, float]], metric_values: List[float]) -> float:
    """Compute the Area Under the Deletion Curve (AUDC) using trapezoidal rule.
    Normalized by the span of deletion steps to yield a comparable area score.
    """
    x = np.array(deletion_steps, dtype=float)
    y = np.array(metric_values, dtype=float)

    if len(x) < 2:
        return 0.0

    # Normalize x to [0, 1] range if not already
    x_norm = (x - x[0]) / (x[-1] - x[0]) if x[-1] > x[0] else x
    trapz_func = getattr(np, "trapezoid", getattr(np, "trapz", None))
    area = trapz_func(y, x_norm)
    return float(area)


def compute_spearman_rank_correlation(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate Spearman rank correlation between two attribution vectors."""
    v1 = np.asarray(vec1).flatten()
    v2 = np.asarray(vec2).flatten()
    corr, _ = spearmanr(v1, v2)
    if np.isnan(corr):
        return 0.0
    return float(corr)


def compute_jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    """Calculate Jaccard similarity index between two feature sets."""
    s_a = set(set_a)
    s_b = set(set_b)
    intersection = len(s_a.intersection(s_b))
    union = len(s_a.union(s_b))
    if union == 0:
        return 1.0
    return float(intersection / union)


def compute_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray
) -> Dict[str, float]:
    """Compute standard classification metrics: Accuracy, Precision, Recall, Macro F1, AUC-ROC, Log-Loss."""
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "Macro F1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "AUC-ROC": float(roc_auc_score(y_true, y_prob)),
        "Log-Loss": float(log_loss(y_true, y_prob)),
    }
