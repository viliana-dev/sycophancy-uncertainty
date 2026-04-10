"""Evaluation metrics: AUROC, g-mean^2, accuracy@FPR, bootstrap SE.

Ported from scheming-probes/src/evaluate.py — core metrics only.
"""

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def compute_auroc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Compute AUROC. Returns 0.5 if only one class present."""
    if len(np.unique(y_true)) < 2:
        return 0.5
    return float(roc_auc_score(y_true, y_scores))


def compute_gmean2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute g-mean^2 = TPR * TNR."""
    tpr = float(np.mean(y_pred[y_true == 1] == 1)) if (y_true == 1).sum() > 0 else 0.0
    tnr = float(np.mean(y_pred[y_true == 0] == 0)) if (y_true == 0).sum() > 0 else 0.0
    return tpr * tnr


def compute_accuracy_at_fpr(
    y_true: np.ndarray, y_scores: np.ndarray, target_fpr: float
) -> float:
    """Compute accuracy at a fixed false positive rate."""
    if len(np.unique(y_true)) < 2:
        return 0.5
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    idx = np.argmin(np.abs(fpr - target_fpr))
    threshold = thresholds[min(idx, len(thresholds) - 1)]
    y_pred = (y_scores >= threshold).astype(int)
    return float(np.mean(y_pred == y_true))


def bootstrap_auroc_se(
    y_true: np.ndarray, y_scores: np.ndarray, n_boot: int = 1000
) -> float:
    """Bootstrap standard error of AUROC."""
    n = len(y_true)
    aucs = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.random.randint(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            aucs[b] = np.nan
            continue
        aucs[b] = roc_auc_score(y_true[idx], y_scores[idx])
    return float(np.nanstd(aucs))
