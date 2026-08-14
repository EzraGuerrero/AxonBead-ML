"""
Evaluate predicted bead locations against ground-truth annotations.

A predicted point counts as a "hit" if it's close enough to a real
annotated bead — this needs an actual assignment between predictions and
ground truth (not just "is there something nearby"), so two predictions
can't both claim credit for the same true bead. We use the Hungarian
algorithm (optimal assignment) rather than greedy nearest-neighbor matching,
since greedy matching can make an early, suboptimal choice that blocks a
better match later.

This module is written once and reused for every model we evaluate —
the classical baseline today, the U-Net later — so "beat the baseline" is
always measured the same way.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def match_points(
    predicted: np.ndarray, ground_truth: np.ndarray, max_distance: float = 15.0
) -> dict:
    """Match predicted (y, x) points to ground-truth (y, x) points.

    max_distance: pixels. Two points further apart than this are never
    considered a match, regardless of what the optimal assignment would
    otherwise suggest.

    Returns a dict with true_positives, false_positives, false_negatives
    (counts) and precision, recall, f1.
    """
    n_pred, n_true = len(predicted), len(ground_truth)

    if n_pred == 0 and n_true == 0:
        return _metrics(tp=0, fp=0, fn=0)
    if n_pred == 0:
        return _metrics(tp=0, fp=0, fn=n_true)
    if n_true == 0:
        return _metrics(tp=0, fp=n_pred, fn=0)

    distances = cdist(predicted, ground_truth)
    pred_idx, true_idx = linear_sum_assignment(distances)

    valid = distances[pred_idx, true_idx] <= max_distance
    true_positives = int(valid.sum())
    false_positives = n_pred - true_positives
    false_negatives = n_true - true_positives

    return _metrics(tp=true_positives, fp=false_positives, fn=false_negatives)


def _metrics(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }