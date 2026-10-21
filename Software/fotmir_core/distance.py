"""Squared distances for column-oriented point clouds (column-oriented convention)."""
import numpy as np


def distance(X, Y=None):
    X = np.asarray(X, dtype=np.float64)
    Y = X if Y is None else np.asarray(Y, dtype=np.float64)
    if X.ndim != 2 or Y.ndim != 2 or X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must be d-by-N arrays with matching dimensions.")
    # Preserve the original expression's grouping; clipping belongs in Sinkhorn.
    return np.sum(X * X, axis=0)[:, None] + (
        np.sum(Y * Y, axis=0)[None, :] - 2 * X.T @ Y
    )
