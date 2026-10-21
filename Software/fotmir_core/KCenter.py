"""Farthest-point partition followed by weighted centroids."""
import numpy as np

if __package__:
    from .distance import distance
else:
    from distance import distance


def KCenter(A, WA, num, *, rng=None, initial_index=None, distance_mode="streaming"):
    """Return (centroids, weights); initial_index is optional and ONE-based.

    Compute only selected distance rows, retaining first-occurrence tie rules.
    distance_mode="dense" retains the original implementation for verification.
    Empty clusters produce a zero centroid and zero weight, with mass preserved.
    """
    A = np.asarray(A, dtype=np.float64)
    WA = np.asarray(WA, dtype=np.float64).reshape(-1)
    if A.ndim != 2 or A.shape[1] == 0 or WA.size != A.shape[1]:
        raise ValueError("A must be d-by-N and WA must contain N weights.")
    if not np.all(np.isfinite(A)) or not np.all(np.isfinite(WA)) or np.any(WA < 0):
        raise ValueError("Points/weights must be finite and weights nonnegative.")
    if WA.sum() <= 0 or int(num) != num or num < 1:
        raise ValueError("Positive total weight and a positive integer num are required.")
    num = int(num)
    rng = np.random.RandomState() if rng is None else rng
    if initial_index is None:
        initial_index = int(rng.randint(A.shape[1])) + 1
    if int(initial_index) != initial_index or not 1 <= initial_index <= A.shape[1]:
        raise ValueError("initial_index must be a one-based point index.")
    if distance_mode not in ("streaming", "dense"):
        raise ValueError("distance_mode must be 'streaming' or 'dense'.")
    labels = np.zeros(A.shape[1], dtype=np.int64)
    if distance_mode == "dense":
        all_distances = distance(A, A)
        def distance_row(index):
            return all_distances[index]
    else:
        norms = np.sum(A * A, axis=0)
        def distance_row(index):
            # Same expression grouping; no clipping or changed tie rule.
            return norms[index] + (norms - 2 * A[:, index] @ A)
    nearest = distance_row(int(initial_index) - 1).copy()
    for i in range(1, num):
        current = int(np.argmax(nearest))
        candidate = distance_row(current)
        # The minimum update keeps the old cluster when distances are tied.
        changed = candidate < nearest
        nearest = np.minimum(nearest, candidate)
        labels[changed] = i
    centroids = np.zeros((A.shape[0], num), dtype=np.float64)
    weights = np.zeros(num, dtype=np.float64)
    for i in range(num):
        selected = labels == i
        weights[i] = np.sum(WA[selected])
        if weights[i] > 0:
            centroids[:, i] = np.sum(A[:, selected] * (WA[selected] / weights[i]), axis=1)
        elif np.any(selected):
            raise ValueError("A nonempty cluster has zero total weight.")
    return centroids, weights
