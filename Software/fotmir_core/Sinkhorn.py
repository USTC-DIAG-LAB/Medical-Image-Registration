"""Partial-mass cost construction and transport."""
import numpy as np

if __package__:
    from .distance import distance
    from .Transport import Transport
else:
    from distance import distance
    from Transport import Transport


def Sinkhorn(TM_A, B, WA, WB, tau):
    if not np.isfinite(tau) or not 0 <= tau < 1:
        raise ValueError("tau must be in [0, 1); tau=1 makes the source EMD formula singular.")
    WA = np.asarray(WA, dtype=np.float64).reshape(-1)
    WB = np.asarray(WB, dtype=np.float64).reshape(-1)
    costs = np.maximum(distance(TM_A, B), 0)
    if costs.shape != (WA.size, WB.size):
        raise ValueError("Weights and point counts do not agree.")
    if tau > 0:
        dummy_cost = np.max(costs) * 10
        WA = np.append(WA, tau * WA.sum())
        WB = np.append(WB, tau * WB.sum())
        costs = np.pad(costs, ((0, 1), (0, 1)))
        costs[-1, -1] = dummy_cost
    median = float(np.median(costs))
    if not np.isfinite(median) or median <= 0:
        raise ValueError("The augmented cost median must be positive; degenerate points cannot use lambda=10/median.")
    lambda_ = 10.0 / median
    kernel = np.exp(-lambda_ * costs)
    D, _, u, v = Transport(WA, WB, kernel, kernel * costs, lambda_)
    flow = u * kernel * v.T
    if tau > 0:
        flow = flow[:-1, :-1]
    return float(D[0] / (1 - tau)), flow
