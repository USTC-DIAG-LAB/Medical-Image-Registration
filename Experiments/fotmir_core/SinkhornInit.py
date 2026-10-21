"""Alternating transport/orthogonal updates, preserving the source loop order."""
import numpy as np

if __package__:
    from .Sinkhorn import Sinkhorn
else:
    from Sinkhorn import Sinkhorn


def SinkhornInit(A, B, WA, WB, TM, tau):
    A, B = np.asarray(A, dtype=np.float64), np.asarray(B, dtype=np.float64)
    WA, WB = np.asarray(WA, dtype=np.float64).reshape(-1), np.asarray(WB, dtype=np.float64).reshape(-1)
    # This unusual sum-of-coordinates test is deliberately retained for reproducibility.
    keep_a, keep_b = np.sum(A, axis=0) != 0, np.sum(B, axis=0) != 0
    A, B, WA, WB = A[:, keep_a], B[:, keep_b], WA[keep_a], WB[keep_b]
    if A.shape[1] == 0 or B.shape[1] == 0 or WA.sum() <= 0 or WB.sum() <= 0:
        raise ValueError("No positive-mass points remain after the coordinate-sum filter.")
    WA, WB = WA / WA.sum(), WB / WB.sum()
    U, _, Vh = np.linalg.svd(np.asarray(TM, dtype=np.float64))
    TM = U @ Vh
    transformed = TM @ A
    previous_cost = -1.0
    updates = 0
    EMD, FM = Sinkhorn(transformed, B, WA, WB, tau)
    while True:
        EMD, FM = Sinkhorn(transformed, B, WA, WB, tau)
        if abs(previous_cost - EMD) <= 1e-6 * EMD:
            break
        previous_cost = EMD
        U, _, Vh = np.linalg.svd(B @ FM.T @ A.T)
        TM = U @ Vh
        transformed = TM @ A
        updates += 1
        # Preserve >50 rather than silently changing it to >=50.
        if updates > 50:
            break
    return EMD, FM, TM
