"""Alternating transport/orthogonal updates, preserving the source loop order."""
import numpy as np

if __package__:
    from .Sinkhorn import Sinkhorn
else:
    from Sinkhorn import Sinkhorn


def SinkhornInit(A, B, WA, WB, TM, tau):
    A, B = np.asarray(A, dtype=np.float64), np.asarray(B, dtype=np.float64)
    WA, WB = np.asarray(WA, dtype=np.float64).reshape(-1), np.asarray(WB, dtype=np.float64).reshape(-1)
    if (A.ndim != 2 or B.ndim != 2 or A.shape[0] != B.shape[0]
            or A.shape[1] != WA.size or B.shape[1] != WB.size):
        raise ValueError("Point and mass dimensions must agree.")
    if any(not np.isfinite(v).all() for v in (A, B, WA, WB)) or np.any(WA < 0) or np.any(WB < 0):
        raise ValueError("Points and masses must be finite; masses must be nonnegative.")
    original_shape = (WA.size, WB.size)
    # A zero coordinate sum is a valid geometric point, not an empty cluster.
    # Empty coreset clusters are identified by their mass alone.
    keep_a, keep_b = WA > 0, WB > 0
    A, B, WA, WB = A[:, keep_a], B[:, keep_b], WA[keep_a], WB[keep_b]
    if A.shape[1] == 0 or B.shape[1] == 0 or WA.sum() <= 0 or WB.sum() <= 0:
        raise ValueError("No positive-mass points remain.")
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
    if FM.shape != original_shape:
        full_flow = np.zeros(original_shape, dtype=np.float64)
        full_flow[np.ix_(keep_a, keep_b)] = FM
        FM = full_flow
    return EMD, FM, TM
