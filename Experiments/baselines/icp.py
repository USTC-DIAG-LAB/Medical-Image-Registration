import numpy as np
from scipy.spatial import KDTree

def rigid_transform_3d(A, B):
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    cA = np.mean(A, axis=0)
    cB = np.mean(B, axis=0)
    H = (A - cA).T @ (B - cB)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = cB - cA @ R.T
    return R, t


def compose_transform(R, t):
    TM = np.eye(4, dtype=np.float64)
    TM[:3, :3] = R
    TM[:3, 3] = t
    return TM


def robust_icp_3d(X, Y, max_iter=100, outlier_scale=2.0, tolerance=1e-6, return_details=False):
    src0 = np.asarray(X, dtype=np.float64)
    dst = np.asarray(Y, dtype=np.float64)
    src = np.copy(src0)
    tree = KDTree(dst)

    TM_total = np.eye(4, dtype=np.float64)
    history = {"iterations": [], "rmse": [], "inlier_count": []}

    for i in range(int(max_iter)):
        distances, indices = tree.query(src)
        mu_dist = float(np.mean(distances))
        keep_threshold = mu_dist * float(outlier_scale)
        mask = distances < keep_threshold

        A = src[mask]
        B = dst[indices[mask]]
        if len(A) < 3:
            break

        R, t = rigid_transform_3d(A, B)
        src = src @ R.T + t

        TM_step = compose_transform(R, t)
        TM_total = TM_step @ TM_total

        rmse = float(np.sqrt(np.mean(np.sum((B - (A @ R.T + t)) ** 2, axis=1))))
        history["iterations"].append(i + 1)
        history["rmse"].append(rmse)
        history["inlier_count"].append(int(mask.sum()))

        delta_rot = np.linalg.norm(R - np.eye(3))
        delta_trans = np.linalg.norm(t)
        if max(delta_rot, delta_trans) < float(tolerance):
            break

    if return_details:
        return src, TM_total, history
    return src
