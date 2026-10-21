"""Standalone NumPy/SciPy registration backend."""
import argparse
import json
from pathlib import Path
import secrets
import time

PACKAGE_VERSION = (Path(__file__).resolve().parents[2] / "VERSION").read_text(encoding="utf-8").strip()

import numpy as np
import scipy
from scipy.spatial import cKDTree

if __package__:
    from .KCenter import KCenter
    from .SinkhornInit import SinkhornInit
else:
    from KCenter import KCenter
    from SinkhornInit import SinkhornInit


def project_to_rotation(R):
    U, _, Vh = np.linalg.svd(R)
    R = U @ Vh
    if np.linalg.det(R) < 0:
        Vh[-1] *= -1
        R = U @ Vh
    return R


def run_one_coreset(A, B, WA, WB, kA, kB, tau, *, initial_indices):
    CSA, CSWA = KCenter(A, WA, kA, initial_index=initial_indices[0])
    CSB, CSWB = KCenter(B, WB, kB, initial_index=initial_indices[1])
    centered_a = CSA - (CSA @ CSWA)[:, None]
    centered_b = CSB - (CSB @ CSWB)[:, None]
    EMD, FM, R = SinkhornInit(centered_a, centered_b, CSWA, CSWB, np.eye(3), tau)
    if FM.shape != (kA, kB):
        raise ValueError(
            "The coordinate-sum filter removed coreset columns, "
            "so the flow no longer matches the original coresets."
        )
    R = project_to_rotation(R)
    row_sum = np.sum(FM, axis=1) + np.finfo(np.float64).eps
    matched_b = (CSB @ FM.T) / row_sum
    weights = row_sum / row_sum.sum()
    t = np.sum((matched_b - R @ CSA) * weights, axis=1, keepdims=True)
    return R, t, EMD, FM, CSA, CSB


def register_points(A, B, *, tau=0.05, ratio=0.05, seed=None, initial_indices=None):
    """Register 3-by-N A to 3-by-M B; return the registration result fields.

    initial_indices is a pair of ONE-based starting point indices, intended for
    deterministic replay. Timing excludes input loading and output saving.
    """
    A, B = np.asarray(A, dtype=np.float64), np.asarray(B, dtype=np.float64)
    for name, cloud in (("A", A), ("B", B)):
        if cloud.ndim != 2 or cloud.shape[0] != 3 or cloud.shape[1] == 0:
            raise ValueError(f"{name} must be a nonempty 3-by-N point cloud; got {cloud.shape}.")
        if not np.all(np.isfinite(cloud)):
            raise ValueError(f"{name} contains NaN or infinite coordinates.")
    if not np.isfinite(tau) or not 0 < tau < 1:
        raise ValueError("Main registration requires 0 < tau < 1.")
    if not np.isfinite(ratio) or not 0 < ratio <= 1:
        raise ValueError("ratio must be in (0, 1].")
    if seed is None:
        seed = secrets.randbelow(2**31 - 1) + 1
    if int(seed) != seed or not 0 <= seed <= 2**31 - 1:
        raise ValueError("seed must be an integer in [0, 2**31 - 1].")
    seed = int(seed)
    NA, NB = A.shape[1], B.shape[1]
    WA, WB = np.ones(NA) / NA, np.ones(NB) / NB
    # Rounding uses half-away-from-zero, unlike Python round/np.rint.
    kA = max(50, int(np.floor(NA * ratio + 0.5)))
    kB = max(50, int(np.floor(NB * ratio + 0.5)))
    if initial_indices is not None:
        raw_indices = np.asarray(initial_indices).reshape(-1)
        if raw_indices.size != 2 or not np.all(np.isfinite(raw_indices)) or not np.all(raw_indices == np.floor(raw_indices)):
            raise ValueError("initial_indices must contain two one-based integer indices.")
        if not 1 <= raw_indices[0] <= NA or not 1 <= raw_indices[1] <= NB:
            raise ValueError("initial_indices exceed the point-cloud bounds.")
        initial_indices = raw_indices.astype(np.int64)
    start = time.perf_counter()
    tree = cKDTree(B.T)
    rng = np.random.RandomState(seed)  # MT19937 deterministic sampling.
    run_start = time.perf_counter()
    if initial_indices is None:
        initial_indices = np.array([rng.randint(NA) + 1, rng.randint(NB) + 1])
    R, t, EMD, FM, CSA, CSB = run_one_coreset(
        A, B, WA, WB, kA, kB, tau, initial_indices=initial_indices
    )
    run_elapsed = time.perf_counter() - run_start
    transformed = R @ A + t
    _, indices = tree.query(transformed.T, k=1)
    residual = B[:, indices] - transformed
    score = float(np.sqrt(np.mean(np.sum(residual**2, axis=0))))
    elapsed = time.perf_counter() - start
    TM = np.eye(4)
    TM[:3, :3], TM[:3, 3:4] = R, t
    return {
        "TM": TM, "R": R, "t": t, "CSA": CSA, "CSB": CSB, "FM": FM,
        "elapsed_time": elapsed, "ratio": ratio, "EMD": EMD, "tau": tau,
        "n_repeats": 1, "seeds": seed, "best_seed": seed, "best_run": 1,
        "best_score_before_refine": score, "final_score_after_refine": score,
        "run_scores": score, "run_elapsed": run_elapsed, "run_EMD": EMD,
        "refine_iter": 0, "refine_rmse": np.nan,
        "selection_metric": "single_pass_full_point_nearest_neighbor_rmse",
        "refine_metric": "not_used", "score_sample_n": NA,
        "score_idx": np.empty((0, 0)), "trim_keep_ratio": np.nan,
        "coreset_method": "KCenterPlus",
        # Extra fields do not change the existing evaluator's required schema.
        "initial_indices": initial_indices.reshape(1, 2),
        "initial_indices_base": 1,
        "implementation": "FOTMIR Python benchmark core streaming",
        "version": PACKAGE_VERSION,
        "coreset_distance_mode": "streaming",
        "numpy_version": np.__version__, "scipy_version": scipy.__version__,
        "rng_algorithm": "NumPy RandomState MT19937",
    }


def main(in_path, out_path, mode="auto", *parameters, seed=None, initial_indices=None, threads=1):
    """NPZ API: main(input, output, mode, tau, ratio).

    Input archives contain A and B, each a 3-by-N point array.
    """
    mode = "auto" if not mode else mode.lower()
    if mode == "auto":
        if parameters:
            raise ValueError("Auto mode does not take manual parameters.")
        tau, ratio = 0.05, 0.05
    elif mode == "manual":
        if len(parameters) == 1:
            tau, ratio = 0.05, float(parameters[0])
        elif len(parameters) == 2:
            tau, ratio = map(float, parameters)
        else:
            raise ValueError("Manual mode requires ratio, or tau and ratio.")
    else:
        raise ValueError("mode must be 'auto' or 'manual'.")
    with np.load(in_path, allow_pickle=False) as archive:
        source = {key: archive[key] for key in archive.files}
    if "A" not in source or "B" not in source:
        raise ValueError("Input NPZ file must contain A and B as 3-by-N point clouds.")
    from threadpoolctl import threadpool_limits, threadpool_info
    if threads is not None and (int(threads) != threads or threads < 1):
        raise ValueError("threads must be a positive integer or None.")
    # None respects the embedding application's policy; CLI defaults to 1.
    with threadpool_limits(limits=threads):
        pools = threadpool_info()
        result = register_points(source["A"], source["B"], tau=tau, ratio=ratio,
                                 seed=seed, initial_indices=initial_indices)
        result["threadpool_info_json"] = json.dumps(pools)
        result["threads_requested"] = -1 if threads is None else int(threads)
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as stream:
        np.savez(stream, **result)
    return result


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version", version="FOTMIR " + PACKAGE_VERSION)
    parser.add_argument("input", help="NPZ input containing A and B (3-by-N).")
    parser.add_argument("output", help="Output NPZ archive.")
    parser.add_argument("mode", nargs="?", default="auto", choices=("auto", "manual"))
    parser.add_argument("parameters", nargs="*", type=float, help="Manual: ratio, or tau ratio.")
    parser.add_argument("--seed", type=int, help="Reproducible Python RNG seed.")
    parser.add_argument("--threads", type=int, default=1, help="BLAS/OpenMP threads (default: 1).")
    parser.add_argument("--initial-indices", nargs=2, type=int, metavar=("A_INDEX", "B_INDEX"),
                        help="One-based first centers; use these for deterministic replay.")
    args = parser.parse_args()
    result = main(args.input, args.output, args.mode, *args.parameters,
                  seed=args.seed, initial_indices=args.initial_indices, threads=args.threads)
    print(f"Saved: {args.output}\nNN RMSE: {result['run_scores']:.8g}\n"
          f"Core elapsed: {result['elapsed_time']:.6f} s\n"
          f"Initial indices (1-based): {result['initial_indices'].ravel().tolist()}")


if __name__ == "__main__":
    cli()
