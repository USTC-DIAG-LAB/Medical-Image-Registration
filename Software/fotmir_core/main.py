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
            "The transport flow must retain the original coreset dimensions."
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
        if cloud.shape[1] < 3 or np.linalg.matrix_rank(cloud - cloud.mean(axis=1, keepdims=True)) < 2:
            raise ValueError(f"{name} needs at least three non-collinear points to determine a 3D rigid transform.")
    if not np.isfinite(tau) or not 0 < tau < 1:
        raise ValueError("Main registration requires 0 < tau < 1.")
    if not np.isfinite(ratio) or not 0 < ratio <= 1:
        raise ValueError("ratio must be in (0, 1].")
    if seed is None:
        seed = secrets.randbelow(2**31 - 1) + 1
    if isinstance(seed, (bool, np.bool_)) or not np.isfinite(seed) or int(seed) != seed or not 0 <= seed <= 2**31 - 1:
        raise ValueError("seed must be an integer in [0, 2**31 - 1].")
    seed = int(seed)
    NA, NB = A.shape[1], B.shape[1]
    WA, WB = np.ones(NA) / NA, np.ones(NB) / NB
    # Rounding uses half-away-from-zero, unlike Python round/np.rint.
    kA = min(NA, max(50, int(np.floor(NA * ratio + 0.5))))
    kB = min(NB, max(50, int(np.floor(NB * ratio + 0.5))))
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
        "implementation": "FOTMIR Python application core streaming",
        "version": PACKAGE_VERSION,
        "coreset_support_policy": "positive_mass_full_flow",
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
    if seed is not None and (isinstance(seed, (bool, np.bool_)) or not np.isfinite(seed)
                             or int(seed) != seed or not 0 <= seed <= 2**31 - 1):
        raise ValueError("seed must be an integer in [0, 2**31 - 1].")
    if threads is not None:
        if isinstance(threads, bool) or int(threads) != threads or threads < 1:
            raise ValueError("threads must be a positive integer or None.")
        import os, subprocess, sys, tempfile, pickle
        env = os.environ.copy()
        for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"]:
            env[key] = str(int(threads))
        env["FOTMIR_CORE_THREADS"] = str(int(threads))
        with tempfile.TemporaryDirectory(prefix="fotmir-core-") as temp:
            request = Path(temp)/"request.json"; response = Path(temp)/"response.pkl"
            request.write_text(json.dumps(dict(input=str(Path(in_path).resolve()), output=str(Path(out_path).resolve()),
                tau=tau, ratio=ratio, seed=None if seed is None else int(seed),
                initial_indices=None if initial_indices is None else np.asarray(initial_indices).tolist())))
            proc = subprocess.run([sys.executable, str(Path(__file__).with_name("thread_worker.py")), str(request), str(response)], env=env, capture_output=True, text=True)
            if proc.returncode: raise RuntimeError("Registration worker failed: " + proc.stderr)
            # Read only the private response created by our own worker.
            with response.open("rb") as f: return pickle.load(f)
    with np.load(in_path, allow_pickle=False) as archive:
        source = {key: archive[key] for key in archive.files}
    if "A" not in source or "B" not in source:
        raise ValueError("Input NPZ file must contain A and B as 3-by-N point clouds.")
    result = register_points(source["A"], source["B"], tau=tau, ratio=ratio,
                             seed=seed, initial_indices=initial_indices)
    import os
    result["threads_requested"] = int(os.environ.get("FOTMIR_CORE_THREADS", "-1"))
    result["thread_control"] = "startup_environment" if "FOTMIR_CORE_THREADS" in os.environ else "inherited_process"
    result["thread_environment_json"] = json.dumps({k: os.environ.get(k) for k in
        ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"]})
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
