import argparse
import os
import random
import time

import numpy as np
import open3d as o3d

DATA_FILE = "data_3d.npz"
OUTPUT_FILE = "result_fgr.npz"
DEFAULT_SEED = 20260716
MAX_O3D_SEED = 2**31 - 1


def npz_string(data, key, default=""):
    if key not in data.files:
        return default
    value = np.asarray(data[key])
    try:
        if value.shape == ():
            return str(value.item())
        return str(value.squeeze().item())
    except Exception:
        return str(value)


def parse_args():
    parser = argparse.ArgumentParser(description="Run Fast Global Registration on one step1 pair.")
    parser.add_argument("--data-file", default=DATA_FILE, help="Input step1 .npz file.")
    parser.add_argument("--output-file", default=OUTPUT_FILE, help="Output result .npz file.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
    parser.add_argument("--voxel-size", type=float, default=0.05, help="Voxel size for downsampling.")
    parser.add_argument("--normal-radius", type=float, default=0.10, help="Radius for normal estimation.")
    parser.add_argument("--fpfh-radius", type=float, default=0.20, help="Radius for FPFH feature extraction.")
    parser.add_argument(
        "--max-correspondence-distance",
        type=float,
        default=0.05,
        help="Maximum correspondence distance for FGR.",
    )
    return parser.parse_args()


def set_seed(seed: int):
    seed = int(seed) % MAX_O3D_SEED
    random.seed(seed)
    np.random.seed(seed)
    o3d.utility.random.seed(seed)
    return seed


def to_pcd(pts: np.ndarray) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(pts, dtype=np.float64))
    return pcd


def preprocess(pcd: o3d.geometry.PointCloud, voxel_size: float, normal_radius: float, fpfh_radius: float):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30)
    )
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=fpfh_radius, max_nn=100),
    )
    return pcd_down, fpfh


def apply_T(X: np.ndarray, T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    return X @ R.T + t


def main():
    args = parse_args()
    if not os.path.exists(args.data_file):
        raise FileNotFoundError(args.data_file)

    seed = set_seed(args.seed)
    print(f"[FGR] seed: {seed}")

    data = np.load(args.data_file, allow_pickle=True)
    X = np.asarray(data["X"], dtype=np.float64)
    Y = np.asarray(data["Y"], dtype=np.float64)
    pair_name = npz_string(data, "pair_name", os.path.basename(args.data_file))

    src = to_pcd(X)
    tgt = to_pcd(Y)

    src_down, src_fpfh = preprocess(src, args.voxel_size, args.normal_radius, args.fpfh_radius)
    tgt_down, tgt_fpfh = preprocess(tgt, args.voxel_size, args.normal_radius, args.fpfh_radius)

    print("[FGR] running ...")
    t0 = time.time()
    option = o3d.pipelines.registration.FastGlobalRegistrationOption(
        maximum_correspondence_distance=args.max_correspondence_distance
    )
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source=src_down,
        target=tgt_down,
        source_feature=src_fpfh,
        target_feature=tgt_fpfh,
        option=option,
    )

    elapsed = time.time() - t0
    T = np.asarray(result.transformation, dtype=np.float64)
    X_warped = apply_T(X, T)

    np.savez(
        args.output_file,
        X_warped=X_warped.astype(np.float64),
        elapsed_time=float(elapsed),
        TM=T.astype(np.float64),
        pair_name=pair_name,
        seed=int(seed),
        voxel_size=float(args.voxel_size),
        normal_radius=float(args.normal_radius),
        fpfh_radius=float(args.fpfh_radius),
        max_correspondence_distance=float(args.max_correspondence_distance),
        source_down_points=int(len(src_down.points)),
        target_down_points=int(len(tgt_down.points)),
    )

    print("[FGR] saved:", args.output_file)
    print("[FGR] time_s:", elapsed)


if __name__ == "__main__":
    main()
