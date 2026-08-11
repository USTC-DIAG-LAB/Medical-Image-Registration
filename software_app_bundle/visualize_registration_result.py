import argparse
import json
import math
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import SimpleITK as sitk
from scipy.spatial import cKDTree


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize 3D registration result from one case.")
    parser.add_argument("--pair-npz", required=True, help="step1 output .npz for one pair")
    parser.add_argument("--result-file", required=True, help="registration result .mat or .npz containing TM")
    parser.add_argument("--output-dir", default="visualization_output", help="folder to save visualization")
    parser.add_argument(
        "--slice-mode",
        choices=("foreground_center", "image_center"),
        default="foreground_center",
        help="how to choose the representative slice center",
    )
    parser.add_argument(
        "--save-fixed",
        action="store_true",
        help="also save the fixed image slices as standalone figures",
    )
    parser.add_argument(
        "--direction-mode",
        choices=("auto", "forward", "inverse"),
        default="auto",
        help="which TM direction to use for warping: auto compares point RMSE, forward uses TM, inverse uses inv(TM)",
    )
    parser.add_argument(
        "--crop-pad-ratio",
        type=float,
        default=0.0,
        help="padding ratio for 3D content cropping before slicing",
    )
    parser.add_argument(
        "--crop-min-pad",
        type=int,
        default=0,
        help="minimum voxel padding for 3D content cropping before slicing",
    )
    parser.add_argument(
        "--canvas-pad-mm",
        type=float,
        default=40.0,
        help="extra physical padding in mm for the enlarged 3D visualization canvas",
    )
    parser.add_argument(
        "--save-volumes",
        action="store_true",
        help="save processed 3D volumes first, then generate screenshots from the saved volumes",
    )
    parser.add_argument(
        "--slice-output-mode",
        choices=("full", "centered", "both"),
        default="both",
        help="full=fixed-aligned full slices, centered=self-centered full slices, both=save both variants",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def npz_string(data, key: str, default: str = "") -> str:
    if key not in data.files:
        return default
    value = np.asarray(data[key])
    try:
        if value.shape == ():
            return str(value.item())
        return str(value.squeeze().item())
    except Exception:
        return str(value)


def save_image(image: sitk.Image, path: Path) -> None:
    ensure_dir(path.parent)
    sitk.WriteImage(image, str(path), useCompression=True)


def make_ones_like(image: sitk.Image) -> sitk.Image:
    ones = sitk.Image(image.GetSize(), sitk.sitkUInt8)
    ones.CopyInformation(image)
    ones += 1
    return ones


def load_tm_and_runtime(result_path: Path) -> Tuple[np.ndarray, float]:
    if result_path.suffix.lower() == ".npz":
        data = np.load(result_path, allow_pickle=True)
        if "TM" not in data.files:
            raise ValueError(f"TM not found in {result_path}")
        tm = np.asarray(data["TM"], dtype=np.float64)
        elapsed_time = float(data["elapsed_time"]) if "elapsed_time" in data.files else math.nan
        return tm, elapsed_time

    if result_path.suffix.lower() == ".mat":
        data = sio.loadmat(result_path)
        if "TM" in data:
            tm = np.asarray(data["TM"], dtype=np.float64)
        elif "R" in data and "t" in data:
            tm = np.eye(4, dtype=np.float64)
            tm[:3, :3] = np.asarray(data["R"], dtype=np.float64)
            tm[:3, 3] = np.asarray(data["t"], dtype=np.float64).reshape(3)
        else:
            raise ValueError(f"TM or R/t not found in {result_path}")
        elapsed = data.get("elapsed_time", np.array([[math.nan]]))
        elapsed_time = float(np.asarray(elapsed).reshape(-1)[0])
        return tm, elapsed_time

    raise ValueError(f"Unsupported result file: {result_path}")


def tm_norm_to_physical(tm_norm: np.ndarray, center_y: np.ndarray, scale: float) -> np.ndarray:
    tm_norm = np.asarray(tm_norm, dtype=np.float64)
    center_y = np.asarray(center_y, dtype=np.float64).reshape(3)
    scale = float(scale)

    r = tm_norm[:3, :3]
    t_norm = tm_norm[:3, 3]
    t_phys = scale * t_norm + center_y - r @ center_y

    tm_phys = np.eye(4, dtype=np.float64)
    tm_phys[:3, :3] = r
    tm_phys[:3, 3] = t_phys
    return tm_phys


def apply_tm_points(points: np.ndarray, tm: np.ndarray) -> np.ndarray:
    r = tm[:3, :3]
    t = tm[:3, 3]
    return np.asarray(points, dtype=np.float64) @ r.T + t


def nearest_neighbor_rmse(moving_points: np.ndarray, fixed_points: np.ndarray) -> float:
    tree = cKDTree(np.asarray(fixed_points, dtype=np.float64))
    distances, _ = tree.query(np.asarray(moving_points, dtype=np.float64))
    return float(np.sqrt(np.mean(distances ** 2)))


def resolve_tm_direction(
    pair_data,
    tm_phys: np.ndarray,
    direction_mode: str,
) -> Tuple[np.ndarray, str, float, float]:
    if direction_mode == "forward":
        return tm_phys, "forward", math.nan, math.nan
    if direction_mode == "inverse":
        return np.linalg.inv(tm_phys), "inverse", math.nan, math.nan

    if not {"X", "Y", "center_Y", "scale"}.issubset(set(pair_data.files)):
        return tm_phys, "forward", math.nan, math.nan

    center_y = np.asarray(pair_data["center_Y"], dtype=np.float64).reshape(3)
    scale = float(pair_data["scale"])
    x_phys = np.asarray(pair_data["X"], dtype=np.float64) * scale + center_y
    y_phys = np.asarray(pair_data["Y"], dtype=np.float64) * scale + center_y

    tm_inv = np.linalg.inv(tm_phys)
    rmse_forward = nearest_neighbor_rmse(apply_tm_points(x_phys, tm_phys), y_phys)
    rmse_inverse = nearest_neighbor_rmse(apply_tm_points(x_phys, tm_inv), y_phys)

    if rmse_inverse < rmse_forward:
        return tm_inv, "inverse", rmse_forward, rmse_inverse
    return tm_phys, "forward", rmse_forward, rmse_inverse


def rotation_angle_deg(tm_phys: np.ndarray) -> float:
    r = np.asarray(tm_phys[:3, :3], dtype=np.float64)
    trace_val = np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.rad2deg(np.arccos(trace_val)))


def build_sitk_affine(tm_phys: np.ndarray) -> sitk.AffineTransform:
    transform = sitk.AffineTransform(3)
    transform.SetMatrix(np.asarray(tm_phys[:3, :3], dtype=np.float64).reshape(-1).tolist())
    transform.SetTranslation(np.asarray(tm_phys[:3, 3], dtype=np.float64).tolist())
    return transform


def image_corner_points_physical(image: sitk.Image) -> np.ndarray:
    size_xyz = np.asarray(image.GetSize(), dtype=np.float64)
    corners = []
    # Use voxel outer boundaries rather than voxel centers.
    # This avoids underestimating the saved 3D canvas and clipping valid anatomy.
    for x in (-0.5, max(-0.5, size_xyz[0] - 0.5)):
        for y in (-0.5, max(-0.5, size_xyz[1] - 0.5)):
            for z in (-0.5, max(-0.5, size_xyz[2] - 0.5)):
                corners.append(image.TransformContinuousIndexToPhysicalPoint((x, y, z)))
    return np.asarray(corners, dtype=np.float64)


def infer_foreground_mask_from_image(image: sitk.Image) -> np.ndarray:
    arr = sitk.GetArrayFromImage(image).astype(np.float64)
    valid = np.isfinite(arr)
    if not np.any(valid):
        return np.zeros_like(arr, dtype=bool)

    values = arr[valid]
    lo = np.percentile(values, 1)
    hi = np.percentile(values, 99)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(values))
        hi = float(np.max(values)) if np.max(values) > np.min(values) else float(np.min(values) + 1.0)

    norm = np.zeros_like(arr, dtype=np.float64)
    norm[valid] = np.clip((arr[valid] - lo) / (hi - lo), 0.0, 1.0)
    mask = norm > 0.03

    if np.count_nonzero(mask) < 64:
        mask = np.abs(arr) > 1e-6
    return mask


def infer_foreground_mask_from_array(arr: np.ndarray) -> np.ndarray:
    data = np.asarray(arr, dtype=np.float64)
    valid = np.isfinite(data)
    if not np.any(valid):
        return np.zeros_like(data, dtype=bool)

    values = data[valid]
    lo = np.percentile(values, 1)
    hi = np.percentile(values, 99)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(values))
        hi = float(np.max(values)) if np.max(values) > np.min(values) else float(np.min(values) + 1.0)

    norm = np.zeros_like(data, dtype=np.float64)
    norm[valid] = np.clip((data[valid] - lo) / (hi - lo), 0.0, 1.0)
    mask = norm > 0.03
    if np.count_nonzero(mask) < 64:
        mask = np.abs(data) > 1e-6
    return mask


def bbox_corners_from_mask(image: sitk.Image, mask_zyx: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask_zyx)
    if coords.size == 0:
        return image_corner_points_physical(image)

    z_min, y_min, x_min = coords.min(axis=0).astype(np.float64)
    z_max, y_max, x_max = coords.max(axis=0).astype(np.float64)

    x_vals = (max(-0.5, x_min - 0.5), x_max + 0.5)
    y_vals = (max(-0.5, y_min - 0.5), y_max + 0.5)
    z_vals = (max(-0.5, z_min - 0.5), z_max + 0.5)

    corners = []
    for x in x_vals:
        for y in y_vals:
            for z in z_vals:
                corners.append(image.TransformContinuousIndexToPhysicalPoint((float(x), float(y), float(z))))
    return np.asarray(corners, dtype=np.float64)


def physical_points_to_local_mm(points_phys: np.ndarray, origin: np.ndarray, direction: np.ndarray) -> np.ndarray:
    rel = np.asarray(points_phys, dtype=np.float64) - np.asarray(origin, dtype=np.float64).reshape(1, 3)
    return rel @ np.asarray(direction, dtype=np.float64)


def image_center_physical(image: sitk.Image) -> np.ndarray:
    size_xyz = np.asarray(image.GetSize(), dtype=np.float64)
    center_xyz = (size_xyz - 1.0) / 2.0
    return np.asarray(image.TransformContinuousIndexToPhysicalPoint(tuple(center_xyz.tolist())), dtype=np.float64)


def physical_point_to_zyx_index(
    reference_image: sitk.Image,
    point_phys: np.ndarray,
    shape_zyx: Tuple[int, int, int],
) -> Tuple[int, int, int]:
    idx_xyz = np.asarray(
        reference_image.TransformPhysicalPointToContinuousIndex(tuple(np.asarray(point_phys, dtype=np.float64).tolist())),
        dtype=np.float64,
    )
    idx_zyx = np.round([idx_xyz[2], idx_xyz[1], idx_xyz[0]]).astype(int)
    return (
        max(0, min(int(idx_zyx[0]), shape_zyx[0] - 1)),
        max(0, min(int(idx_zyx[1]), shape_zyx[1] - 1)),
        max(0, min(int(idx_zyx[2]), shape_zyx[2] - 1)),
    )


def compute_crop_box_from_masks(
    masks_zyx: Sequence[np.ndarray],
    pad_ratio: float = 0.12,
    min_pad: int = 10,
) -> Optional[Tuple[int, int, int, int, int, int]]:
    if not masks_zyx:
        return None

    union = np.zeros_like(np.asarray(masks_zyx[0]), dtype=bool)
    for mask in masks_zyx:
        union |= np.asarray(mask, dtype=bool)

    coords = np.argwhere(union)
    if coords.size == 0:
        return None

    z_min, y_min, x_min = coords.min(axis=0)
    z_max, y_max, x_max = coords.max(axis=0)

    size_z = int(z_max - z_min + 1)
    size_y = int(y_max - y_min + 1)
    size_x = int(x_max - x_min + 1)

    pad_z = max(min_pad, int(round(size_z * pad_ratio)))
    pad_y = max(min_pad, int(round(size_y * pad_ratio)))
    pad_x = max(min_pad, int(round(size_x * pad_ratio)))

    dz, dy, dx = union.shape
    z_min = max(0, int(z_min) - pad_z)
    z_max = min(dz - 1, int(z_max) + pad_z)
    y_min = max(0, int(y_min) - pad_y)
    y_max = min(dy - 1, int(y_max) + pad_y)
    x_min = max(0, int(x_min) - pad_x)
    x_max = min(dx - 1, int(x_max) + pad_x)
    return z_min, z_max, y_min, y_max, x_min, x_max


def crop_masks_to_box(
    masks_zyx: Sequence[np.ndarray],
    crop_box: Optional[Tuple[int, int, int, int, int, int]],
) -> Sequence[np.ndarray]:
    if crop_box is None:
        return [np.asarray(mask, dtype=bool) for mask in masks_zyx]
    z_min, z_max, y_min, y_max, x_min, x_max = crop_box
    return [np.asarray(mask, dtype=bool)[z_min : z_max + 1, y_min : y_max + 1, x_min : x_max + 1] for mask in masks_zyx]


def crop_image_to_box(
    image: sitk.Image,
    crop_box: Optional[Tuple[int, int, int, int, int, int]],
) -> sitk.Image:
    if crop_box is None:
        return sitk.Image(image)
    z_min, z_max, y_min, y_max, x_min, x_max = crop_box
    index_xyz = [int(x_min), int(y_min), int(z_min)]
    size_xyz = [int(x_max - x_min + 1), int(y_max - y_min + 1), int(z_max - z_min + 1)]
    return sitk.RegionOfInterest(image, size=size_xyz, index=index_xyz)


def support_mask_from_image(image: sitk.Image) -> np.ndarray:
    return sitk.GetArrayFromImage(image) > 0.5


def resolve_effective_crop_padding(crop_pad_ratio: float, crop_min_pad: int) -> Tuple[float, int]:
    effective_ratio = float(crop_pad_ratio) if crop_pad_ratio > 0 else 0.20
    effective_min_pad = int(crop_min_pad) if crop_min_pad > 0 else 24
    return effective_ratio, effective_min_pad


def build_visualization_reference(
    fixed_image: sitk.Image,
    moving_image: sitk.Image,
    tm_phys: np.ndarray,
    canvas_pad_mm: float,
) -> sitk.Image:
    fixed_full_corners = image_corner_points_physical(fixed_image)
    moving_full_corners = image_corner_points_physical(moving_image)
    moving_after_corners = apply_tm_points(moving_full_corners, tm_phys)
    all_points = np.vstack([fixed_full_corners, moving_full_corners, moving_after_corners])

    origin = np.asarray(fixed_image.GetOrigin(), dtype=np.float64)
    direction = np.asarray(fixed_image.GetDirection(), dtype=np.float64).reshape(3, 3)
    spacing = np.asarray(fixed_image.GetSpacing(), dtype=np.float64)

    local_mm = physical_points_to_local_mm(all_points, origin=origin, direction=direction)
    # Be intentionally conservative here:
    # 1) a fixed mm padding from the CLI argument
    # 2) at least two voxels on each axis
    # 3) an extent-proportional padding so rotated anatomy is not clipped near borders
    local_extent = np.maximum(local_mm.max(axis=0) - local_mm.min(axis=0), spacing)
    extra_margin = np.maximum.reduce(
        [
            np.full(3, float(canvas_pad_mm), dtype=np.float64),
            2.0 * spacing,
            0.35 * local_extent,
        ]
    )
    local_min = local_mm.min(axis=0) - extra_margin
    local_max = local_mm.max(axis=0) + extra_margin

    size_xyz = np.maximum(1, np.ceil((local_max - local_min) / spacing).astype(int) + 1)
    new_origin = origin + direction @ local_min

    reference = sitk.Image([int(size_xyz[0]), int(size_xyz[1]), int(size_xyz[2])], fixed_image.GetPixelID())
    reference.SetOrigin(tuple(new_origin.tolist()))
    reference.SetSpacing(tuple(spacing.tolist()))
    reference.SetDirection(tuple(direction.reshape(-1).tolist()))
    return reference


def build_transformed_native_reference(
    moving_image: sitk.Image,
    tm_phys: np.ndarray,
    extra_pad_mm: float = 2.0,
) -> sitk.Image:
    spacing = np.asarray(moving_image.GetSpacing(), dtype=np.float64)
    direction_in = np.asarray(moving_image.GetDirection(), dtype=np.float64).reshape(3, 3)
    rotation = np.asarray(tm_phys[:3, :3], dtype=np.float64)

    direction_out = rotation @ direction_in
    u, _, vt = np.linalg.svd(direction_out)
    direction_out = u @ vt

    origin_in = np.asarray(moving_image.GetOrigin(), dtype=np.float64).reshape(1, 3)
    origin_after = apply_tm_points(origin_in, tm_phys)[0]
    corners_after = apply_tm_points(image_corner_points_physical(moving_image), tm_phys)
    local_mm = physical_points_to_local_mm(corners_after, origin=origin_after, direction=direction_out)

    extra_margin = np.maximum(2.0 * spacing, float(extra_pad_mm))
    local_min = local_mm.min(axis=0) - extra_margin
    local_max = local_mm.max(axis=0) + extra_margin
    size_xyz = np.maximum(1, np.ceil((local_max - local_min) / spacing).astype(int) + 1)
    origin_out = origin_after + direction_out @ local_min

    reference = sitk.Image([int(size_xyz[0]), int(size_xyz[1]), int(size_xyz[2])], moving_image.GetPixelID())
    reference.SetOrigin(tuple(origin_out.tolist()))
    reference.SetSpacing(tuple(spacing.tolist()))
    reference.SetDirection(tuple(direction_out.reshape(-1).tolist()))
    return reference


def resample_image_to_reference(
    input_image: sitk.Image,
    reference_image: sitk.Image,
    tm_phys: np.ndarray,
    interpolator: int,
    default_value: float = 0.0,
) -> sitk.Image:
    transform = build_sitk_affine(tm_phys)
    inverse_transform = transform.GetInverse()
    return sitk.Resample(
        input_image,
        reference_image,
        inverse_transform,
        interpolator,
        default_value,
        input_image.GetPixelID(),
    )


def resample_moving_to_fixed(
    moving_image: sitk.Image,
    fixed_image: sitk.Image,
    tm_phys: np.ndarray,
    interpolator: int,
    default_value: float = 0.0,
) -> sitk.Image:
    return resample_image_to_reference(moving_image, fixed_image, tm_phys, interpolator, default_value)


def robust_normalize(image_array: np.ndarray) -> np.ndarray:
    arr = np.asarray(image_array, dtype=np.float64)
    valid = np.isfinite(arr)
    if not np.any(valid):
        return np.zeros_like(arr, dtype=np.float64)
    v = arr[valid]
    lo = np.percentile(v, 1)
    hi = np.percentile(v, 99)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(v))
        hi = float(np.max(v)) if np.max(v) > np.min(v) else float(np.min(v) + 1.0)
    out = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return out


def choose_slice_center_from_mask(
    mask_zyx: np.ndarray,
    shape_zyx: Tuple[int, int, int],
    mode: str,
) -> Tuple[int, int, int]:
    if mode == "foreground_center":
        return mask_center_zyx(mask_zyx, shape_zyx)
    return shape_zyx[0] // 2, shape_zyx[1] // 2, shape_zyx[2] // 2


def extract_views(volume_zyx: np.ndarray, zyx_index: Tuple[int, int, int]) -> Dict[str, np.ndarray]:
    z, y, x = zyx_index
    z = max(0, min(z, volume_zyx.shape[0] - 1))
    y = max(0, min(y, volume_zyx.shape[1] - 1))
    x = max(0, min(x, volume_zyx.shape[2] - 1))
    return {
        "axial": volume_zyx[z, :, :],
        "coronal": volume_zyx[:, y, :],
        "sagittal": volume_zyx[:, :, x],
    }


def compute_2d_crop_box_from_masks(
    masks_2d: Sequence[np.ndarray],
    pad_ratio: float = 0.30,
    min_pad: int = 24,
) -> Optional[Tuple[int, int, int, int]]:
    if not masks_2d:
        return None

    union = np.zeros_like(np.asarray(masks_2d[0]), dtype=bool)
    for mask in masks_2d:
        union |= np.asarray(mask, dtype=bool)

    coords = np.argwhere(union)
    if coords.size == 0:
        return None

    r_min, c_min = coords.min(axis=0)
    r_max, c_max = coords.max(axis=0)

    size_r = int(r_max - r_min + 1)
    size_c = int(c_max - c_min + 1)
    pad_r = max(int(min_pad), int(round(size_r * pad_ratio)))
    pad_c = max(int(min_pad), int(round(size_c * pad_ratio)))

    h, w = union.shape
    r_min = max(0, int(r_min) - pad_r)
    r_max = min(h - 1, int(r_max) + pad_r)
    c_min = max(0, int(c_min) - pad_c)
    c_max = min(w - 1, int(c_max) + pad_c)
    return r_min, r_max, c_min, c_max


def crop_view_2d(
    image_2d: np.ndarray,
    crop_box: Optional[Tuple[int, int, int, int]],
) -> np.ndarray:
    if crop_box is None:
        return image_2d
    r_min, r_max, c_min, c_max = crop_box
    return image_2d[r_min : r_max + 1, c_min : c_max + 1]


def crop_views_with_boxes(
    views: Dict[str, np.ndarray],
    crop_boxes: Dict[str, Optional[Tuple[int, int, int, int]]],
) -> Dict[str, np.ndarray]:
    return {name: crop_view_2d(views[name], crop_boxes.get(name)) for name in ("axial", "coronal", "sagittal")}


def compute_view_crop_boxes(
    mask_views_group: Sequence[Dict[str, np.ndarray]],
    pad_ratio: float = 0.30,
    min_pad: int = 24,
) -> Dict[str, Optional[Tuple[int, int, int, int]]]:
    crop_boxes: Dict[str, Optional[Tuple[int, int, int, int]]] = {}
    for name in ("axial", "coronal", "sagittal"):
        crop_boxes[name] = compute_2d_crop_box_from_masks(
            [views[name] for views in mask_views_group],
            pad_ratio=pad_ratio,
            min_pad=min_pad,
        )
    return crop_boxes


def mask_center_zyx(mask_zyx: np.ndarray, fallback_shape: Tuple[int, int, int]) -> Tuple[int, int, int]:
    coords = np.argwhere(np.asarray(mask_zyx, dtype=bool))
    if coords.size == 0:
        return fallback_shape[0] // 2, fallback_shape[1] // 2, fallback_shape[2] // 2
    center = np.round(coords.mean(axis=0)).astype(int)
    return int(center[0]), int(center[1]), int(center[2])


def compute_view_window_sizes(
    masks_zyx: Sequence[np.ndarray],
    pad_ratio: float = 0.20,
    min_pad: int = 24,
) -> Dict[str, Tuple[int, int]]:
    union = np.zeros_like(np.asarray(masks_zyx[0]), dtype=bool)
    for mask in masks_zyx:
        union |= np.asarray(mask, dtype=bool)

    coords = np.argwhere(union)
    if coords.size == 0:
        z, y, x = union.shape
        return {
            "axial": (y, x),
            "coronal": (z, x),
            "sagittal": (z, y),
        }

    z_min, y_min, x_min = coords.min(axis=0)
    z_max, y_max, x_max = coords.max(axis=0)

    size_z = int(z_max - z_min + 1)
    size_y = int(y_max - y_min + 1)
    size_x = int(x_max - x_min + 1)

    pad_z = max(min_pad, int(round(size_z * pad_ratio)))
    pad_y = max(min_pad, int(round(size_y * pad_ratio)))
    pad_x = max(min_pad, int(round(size_x * pad_ratio)))

    return {
        "axial": (size_y + 2 * pad_y, size_x + 2 * pad_x),
        "coronal": (size_z + 2 * pad_z, size_x + 2 * pad_x),
        "sagittal": (size_z + 2 * pad_z, size_y + 2 * pad_y),
    }


def extract_centered_patch_2d(image_2d: np.ndarray, center_rc: Tuple[int, int], out_hw: Tuple[int, int]) -> np.ndarray:
    out_h, out_w = int(out_hw[0]), int(out_hw[1])
    out_h = max(1, out_h)
    out_w = max(1, out_w)

    cy, cx = int(center_rc[0]), int(center_rc[1])
    y0 = cy - out_h // 2
    x0 = cx - out_w // 2
    y1 = y0 + out_h
    x1 = x0 + out_w

    src_y0 = max(0, y0)
    src_x0 = max(0, x0)
    src_y1 = min(image_2d.shape[0], y1)
    src_x1 = min(image_2d.shape[1], x1)

    dst_y0 = src_y0 - y0
    dst_x0 = src_x0 - x0
    dst_y1 = dst_y0 + (src_y1 - src_y0)
    dst_x1 = dst_x0 + (src_x1 - src_x0)

    patch = np.zeros((out_h, out_w), dtype=image_2d.dtype)
    patch[dst_y0:dst_y1, dst_x0:dst_x1] = image_2d[src_y0:src_y1, src_x0:src_x1]
    return patch


def extract_views_centered(
    volume_zyx: np.ndarray,
    center_zyx: Tuple[int, int, int],
    window_sizes: Dict[str, Tuple[int, int]],
) -> Dict[str, np.ndarray]:
    z, y, x = center_zyx
    z = max(0, min(int(z), volume_zyx.shape[0] - 1))
    y = max(0, min(int(y), volume_zyx.shape[1] - 1))
    x = max(0, min(int(x), volume_zyx.shape[2] - 1))

    axial = extract_centered_patch_2d(volume_zyx[z, :, :], (y, x), window_sizes["axial"])
    coronal = extract_centered_patch_2d(volume_zyx[:, y, :], (z, x), window_sizes["coronal"])
    sagittal = extract_centered_patch_2d(volume_zyx[:, :, x], (z, y), window_sizes["sagittal"])
    return {
        "axial": axial,
        "coronal": coronal,
        "sagittal": sagittal,
    }


def compute_3d_crop_box(
    volumes: Sequence[np.ndarray],
    threshold: float = 1e-6,
    pad_ratio: float = 0.18,
    min_pad: int = 12,
) -> Optional[Tuple[int, int, int, int, int, int]]:
    if not volumes:
        return None

    mask = np.zeros_like(np.asarray(volumes[0]), dtype=bool)
    for volume in volumes:
        arr = np.asarray(volume, dtype=np.float64)
        mask |= np.isfinite(arr) & (np.abs(arr) > threshold)

    coords = np.argwhere(mask)
    if coords.size == 0:
        return None

    z_min, y_min, x_min = coords.min(axis=0)
    z_max, y_max, x_max = coords.max(axis=0)

    dz, dy, dx = mask.shape
    pad_z = max(min_pad, int(round(dz * pad_ratio)))
    pad_y = max(min_pad, int(round(dy * pad_ratio)))
    pad_x = max(min_pad, int(round(dx * pad_ratio)))

    z_min = max(0, int(z_min) - pad_z)
    z_max = min(dz - 1, int(z_max) + pad_z)
    y_min = max(0, int(y_min) - pad_y)
    y_max = min(dy - 1, int(y_max) + pad_y)
    x_min = max(0, int(x_min) - pad_x)
    x_max = min(dx - 1, int(x_max) + pad_x)
    return z_min, z_max, y_min, y_max, x_min, x_max


def crop_volumes_to_content(
    volumes: Sequence[np.ndarray],
    crop_box: Optional[Tuple[int, int, int, int, int, int]],
) -> Sequence[np.ndarray]:
    if crop_box is None:
        return list(volumes)
    z_min, z_max, y_min, y_max, x_min, x_max = crop_box
    return [volume[z_min : z_max + 1, y_min : y_max + 1, x_min : x_max + 1] for volume in volumes]


def prepare_visualization_case(
    moving_image: sitk.Image,
    fixed_image: sitk.Image,
    tm_phys: np.ndarray,
    slice_mode: str,
    canvas_pad_mm: float,
    crop_pad_ratio: float,
    crop_min_pad: int,
) -> Dict[str, object]:
    reference_image = build_visualization_reference(
        fixed_image,
        moving_image,
        tm_phys,
        canvas_pad_mm=canvas_pad_mm,
    )

    fixed_on_canvas = resample_image_to_reference(fixed_image, reference_image, np.eye(4), sitk.sitkLinear, 0.0)
    moving_before_on_canvas = resample_image_to_reference(moving_image, reference_image, np.eye(4), sitk.sitkLinear, 0.0)
    moving_after_on_canvas = resample_image_to_reference(moving_image, reference_image, tm_phys, sitk.sitkLinear, 0.0)

    fixed_support_on_canvas = resample_image_to_reference(
        make_ones_like(fixed_image),
        reference_image,
        np.eye(4),
        sitk.sitkNearestNeighbor,
        0.0,
    )
    moving_before_support_on_canvas = resample_image_to_reference(
        make_ones_like(moving_image),
        reference_image,
        np.eye(4),
        sitk.sitkNearestNeighbor,
        0.0,
    )
    moving_after_support_on_canvas = resample_image_to_reference(
        make_ones_like(moving_image),
        reference_image,
        tm_phys,
        sitk.sitkNearestNeighbor,
        0.0,
    )

    fixed_support_mask = support_mask_from_image(fixed_support_on_canvas)
    moving_before_support_mask = support_mask_from_image(moving_before_support_on_canvas)
    moving_after_support_mask = support_mask_from_image(moving_after_support_on_canvas)

    effective_pad_ratio, effective_min_pad = resolve_effective_crop_padding(crop_pad_ratio, crop_min_pad)
    crop_box = compute_crop_box_from_masks(
        [fixed_support_mask, moving_before_support_mask, moving_after_support_mask],
        pad_ratio=effective_pad_ratio,
        min_pad=effective_min_pad,
    )

    fixed_processed_image = crop_image_to_box(fixed_on_canvas, crop_box)
    moving_before_processed_image = crop_image_to_box(moving_before_on_canvas, crop_box)
    moving_after_processed_image = crop_image_to_box(moving_after_on_canvas, crop_box)

    (
        fixed_support_mask,
        moving_before_support_mask,
        moving_after_support_mask,
    ) = crop_masks_to_box(
        [fixed_support_mask, moving_before_support_mask, moving_after_support_mask],
        crop_box,
    )

    fixed_arr = robust_normalize(sitk.GetArrayFromImage(fixed_processed_image))
    moving_before_arr = robust_normalize(sitk.GetArrayFromImage(moving_before_processed_image))
    moving_after_arr = robust_normalize(sitk.GetArrayFromImage(moving_after_processed_image))

    if slice_mode == "image_center":
        fixed_center = (
            fixed_arr.shape[0] // 2,
            fixed_arr.shape[1] // 2,
            fixed_arr.shape[2] // 2,
        )
        before_center = fixed_center
        after_center = fixed_center
    else:
        fixed_center = choose_slice_center_from_mask(fixed_support_mask, fixed_arr.shape, slice_mode)
        before_center = choose_slice_center_from_mask(moving_before_support_mask, moving_before_arr.shape, slice_mode)
        after_center = choose_slice_center_from_mask(moving_after_support_mask, moving_after_arr.shape, slice_mode)

    return {
        "reference_image": reference_image,
        "crop_box": crop_box,
        "fixed_processed_image": fixed_processed_image,
        "moving_before_processed_image": moving_before_processed_image,
        "moving_after_processed_image": moving_after_processed_image,
        "fixed_arr": fixed_arr,
        "moving_before_arr": moving_before_arr,
        "moving_after_arr": moving_after_arr,
        "fixed_support_mask": fixed_support_mask,
        "moving_before_support_mask": moving_before_support_mask,
        "moving_after_support_mask": moving_after_support_mask,
        "fixed_center": fixed_center,
        "before_center": before_center,
        "after_center": after_center,
        "effective_crop_pad_ratio": effective_pad_ratio,
        "effective_crop_min_pad": effective_min_pad,
    }


def prepare_self_oriented_moving_after(
    moving_image: sitk.Image,
    tm_phys: np.ndarray,
    slice_mode: str,
    crop_pad_ratio: float,
    crop_min_pad: int,
) -> Dict[str, object]:
    reference_image = build_transformed_native_reference(moving_image, tm_phys, extra_pad_mm=2.0)
    moving_after_image = resample_image_to_reference(moving_image, reference_image, tm_phys, sitk.sitkLinear, 0.0)
    moving_after_support = resample_image_to_reference(
        make_ones_like(moving_image),
        reference_image,
        tm_phys,
        sitk.sitkNearestNeighbor,
        0.0,
    )

    support_mask = support_mask_from_image(moving_after_support)
    effective_pad_ratio, effective_min_pad = resolve_effective_crop_padding(crop_pad_ratio, crop_min_pad)
    crop_box = compute_crop_box_from_masks([support_mask], pad_ratio=effective_pad_ratio, min_pad=effective_min_pad)
    moving_after_image = crop_image_to_box(moving_after_image, crop_box)
    support_mask = crop_masks_to_box([support_mask], crop_box)[0]

    moving_after_arr = robust_normalize(sitk.GetArrayFromImage(moving_after_image))
    if slice_mode == "image_center":
        center = (
            moving_after_arr.shape[0] // 2,
            moving_after_arr.shape[1] // 2,
            moving_after_arr.shape[2] // 2,
        )
    else:
        center = choose_slice_center_from_mask(support_mask, moving_after_arr.shape, slice_mode)

    return {
        "reference_image": reference_image,
        "processed_image": moving_after_image,
        "support_mask": support_mask,
        "arr": moving_after_arr,
        "center": center,
        "crop_box": crop_box,
    }


def shift_slice_index(
    zyx_index: Tuple[int, int, int],
    crop_box: Optional[Tuple[int, int, int, int, int, int]],
    cropped_shape: Tuple[int, int, int],
) -> Tuple[int, int, int]:
    z, y, x = zyx_index
    if crop_box is not None:
        z_min, _, y_min, _, x_min, _ = crop_box
        z -= z_min
        y -= y_min
        x -= x_min

    return (
        max(0, min(int(z), cropped_shape[0] - 1)),
        max(0, min(int(y), cropped_shape[1] - 1)),
        max(0, min(int(x), cropped_shape[2] - 1)),
    )


def euler_xyz_from_rotation(r: np.ndarray) -> Tuple[float, float, float]:
    sy = math.sqrt(r[0, 0] * r[0, 0] + r[1, 0] * r[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(r[2, 1], r[2, 2])
        y = math.atan2(-r[2, 0], sy)
        z = math.atan2(r[1, 0], r[0, 0])
    else:
        x = math.atan2(-r[1, 2], r[1, 1])
        y = math.atan2(-r[2, 0], sy)
        z = 0.0
    return tuple(np.rad2deg([x, y, z]))


def save_volume_triptych(
    output_path: Path,
    views: Dict[str, np.ndarray],
    title: str,
) -> None:
    order = ["axial", "coronal", "sagittal"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(title, fontsize=14)

    for col, name in enumerate(order):
        axes[col].imshow(views[name], cmap="gray")
        axes[col].set_title(name)
        axes[col].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_single_slices(output_dir: Path, prefix: str, views: Dict[str, np.ndarray]) -> None:
    for name in ("axial", "coronal", "sagittal"):
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.imshow(views[name], cmap="gray")
        ax.set_title(f"{prefix} {name}")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(output_dir / f"{prefix}_{name}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def save_named_views(output_dir: Path, prefix: str, views: Dict[str, np.ndarray], title_suffix: str = "") -> None:
    triptych_path = output_dir / f"{prefix}_triptych.png"
    title = prefix.replace("_", " ")
    if title_suffix:
        title = f"{title} | {title_suffix}"
    save_volume_triptych(triptych_path, views, title)
    save_single_slices(output_dir, prefix, views)


def main() -> None:
    args = parse_args()
    pair_npz_path = Path(args.pair_npz).resolve()
    result_path = Path(args.result_file).resolve()
    output_dir = Path(args.output_dir).resolve()
    ensure_dir(output_dir)

    pair_data = np.load(pair_npz_path, allow_pickle=True)
    tm_norm, elapsed_time = load_tm_and_runtime(result_path)

    moving_image_path = Path(npz_string(pair_data, "moving_image_file")).resolve()
    fixed_image_path = Path(npz_string(pair_data, "fixed_image_file")).resolve()
    pair_name = npz_string(pair_data, "pair_name", pair_npz_path.stem)

    moving_image = sitk.ReadImage(str(moving_image_path))
    fixed_image = sitk.ReadImage(str(fixed_image_path))

    center_y = np.asarray(pair_data["center_Y"], dtype=np.float64).reshape(3)
    scale = float(pair_data["scale"])
    tm_phys_raw = tm_norm_to_physical(tm_norm, center_y, scale)
    x_phys = np.asarray(pair_data["X"], dtype=np.float64) * scale + center_y
    y_phys = np.asarray(pair_data["Y"], dtype=np.float64) * scale + center_y
    tm_phys, chosen_direction, rmse_forward, rmse_inverse = resolve_tm_direction(
        pair_data,
        tm_phys_raw,
        args.direction_mode,
    )
    rmse_before = nearest_neighbor_rmse(x_phys, y_phys)
    rmse_after = nearest_neighbor_rmse(apply_tm_points(x_phys, tm_phys), y_phys)

    prepared = prepare_visualization_case(
        moving_image=moving_image,
        fixed_image=fixed_image,
        tm_phys=tm_phys,
        slice_mode=args.slice_mode,
        canvas_pad_mm=args.canvas_pad_mm,
        crop_pad_ratio=args.crop_pad_ratio,
        crop_min_pad=args.crop_min_pad,
    )

    reference_image = prepared["reference_image"]
    crop_box = prepared["crop_box"]
    fixed_arr = prepared["fixed_arr"]
    moving_before_arr = prepared["moving_before_arr"]
    moving_after_arr = prepared["moving_after_arr"]
    fixed_support_mask = prepared["fixed_support_mask"]
    moving_before_support_mask = prepared["moving_before_support_mask"]
    moving_after_support_mask = prepared["moving_after_support_mask"]
    fixed_center = prepared["fixed_center"]
    before_center = prepared["before_center"]
    after_center = prepared["after_center"]

    moving_after_self_prepared = prepare_self_oriented_moving_after(
        moving_image=moving_image,
        tm_phys=tm_phys,
        slice_mode=args.slice_mode,
        crop_pad_ratio=args.crop_pad_ratio,
        crop_min_pad=args.crop_min_pad,
    )
    moving_after_self_arr = moving_after_self_prepared["arr"]
    moving_after_self_mask = moving_after_self_prepared["support_mask"]
    moving_after_self_center = moving_after_self_prepared["center"]

    fixed_volume_path = output_dir / "fixed_processed.nii.gz"
    moving_before_volume_path = output_dir / "moving_before_processed.nii.gz"
    moving_after_volume_path = output_dir / "moving_after_processed.nii.gz"

    if args.save_volumes:
        save_image(prepared["fixed_processed_image"], fixed_volume_path)
        save_image(prepared["moving_before_processed_image"], moving_before_volume_path)
        save_image(moving_after_self_prepared["processed_image"], moving_after_volume_path)

    fixed_views_full = extract_views(fixed_arr, fixed_center)
    moving_before_views_full = extract_views(moving_before_arr, fixed_center)
    moving_after_views_full = extract_views(moving_after_arr, fixed_center)

    fixed_mask_views_full = extract_views(fixed_support_mask.astype(np.uint8), fixed_center)
    moving_before_mask_views_full = extract_views(moving_before_support_mask.astype(np.uint8), fixed_center)
    moving_after_mask_views_full = extract_views(moving_after_support_mask.astype(np.uint8), fixed_center)
    full_crop_boxes = compute_view_crop_boxes(
        [fixed_mask_views_full, moving_before_mask_views_full, moving_after_mask_views_full],
        pad_ratio=0.30,
        min_pad=24,
    )
    fixed_views_full = crop_views_with_boxes(fixed_views_full, full_crop_boxes)
    moving_before_views_full = crop_views_with_boxes(moving_before_views_full, full_crop_boxes)
    moving_after_views_full = crop_views_with_boxes(moving_after_views_full, full_crop_boxes)

    fixed_views_self = extract_views(fixed_arr, fixed_center)
    moving_before_views_self = extract_views(moving_before_arr, before_center)
    moving_after_views_self = extract_views(moving_after_self_arr, moving_after_self_center)

    fixed_mask_views_self = extract_views(fixed_support_mask.astype(np.uint8), fixed_center)
    moving_before_mask_views_self = extract_views(moving_before_support_mask.astype(np.uint8), before_center)
    moving_after_mask_views_self = extract_views(moving_after_self_mask.astype(np.uint8), moving_after_self_center)
    fixed_self_crop_boxes = compute_view_crop_boxes([fixed_mask_views_self], pad_ratio=0.30, min_pad=24)
    before_self_crop_boxes = compute_view_crop_boxes([moving_before_mask_views_self], pad_ratio=0.30, min_pad=24)
    after_self_crop_boxes = compute_view_crop_boxes([moving_after_mask_views_self], pad_ratio=0.30, min_pad=24)
    fixed_views_self = crop_views_with_boxes(fixed_views_self, fixed_self_crop_boxes)
    moving_before_views_self = crop_views_with_boxes(moving_before_views_self, before_self_crop_boxes)
    moving_after_views_self = crop_views_with_boxes(moving_after_views_self, after_self_crop_boxes)

    if args.slice_output_mode in {"full", "both"}:
        save_named_views(output_dir, "moving_before_fixed_aligned_full", moving_before_views_full, f"{pair_name} | fixed aligned full")
        save_named_views(output_dir, "moving_after_fixed_aligned_full", moving_after_views_full, f"{pair_name} | fixed aligned full")
        if args.save_fixed:
            save_named_views(output_dir, "fixed_fixed_aligned_full", fixed_views_full, f"{pair_name} | fixed aligned full")

    if args.slice_output_mode in {"centered", "both"}:
        save_named_views(output_dir, "moving_before_self_centered_full", moving_before_views_self, f"{pair_name} | self centered full")
        save_named_views(output_dir, "moving_after_self_centered_full", moving_after_views_self, f"{pair_name} | self centered full")
        if args.save_fixed:
            save_named_views(output_dir, "fixed_self_centered_full", fixed_views_self, f"{pair_name} | self centered full")

    r = tm_phys[:3, :3]
    t = tm_phys[:3, 3]
    euler_x, euler_y, euler_z = euler_xyz_from_rotation(r)

    summary = {
        "pair_npz": str(pair_npz_path),
        "result_file": str(result_path),
        "pair_name": pair_name,
        "moving_image_file": str(moving_image_path),
        "fixed_image_file": str(fixed_image_path),
        "elapsed_time_s": elapsed_time,
        "direction_mode": args.direction_mode,
        "chosen_direction": chosen_direction,
        "point_rmse_before_mm": rmse_before,
        "point_rmse_after_mm": rmse_after,
        "point_rmse_forward_mm": rmse_forward,
        "point_rmse_inverse_mm": rmse_inverse,
        "translation_norm_mm": float(np.linalg.norm(t)),
        "rotation_angle_deg": rotation_angle_deg(tm_phys),
        "translation_mm": {
            "tx": float(t[0]),
            "ty": float(t[1]),
            "tz": float(t[2]),
        },
        "rotation_euler_deg_xyz": {
            "rx": float(euler_x),
            "ry": float(euler_y),
            "rz": float(euler_z),
        },
        "tm_physical": tm_phys.tolist(),
        "slice_index_zyx": {
            "z": int(fixed_center[0]),
            "y": int(fixed_center[1]),
            "x": int(fixed_center[2]),
        },
        "moving_before_center_zyx": {
            "z": int(before_center[0]),
            "y": int(before_center[1]),
            "x": int(before_center[2]),
        },
        "moving_after_center_zyx": {
            "z": int(moving_after_self_center[0]),
            "y": int(moving_after_self_center[1]),
            "x": int(moving_after_self_center[2]),
        },
        "slice_output_mode": args.slice_output_mode,
        "canvas_pad_mm": float(args.canvas_pad_mm),
        "saved_volumes": {
            "fixed_processed": str(fixed_volume_path) if args.save_volumes else "",
            "moving_before_processed": str(moving_before_volume_path) if args.save_volumes else "",
            "moving_after_processed": str(moving_after_volume_path) if args.save_volumes else "",
        },
        "reference_size_xyz": [int(v) for v in reference_image.GetSize()],
        "cropped_size_zyx": [int(v) for v in fixed_arr.shape],
        "crop_box_zyx": list(crop_box) if crop_box is not None else None,
        "effective_crop_pad_ratio": float(prepared["effective_crop_pad_ratio"]),
        "effective_crop_min_pad": int(prepared["effective_crop_min_pad"]),
        "moving_after_self_oriented_size_zyx": [int(v) for v in moving_after_self_arr.shape],
        "moving_after_self_oriented_crop_box_zyx": list(moving_after_self_prepared["crop_box"]) if moving_after_self_prepared["crop_box"] is not None else None,
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Saved visualization to: {output_dir}")
    if args.slice_output_mode in {"full", "both"}:
        print(f"Before fixed-aligned figure: {output_dir / 'moving_before_fixed_aligned_full_triptych.png'}")
        print(f"After fixed-aligned figure: {output_dir / 'moving_after_fixed_aligned_full_triptych.png'}")
    if args.slice_output_mode in {"centered", "both"}:
        print(f"Before self-centered figure: {output_dir / 'moving_before_self_centered_full_triptych.png'}")
        print(f"After self-centered figure: {output_dir / 'moving_after_self_centered_full_triptych.png'}")
    if args.save_volumes:
        print(f"Saved 3D volume: {moving_after_volume_path}")
    print(f"Chosen direction: {chosen_direction}")
    print(f"Point RMSE before: {rmse_before:.6f} mm")
    print(f"Point RMSE after : {rmse_after:.6f} mm")


if __name__ == "__main__":
    main()
