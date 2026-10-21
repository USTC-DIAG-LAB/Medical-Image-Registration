"""Physical geometry and metric definitions for Python registration."""
import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import binary_erosion, binary_fill_holes, distance_transform_edt, label as ndi_label
from scipy.spatial import cKDTree

MODALITY_GROUPS = {
    "CT_LIKE": {"CT", "CBCT"},
    "PET_LIKE": {"PET", "SPECT"},
    "MRI_LIKE": {"MRI"},
}


def npz_string(data, key: str, default: str = "") -> str:
    if key not in data:
        return default
    value = np.asarray(data[key])
    try:
        if value.shape == ():
            return str(value.item())
        return str(value.squeeze().item())
    except Exception:
        return str(value)

def tm_norm_to_physical(tm_norm: np.ndarray, center_y: np.ndarray, scale: float) -> np.ndarray:
    tm_norm = np.asarray(tm_norm, dtype=np.float64)
    center_y = np.asarray(center_y, dtype=np.float64).reshape(3)
    scale = float(scale)

    R = tm_norm[:3, :3]
    t_norm = tm_norm[:3, 3]
    t_phys = scale * t_norm + center_y - R @ center_y

    tm_phys = np.eye(4, dtype=np.float64)
    tm_phys[:3, :3] = R
    tm_phys[:3, 3] = t_phys
    return tm_phys

def apply_tm_points(points: np.ndarray, tm: np.ndarray) -> np.ndarray:
    R = tm[:3, :3]
    t = tm[:3, 3]
    return points @ R.T + t

def nearest_neighbor_rmse(moving_points: np.ndarray, fixed_points: np.ndarray) -> float:
    tree = cKDTree(fixed_points)
    distances, _ = tree.query(moving_points)
    return float(np.sqrt(np.mean(distances ** 2)))

def build_sitk_affine(tm_phys: np.ndarray) -> sitk.AffineTransform:
    transform = sitk.AffineTransform(3)
    transform.SetMatrix(np.asarray(tm_phys[:3, :3], dtype=np.float64).reshape(-1).tolist())
    transform.SetTranslation(np.asarray(tm_phys[:3, 3], dtype=np.float64).tolist())
    return transform

def resample_moving_to_fixed(
    moving_image: sitk.Image,
    fixed_image: sitk.Image,
    tm_phys: np.ndarray,
    interpolator: int,
    default_value: float = 0.0,
) -> sitk.Image:
    transform = build_sitk_affine(tm_phys)
    inverse_transform = transform.GetInverse()
    return sitk.Resample(
        moving_image,
        fixed_image,
        inverse_transform,
        interpolator,
        default_value,
        moving_image.GetPixelID(),
    )

def get_spacing_zyx(image: sitk.Image) -> Tuple[float, float, float]:
    spacing = image.GetSpacing()
    return float(spacing[2]), float(spacing[1]), float(spacing[0])

def modality_group(modality: str) -> str:
    modality_upper = str(modality).upper()
    for group_name, members in MODALITY_GROUPS.items():
        if modality_upper in members:
            return group_name
    return "UNKNOWN"

def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    labeled, num = ndi_label(np.asarray(mask, dtype=bool))
    if num <= 1:
        return np.asarray(mask, dtype=bool)
    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    best = int(np.argmax(counts))
    return labeled == best

def postprocess_foreground_mask(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return mask
    mask = binary_fill_holes(mask)
    mask = keep_largest_component(mask)
    return mask

def otsu_mask_from_image(image: sitk.Image) -> np.ndarray:
    arr = sitk.GetArrayFromImage(image).astype(np.float32)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.zeros(arr.shape, dtype=bool)

    vals = arr[finite]
    lo = float(np.percentile(vals, 1))
    hi = float(np.percentile(vals, 99))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(vals))
        hi = float(np.max(vals))

    clipped = np.clip(arr, lo, hi).astype(np.float32)
    clipped_image = sitk.GetImageFromArray(clipped)
    mask_image = sitk.OtsuThreshold(clipped_image, 0, 1)
    return sitk.GetArrayFromImage(mask_image).astype(bool)

def infer_foreground_mask_for_eval(image: sitk.Image, modality: str) -> np.ndarray:
    arr = sitk.GetArrayFromImage(image).astype(np.float32)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.zeros(arr.shape, dtype=bool)

    vals = arr[finite]
    group = modality_group(modality)
    if group == "CT_LIKE":
        body_thr = max(-700.0, float(np.percentile(vals, 5)))
        mask = arr > body_thr
    elif group == "MRI_LIKE":
        mask = otsu_mask_from_image(image)
    elif group == "PET_LIKE":
        positive = vals[vals > 0]
        if positive.size == 0:
            return np.zeros(arr.shape, dtype=bool)
        thr = max(float(np.percentile(positive, 70)), 0.10 * float(np.max(positive)))
        mask = arr >= thr
    else:
        mask = otsu_mask_from_image(image)

    mask &= finite
    if not np.any(mask):
        fallback_thr = float(np.percentile(vals, 80))
        mask = (arr >= fallback_thr) & finite
    return postprocess_foreground_mask(mask)

def get_directed_surface_distances(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    spacing_zyx: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Return directed surface distances A->B and B->A in physical units."""
    mask_a = np.asarray(mask_a > 0, dtype=bool)
    mask_b = np.asarray(mask_b > 0, dtype=bool)

    if not np.any(mask_a) or not np.any(mask_b):
        empty = np.asarray([], dtype=np.float64)
        return empty, empty

    surface_a = np.logical_xor(mask_a, binary_erosion(mask_a))
    surface_b = np.logical_xor(mask_b, binary_erosion(mask_b))

    if not np.any(surface_a) or not np.any(surface_b):
        empty = np.asarray([], dtype=np.float64)
        return empty, empty

    dt_a = distance_transform_edt(~surface_a, sampling=spacing_zyx)
    dt_b = distance_transform_edt(~surface_b, sampling=spacing_zyx)

    dist_a_to_b = np.asarray(dt_b[surface_a], dtype=np.float64)
    dist_b_to_a = np.asarray(dt_a[surface_b], dtype=np.float64)
    return dist_a_to_b, dist_b_to_a

def compute_dice(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    mask_a = np.asarray(mask_a > 0, dtype=bool)
    mask_b = np.asarray(mask_b > 0, dtype=bool)
    denom = int(mask_a.sum()) + int(mask_b.sum())
    if denom == 0:
        return math.nan
    inter = int(np.logical_and(mask_a, mask_b).sum())
    return float(2.0 * inter / denom)

def compute_asd_assd_hd95(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    spacing_zyx: Sequence[float],
) -> Tuple[float, float, float]:
    """Compute directed ASD, symmetric ASSD, and pooled bidirectional HD95.

    ASD is measured from mask_a (warped moving) to mask_b (fixed). ASSD gives
    the two directed means equal weight. HD95 is the 95th percentile after
    pooling the two directed surface-distance arrays.
    """
    dist_a_to_b, dist_b_to_a = get_directed_surface_distances(mask_a, mask_b, spacing_zyx)
    if dist_a_to_b.size == 0 or dist_b_to_a.size == 0:
        return math.nan, math.nan, math.nan

    asd = float(np.mean(dist_a_to_b))
    assd = float(0.5 * (np.mean(dist_a_to_b) + np.mean(dist_b_to_a)))
    hd95 = float(np.percentile(np.concatenate([dist_a_to_b, dist_b_to_a]), 95))
    return asd, assd, hd95

def compute_mutual_information(image_a: np.ndarray, image_b: np.ndarray, roi_mask: Optional[np.ndarray] = None, bins: int = 64) -> float:
    a = np.asarray(image_a, dtype=np.float64)
    b = np.asarray(image_b, dtype=np.float64)

    if roi_mask is not None:
        roi = np.asarray(roi_mask > 0, dtype=bool)
        a = a[roi]
        b = b[roi]
    else:
        a = a.reshape(-1)
        b = b.reshape(-1)

    valid = np.isfinite(a) & np.isfinite(b)
    a = a[valid]
    b = b[valid]
    if a.size == 0 or b.size == 0:
        return math.nan

    hist_2d, _, _ = np.histogram2d(a, b, bins=bins)
    pxy = hist_2d / np.sum(hist_2d)
    px = np.sum(pxy, axis=1, keepdims=True)
    py = np.sum(pxy, axis=0, keepdims=True)
    nz = pxy > 0
    mi = np.sum(pxy[nz] * np.log(pxy[nz] / (px @ py)[nz]))
    return float(mi)

def read_landmark_csv(csv_path: Path) -> np.ndarray:
    points = np.loadtxt(str(csv_path), delimiter=",", dtype=np.float64)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    return points

def csv_index_to_physical_points(image: sitk.Image, xyz_points: np.ndarray) -> np.ndarray:
    physical_points = []
    for xyz in np.asarray(xyz_points, dtype=np.float64):
        physical_points.append(image.TransformContinuousIndexToPhysicalPoint(tuple(float(v) for v in xyz)))
    return np.asarray(physical_points, dtype=np.float64)

def find_lung_landmark_csv(project_root: Path, image_path: Path) -> Optional[Path]:
    base_name = image_path.name.replace(".nii.gz", ".csv").replace(".nii", ".csv")
    # Resolve the dataset entry explicitly: recursive glob does not traverse directory symlinks.
    lung_root = (project_root / "LungCT").resolve()
    candidates = list(lung_root.glob(f"**/landmarksTr/{base_name}"))
    if not candidates:
        candidates = list(project_root.glob(f"**/landmarksTr/{base_name}"))
    return candidates[0] if candidates else None

def evaluate_case(dataset_name: str, pair_data, tm_phys: np.ndarray, project_root: Path) -> Dict[str, object]:
    """Evaluate independent metric families; missing annotations never erase RMSE."""
    result = {}
    errors = []
    statuses = []

    def attempt(family, operation):
        try:
            values = operation()
            if values is None:
                status = "unavailable"
            else:
                result.update(values)
                numbers = [v for v in values.values() if isinstance(v, (float, np.floating))]
                status = "ok" if numbers and all(np.isfinite(v) for v in numbers) else "undefined"
        except Exception as exc:
            status = "error"
            errors.append(f"{family}: {exc}")
        result[f"{family}_status"] = status
        statuses.append(status)

    def rmse():
        center = np.asarray(pair_data["center_Y"], dtype=float).reshape(3)
        scale = float(pair_data["scale"])
        x = np.asarray(pair_data["X"], dtype=float) * scale + center
        y = np.asarray(pair_data["Y"], dtype=float) * scale + center
        return {"rmse_mm": nearest_neighbor_rmse(apply_tm_points(x, tm_phys), y)}
    attempt("rmse", rmse)

    images = {}
    def image(side):
        if side not in images:
            path = npz_string(pair_data, f"{side}_image_file")
            if not path or not Path(path).is_file():
                raise FileNotFoundError(f"Missing {side} image: {path}")
            images[side] = sitk.ReadImage(path)
        return images[side]

    def references(kind):
        paths = [npz_string(pair_data, f"{side}_{kind}_file") for side in ("moving", "fixed")]
        # Legacy lung files called their ROI masks labels. New inputs keep them separate.
        if kind == "mask" and dataset_name.upper().startswith("LUNG") and "evaluation_schema" not in pair_data:
            paths = [npz_string(pair_data, f"{side}_label_file") for side in ("moving", "fixed")]
        if not all(paths):
            return None
        if not all(Path(p).is_file() for p in paths):
            raise FileNotFoundError(f"Referenced {kind} file missing: {paths}")
        return paths

    def surfaces(kind):
        paths = references(kind)
        if paths is None:
            return None
        moving, fixed = [sitk.ReadImage(p) for p in paths]
        warped = resample_moving_to_fixed(moving, fixed, tm_phys, sitk.sitkNearestNeighbor, 0)
        a, b = sitk.GetArrayFromImage(warped) > 0, sitk.GetArrayFromImage(fixed) > 0
        asd, assd, hd95 = compute_asd_assd_hd95(a, b, get_spacing_zyx(fixed))
        prefix = "organ_" if kind == "label" else ""
        values = {prefix + "asd_mm": asd, prefix + "assd_mm": assd, prefix + "hd95_mm": hd95}
        if kind == "label":
            values["dice"] = compute_dice(a, b)
        return values

    result["surface_source"] = "dataset_roi_mask"
    if dataset_name.upper().startswith("RIRE"):
        result["surface_source"] = "image_inferred_foreground"
        def inferred_surfaces():
            moving, fixed = image("moving"), image("fixed")
            native = infer_foreground_mask_for_eval(moving, npz_string(pair_data, "moving_modality"))
            reference = infer_foreground_mask_for_eval(fixed, npz_string(pair_data, "fixed_modality"))
            mask = sitk.GetImageFromArray(native.astype(np.uint8))
            mask.CopyInformation(moving)
            warped = resample_moving_to_fixed(mask, fixed, tm_phys, sitk.sitkNearestNeighbor, 0)
            asd, assd, hd95 = compute_asd_assd_hd95(sitk.GetArrayFromImage(warped), reference, get_spacing_zyx(fixed))
            return {"asd_mm": asd, "assd_mm": assd, "hd95_mm": hd95}
        attempt("surface", inferred_surfaces)
    else:
        attempt("surface", lambda: surfaces("mask"))
    if dataset_name.upper().startswith("ABDOMEN"):
        attempt("organ", lambda: surfaces("label"))
    else:
        attempt("organ", lambda: None)

    def mutual_information():
        # Old lung NPZ files stored ROI masks under *_label_file; these are
        # not organ annotations and must not make the case eligible for MI.
        if dataset_name.upper().startswith("LUNG") and "evaluation_schema" not in pair_data:
            return None
        label_paths = references("label")
        if label_paths is None:
            return None
        moving_label, fixed_label = [sitk.ReadImage(path) for path in label_paths]
        source = "organ_union_intersect_image_overlap"
        if not all(np.any(sitk.GetArrayFromImage(label) > 0) for label in (moving_label, fixed_label)):
            return {"mi": math.nan, "mi_roi_source": source, "mi_voxel_count": 0}
        moving, fixed = image("moving"), image("fixed")
        # Float output avoids integer quantization during linear interpolation.
        moving_float = sitk.Cast(moving, sitk.sitkFloat32)
        warped = resample_moving_to_fixed(moving_float, fixed, tm_phys, sitk.sitkLinear, 0)
        ones = sitk.GetImageFromArray(np.ones(sitk.GetArrayFromImage(moving).shape, dtype=np.uint8))
        ones.CopyInformation(moving)
        support = resample_moving_to_fixed(ones, fixed, tm_phys, sitk.sitkNearestNeighbor, 0)
        roi = sitk.GetArrayFromImage(support) > 0
        warped_label = resample_moving_to_fixed(moving_label, fixed, tm_phys, sitk.sitkNearestNeighbor, 0)
        fixed_label = resample_moving_to_fixed(fixed_label, fixed, np.eye(4), sitk.sitkNearestNeighbor, 0)
        roi &= (sitk.GetArrayFromImage(warped_label) > 0) | (sitk.GetArrayFromImage(fixed_label) > 0)
        warped_array = sitk.GetArrayFromImage(warped)
        fixed_array = sitk.GetArrayFromImage(fixed)
        roi &= np.isfinite(warped_array) & np.isfinite(fixed_array)
        return {"mi": compute_mutual_information(warped_array, fixed_array, roi),
                "mi_roi_source": source, "mi_voxel_count": int(roi.sum())}
    attempt("mi", mutual_information)

    def landmarks():
        paths = references("landmark")
        if paths is None and "evaluation_schema" not in pair_data and dataset_name.upper().startswith("LUNG"):
            paths = [find_lung_landmark_csv(project_root, Path(npz_string(pair_data, f"{side}_image_file")))
                     for side in ("moving", "fixed")]
            if not all(paths):
                return None
        if paths is None:
            return None
        moving_idx, fixed_idx = [read_landmark_csv(Path(p)) for p in paths]
        if (moving_idx.shape != fixed_idx.shape or moving_idx.ndim != 2 or moving_idx.shape[1] != 3
                or not moving_idx.size or not np.isfinite(moving_idx).all() or not np.isfinite(fixed_idx).all()):
            raise ValueError("Landmarks must be matched, finite Nx3 voxel-index coordinates")
        a = csv_index_to_physical_points(image("moving"), moving_idx)
        b = csv_index_to_physical_points(image("fixed"), fixed_idx)
        tre = np.linalg.norm(apply_tm_points(a, tm_phys) - b, axis=1)
        return {"tre_mean_mm": float(tre.mean()), "tre95_mm": float(np.percentile(tre, 95)),
                "landmark_count": len(tre)}
    attempt("tre", landmarks)
    result["status"] = "partial_error" if "error" in statuses else "success"
    result["message"] = " | ".join(errors)
    return result
