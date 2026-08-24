import argparse
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.io as sio
import scipy.ndimage as ndi
import SimpleITK as sitk


SUPPORTED_SUFFIXES = (
    ".nii.gz",
    ".nii",
    ".mha",
    ".mhd",
    ".nrrd",
    ".img",
    ".hdr",
    ".mnc",
    ".mnc2",
    ".dcm",
    ".ima",
)

MODALITY_KEYWORDS = {
    "CT": {"ct"},
    "PET": {"pet", "pt"},
    "MRI": {"mri", "mr", "t1", "t2", "t1ce", "t1c", "flair", "dwi", "adc", "swi"},
    "SPECT": {"spect", "nm"},
    "CBCT": {"cbct"},
    "XRAY": {"xray", "dx", "cr", "dr", "xa"},
    "US": {"us", "ultrasound", "echo"},
    "OCT": {"oct"},
    "MG": {"mg", "mammo", "mammography"},
}

DICOM_MODALITY_MAP = {
    "CT": "CT",
    "MR": "MRI",
    "PT": "PET",
    "NM": "SPECT",
    "US": "US",
    "XA": "XRAY",
    "DX": "XRAY",
    "CR": "XRAY",
    "MG": "MG",
    "OCT": "OCT",
}

MODALITY_GROUPS = {
    "CT_LIKE": {"CT", "CBCT"},
    "PET_LIKE": {"PET", "SPECT"},
    "MRI_LIKE": {"MRI"},
    "XRAY_LIKE": {"XRAY", "MG"},
    "US_LIKE": {"US"},
    "OCT_LIKE": {"OCT"},
}


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    stem: str
    case_id: str
    modality: str
    channel_code: Optional[str]
    metadata_modality: Optional[str]
    series_description: str
    patient_id: str


def strip_medical_suffix(path: Path) -> str:
    name = path.name
    for suffix in SUPPORTED_SUFFIXES:
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def normalize_token(token: str) -> str:
    return token.strip().lower()


def standardize_modality_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    upper = name.strip().upper()
    if upper in DICOM_MODALITY_MAP:
        return DICOM_MODALITY_MAP[upper]
    return upper


def parse_mapping_items(items: Sequence[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid mapping item: {item}. Expected KEY=VALUE.")
        key, value = item.split("=", 1)
        mapping[key.strip()] = standardize_modality_name(value.strip()) or value.strip()
    return mapping


def split_tokens(stem: str) -> List[str]:
    return [tok for tok in stem.replace("-", "_").split("_") if tok]


def detect_channel_code(tokens: Sequence[str]) -> Optional[str]:
    if not tokens:
        return None
    last = tokens[-1]
    return last if last.isdigit() and len(last) == 4 else None


def read_image_metadata(path: Path) -> Tuple[Optional[str], str, str]:
    try:
        reader = sitk.ImageFileReader()
        reader.SetFileName(str(path))
        reader.LoadPrivateTagsOn()
        reader.ReadImageInformation()
    except Exception:
        return None, "", ""

    metadata_modality = None
    series_description = ""
    patient_id = ""

    if reader.HasMetaDataKey("0008|0060"):
        metadata_modality = standardize_modality_name(reader.GetMetaData("0008|0060"))
    if reader.HasMetaDataKey("0008|103e"):
        series_description = reader.GetMetaData("0008|103e")
    elif reader.HasMetaDataKey("0008|1030"):
        series_description = reader.GetMetaData("0008|1030")
    if reader.HasMetaDataKey("0010|0020"):
        patient_id = reader.GetMetaData("0010|0020").strip()

    return metadata_modality, series_description, patient_id


def detect_modality(
    stem: str,
    channel_map: Dict[str, str],
    metadata_modality: Optional[str],
    series_description: str,
) -> Tuple[str, Optional[str]]:
    tokens = split_tokens(stem)
    channel_code = detect_channel_code(tokens)
    if channel_code and channel_code in channel_map:
        return channel_map[channel_code], channel_code

    if metadata_modality:
        return metadata_modality, channel_code

    lowered = [normalize_token(tok) for tok in tokens]
    lowered.extend(normalize_token(tok) for tok in split_tokens(series_description))
    for modality, keywords in MODALITY_KEYWORDS.items():
        if any(tok in keywords for tok in lowered):
            return modality, channel_code

    if channel_code:
        return f"CH{channel_code}", channel_code
    return "UNKNOWN", None


def derive_case_id(stem: str, parent_name: str, patient_id: str) -> str:
    if patient_id:
        return patient_id

    tokens = split_tokens(stem)
    if not tokens:
        return parent_name or stem

    if detect_channel_code(tokens):
        tokens = tokens[:-1]

    if tokens:
        last_norm = normalize_token(tokens[-1])
        is_modality_token = any(last_norm in keywords for keywords in MODALITY_KEYWORDS.values())
        if is_modality_token:
            tokens = tokens[:-1]

    case_id = "_".join(tokens) if tokens else stem
    if case_id.lower() in {"image", "scan", "series", "volume"} and parent_name:
        return parent_name
    return case_id


def list_medical_files(folder: Path) -> List[Path]:
    files: List[Path] = []
    for path in folder.rglob("*"):
        if not path.is_file():
            continue
        lower_name = path.name.lower()
        if any(lower_name.endswith(suffix) for suffix in SUPPORTED_SUFFIXES):
            files.append(path)
    return sorted(files)


def build_records(folder: Path, channel_map: Dict[str, str]) -> List[ImageRecord]:
    records: List[ImageRecord] = []
    for path in list_medical_files(folder):
        stem = strip_medical_suffix(path)
        metadata_modality, series_description, patient_id = read_image_metadata(path)
        modality, channel_code = detect_modality(stem, channel_map, metadata_modality, series_description)
        case_id = derive_case_id(stem, path.parent.name, patient_id)
        records.append(
            ImageRecord(
                path=path,
                stem=stem,
                case_id=case_id,
                modality=modality,
                channel_code=channel_code,
                metadata_modality=metadata_modality,
                series_description=series_description,
                patient_id=patient_id,
            )
        )
    return records


def index_label_records(records: Sequence[ImageRecord]) -> Tuple[Dict[str, ImageRecord], Dict[Tuple[str, str], List[ImageRecord]], Dict[str, List[ImageRecord]]]:
    by_stem: Dict[str, ImageRecord] = {}
    by_case_modality: Dict[Tuple[str, str], List[ImageRecord]] = {}
    by_case: Dict[str, List[ImageRecord]] = {}

    for record in records:
        by_stem[record.stem] = record
        by_case_modality.setdefault((record.case_id, record.modality), []).append(record)
        by_case.setdefault(record.case_id, []).append(record)
    return by_stem, by_case_modality, by_case


def match_label_record(
    image_record: ImageRecord,
    label_indexes: Tuple[Dict[str, ImageRecord], Dict[Tuple[str, str], List[ImageRecord]], Dict[str, List[ImageRecord]]],
) -> Optional[ImageRecord]:
    by_stem, by_case_modality, by_case = label_indexes

    if image_record.stem in by_stem:
        return by_stem[image_record.stem]

    stem_tokens = split_tokens(image_record.stem)
    if detect_channel_code(stem_tokens):
        no_channel_stem = "_".join(stem_tokens[:-1])
        if no_channel_stem in by_stem:
            return by_stem[no_channel_stem]

    same_case_modality = by_case_modality.get((image_record.case_id, image_record.modality), [])
    if len(same_case_modality) == 1:
        return same_case_modality[0]

    same_case = by_case.get(image_record.case_id, [])
    if len(same_case) == 1:
        return same_case[0]

    return None


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    labeled, num = ndi.label(mask)
    if num <= 1:
        return mask
    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    best = counts.argmax()
    return labeled == best


def postprocess_mask(mask: np.ndarray) -> np.ndarray:
    structure = ndi.generate_binary_structure(rank=3, connectivity=1)
    mask = ndi.binary_closing(mask, structure=structure, iterations=1)
    mask = ndi.binary_opening(mask, structure=structure, iterations=1)
    mask = ndi.binary_fill_holes(mask)
    return keep_largest_component(mask)


def otsu_mask_from_array(arr: np.ndarray) -> np.ndarray:
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros(arr.shape, dtype=bool)

    vals = arr[finite]
    lo = float(np.percentile(vals, 1))
    hi = float(np.percentile(vals, 99))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        hi = float(vals.max())
        lo = float(vals.min())
    clipped = np.clip(arr, lo, hi).astype(np.float32)
    img = sitk.GetImageFromArray(clipped)
    mask_img = sitk.OtsuThreshold(img, 0, 1)
    return sitk.GetArrayFromImage(mask_img).astype(bool)


def modality_group(modality: str) -> str:
    modality_upper = modality.upper()
    for group_name, members in MODALITY_GROUPS.items():
        if modality_upper in members:
            return group_name
    return "UNKNOWN"


def smart_fallback_mask(arr: np.ndarray, finite: np.ndarray) -> np.ndarray:
    vals = arr[finite]
    candidates: List[np.ndarray] = []

    otsu = otsu_mask_from_array(arr) & finite
    if otsu.sum() > 0:
        candidates.append(postprocess_mask(otsu))

    for percentile in (90, 80, 70):
        thr = float(np.percentile(vals, percentile))
        candidate = postprocess_mask((arr >= thr) & finite)
        if candidate.sum() > 0:
            candidates.append(candidate)

    positive = vals[vals > 0]
    if positive.size > 0:
        thr = float(np.percentile(positive, 75))
        candidate = postprocess_mask((arr >= thr) & finite)
        if candidate.sum() > 0:
            candidates.append(candidate)

    if not candidates:
        return np.zeros(arr.shape, dtype=bool)
    return max(candidates, key=lambda x: int(x.sum()))


def infer_foreground_mask(image: sitk.Image, modality: str, processing_mode: str) -> np.ndarray:
    arr = sitk.GetArrayFromImage(image).astype(np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros(arr.shape, dtype=bool)

    vals = arr[finite]
    group = modality_group(modality)

    if group == "CT_LIKE":
        body_thr = max(-700.0, float(np.percentile(vals, 5)))
        mask = arr > body_thr
    elif group == "PET_LIKE":
        positive = vals[vals > 0]
        if positive.size == 0:
            return np.zeros(arr.shape, dtype=bool)
        thr = max(float(np.percentile(positive, 70)), 0.10 * float(positive.max()))
        mask = arr >= thr
    elif group == "MRI_LIKE":
        mask = otsu_mask_from_array(arr)
    elif group in {"XRAY_LIKE", "US_LIKE", "OCT_LIKE"}:
        mask = otsu_mask_from_array(arr)
    else:
        if processing_mode == "smart":
            mask = smart_fallback_mask(arr, finite)
        else:
            mask = otsu_mask_from_array(arr)

    mask &= finite
    if mask.sum() == 0:
        fallback_thr = float(np.percentile(vals, 80))
        mask = (arr >= fallback_thr) & finite
    if mask.sum() == 0:
        return np.zeros(arr.shape, dtype=bool)
    processed = postprocess_mask(mask)
    if processing_mode == "smart" and processed.sum() == 0:
        processed = smart_fallback_mask(arr, finite)
    return processed


def surface_voxels(mask: np.ndarray) -> np.ndarray:
    structure = ndi.generate_binary_structure(rank=3, connectivity=1)
    eroded = ndi.binary_erosion(mask, structure=structure, iterations=1)
    surf = mask & (~eroded)
    return surf if surf.sum() > 0 else mask


def mask_to_points(image: sitk.Image, mask: np.ndarray, max_samples: Optional[int], rng: np.random.Generator) -> np.ndarray:
    z, y, x = np.where(mask)
    num_points = len(x)
    if num_points == 0:
        return np.zeros((0, 3), dtype=np.float32)

    if max_samples is not None and max_samples < num_points:
        idx = rng.choice(num_points, size=max_samples, replace=False)
    else:
        idx = np.arange(num_points)

    points = [image.TransformIndexToPhysicalPoint((int(x[i]), int(y[i]), int(z[i]))) for i in idx]
    return np.asarray(points, dtype=np.float32)


def allocate_integer_counts(weights: Sequence[int], total: int) -> List[int]:
    if total <= 0 or not weights:
        return [0] * len(weights)

    weights_arr = np.asarray(weights, dtype=np.float64)
    weights_arr = np.maximum(weights_arr, 0.0)
    weight_sum = float(weights_arr.sum())
    if weight_sum <= 0:
        base = np.zeros(len(weights), dtype=np.int64)
        base[: min(total, len(weights))] = 1
        return base.tolist()

    raw = weights_arr / weight_sum * total
    counts = np.floor(raw).astype(np.int64)
    remainder = int(total - counts.sum())
    if remainder > 0:
        order = np.argsort(-(raw - counts))
        for idx in order[:remainder]:
            counts[idx] += 1
    return counts.tolist()


def rebalance_counts_to_capacity(
    counts: Sequence[int],
    capacities: Sequence[int],
    preferred_weights: Optional[Sequence[int]] = None,
) -> List[int]:
    counts_arr = np.minimum(np.asarray(counts, dtype=np.int64), np.asarray(capacities, dtype=np.int64))
    capacities_arr = np.asarray(capacities, dtype=np.int64)
    leftover = int(np.asarray(counts, dtype=np.int64).sum() - counts_arr.sum())

    if leftover <= 0:
        return counts_arr.tolist()

    spare = capacities_arr - counts_arr
    if preferred_weights is None:
        weights = np.maximum(spare, 0)
    else:
        weights = np.minimum(np.asarray(preferred_weights, dtype=np.int64), np.maximum(spare, 0))

    while leftover > 0 and np.any(spare > 0):
        extra = allocate_integer_counts(weights.tolist(), leftover)
        extra_arr = np.minimum(np.asarray(extra, dtype=np.int64), spare)
        gained = int(extra_arr.sum())
        if gained <= 0:
            extra_arr = (spare > 0).astype(np.int64)
            gained = int(min(leftover, extra_arr.sum()))
            if gained <= 0:
                break
            chosen = np.where(spare > 0)[0][:gained]
            extra_arr = np.zeros_like(spare)
            extra_arr[chosen] = 1
        counts_arr += extra_arr
        spare -= extra_arr
        leftover -= gained
        if preferred_weights is None:
            weights = np.maximum(spare, 0)
        else:
            weights = np.minimum(np.asarray(preferred_weights, dtype=np.int64), np.maximum(spare, 0))

    return counts_arr.tolist()


def build_label_masks(label_image: sitk.Image, allowed_labels: Optional[Sequence[int]]) -> Tuple[List[np.ndarray], List[int]]:
    arr = sitk.GetArrayFromImage(label_image)
    present = sorted(int(v) for v in np.unique(arr) if v > 0)

    if allowed_labels:
        allowed = [int(v) for v in allowed_labels if int(v) in present]
    else:
        allowed = present

    masks = [(arr == label_id) for label_id in allowed]
    return masks, allowed


def extract_points_from_masks(
    image: sitk.Image,
    masks: Sequence[np.ndarray],
    strategy: str,
    total_samples: Optional[int],
    rng: np.random.Generator,
) -> np.ndarray:
    if not masks:
        return np.zeros((0, 3), dtype=np.float32)

    if strategy == "all":
        parts = [mask_to_points(image, mask, None, rng) for mask in masks]
        return np.concatenate(parts, axis=0) if parts else np.zeros((0, 3), dtype=np.float32)

    if total_samples is None or total_samples <= 0:
        if strategy == "boundary":
            parts = [mask_to_points(image, surface_voxels(mask), None, rng) for mask in masks]
            valid_parts = [part for part in parts if part.shape[0] > 0]
            return np.concatenate(valid_parts, axis=0) if valid_parts else np.zeros((0, 3), dtype=np.float32)
        if strategy == "mixed":
            all_parts: List[np.ndarray] = []
            for mask in masks:
                boundary = surface_voxels(mask)
                interior = mask & (~boundary)
                all_parts.append(mask_to_points(image, boundary, None, rng))
                if interior.sum() > 0:
                    all_parts.append(mask_to_points(image, interior, None, rng))
            valid_parts = [part for part in all_parts if part.shape[0] > 0]
            return np.concatenate(valid_parts, axis=0) if valid_parts else np.zeros((0, 3), dtype=np.float32)

    all_parts: List[np.ndarray] = []
    for mask in masks:
        if strategy == "boundary":
            continue
        elif strategy == "mixed":
            continue
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    if strategy == "boundary":
        boundary_masks = [surface_voxels(mask) for mask in masks]
        boundary_sizes = [int(mask.sum()) for mask in boundary_masks]
        counts = allocate_integer_counts(boundary_sizes, total_samples)
        counts = rebalance_counts_to_capacity(counts, boundary_sizes, boundary_sizes)
        parts = [
            mask_to_points(image, boundary_mask, count if count > 0 else 0, rng)
            for boundary_mask, count in zip(boundary_masks, counts)
            if count > 0
        ]
        valid_parts = [part for part in parts if part.shape[0] > 0]
        return np.concatenate(valid_parts, axis=0) if valid_parts else np.zeros((0, 3), dtype=np.float32)

    if strategy == "mixed":
        mixed_ratio_boundary = 0.8
        target_boundary_total = int(round(total_samples * mixed_ratio_boundary))
        target_boundary_total = min(total_samples, max(0, target_boundary_total))
        target_interior_total = total_samples - target_boundary_total

        boundary_masks = [surface_voxels(mask) for mask in masks]
        interior_masks = [mask & (~boundary) for mask, boundary in zip(masks, boundary_masks)]
        boundary_sizes = [int(mask.sum()) for mask in boundary_masks]
        interior_sizes = [int(mask.sum()) for mask in interior_masks]

        boundary_counts = allocate_integer_counts(boundary_sizes, target_boundary_total)
        interior_counts = allocate_integer_counts(interior_sizes, target_interior_total)

        boundary_counts = rebalance_counts_to_capacity(boundary_counts, boundary_sizes, boundary_sizes)
        interior_counts = rebalance_counts_to_capacity(interior_counts, interior_sizes, interior_sizes)

        used_boundary = sum(boundary_counts)
        used_interior = sum(interior_counts)
        leftover = total_samples - used_boundary - used_interior

        if leftover > 0:
            spare_boundary = [max(0, size - count) for size, count in zip(boundary_sizes, boundary_counts)]
            spare_interior = [max(0, size - count) for size, count in zip(interior_sizes, interior_counts)]
            spare_boundary_total = sum(spare_boundary)
            spare_interior_total = sum(spare_interior)

            if spare_boundary_total > 0:
                add_boundary = min(leftover, spare_boundary_total)
                extra = allocate_integer_counts(spare_boundary, add_boundary)
                boundary_counts = [count + inc for count, inc in zip(boundary_counts, extra)]
                leftover -= add_boundary

            if leftover > 0 and spare_interior_total > 0:
                spare_interior = [max(0, size - count) for size, count in zip(interior_sizes, interior_counts)]
                add_interior = min(leftover, sum(spare_interior))
                extra = allocate_integer_counts(spare_interior, add_interior)
                interior_counts = [count + inc for count, inc in zip(interior_counts, extra)]

        for boundary_mask, boundary_count in zip(boundary_masks, boundary_counts):
            if boundary_count > 0:
                all_parts.append(mask_to_points(image, boundary_mask, boundary_count, rng))
        for interior_mask, interior_count in zip(interior_masks, interior_counts):
            if interior_count > 0:
                all_parts.append(mask_to_points(image, interior_mask, interior_count, rng))

        valid_parts = [part for part in all_parts if part.shape[0] > 0]
        return np.concatenate(valid_parts, axis=0) if valid_parts else np.zeros((0, 3), dtype=np.float32)

    raise ValueError(f"Unknown strategy: {strategy}")


def extract_point_cloud(
    image_record: ImageRecord,
    label_record: Optional[ImageRecord],
    strategy: str,
    num_samples: Optional[int],
    labels_to_use: Optional[Sequence[int]],
    processing_mode: str,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, str, List[int]]:
    if label_record is not None:
        label_image = sitk.ReadImage(str(label_record.path))
        masks, used_labels = build_label_masks(label_image, labels_to_use)
        if not masks:
            raise ValueError(f"No non-zero labels found in {label_record.path}")
        points = extract_points_from_masks(label_image, masks, strategy, num_samples, rng)
        return points, "label", used_labels

    image = sitk.ReadImage(str(image_record.path))
    mask = infer_foreground_mask(image, image_record.modality, processing_mode)
    if mask.sum() == 0:
        raise ValueError(f"Failed to infer foreground from {image_record.path}")
    points = extract_points_from_masks(image, [mask], strategy, num_samples, rng)
    return points, "intensity", []


def normalize_pair(moving_points: np.ndarray, fixed_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    center_fixed = np.mean(fixed_points, axis=0)
    fixed_norm = fixed_points - center_fixed
    moving_norm = moving_points - center_fixed

    scale = float(np.max(np.linalg.norm(fixed_norm, axis=1)))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Failed to compute a positive normalization scale.")

    fixed_norm /= scale
    moving_norm /= scale
    return moving_norm, fixed_norm, center_fixed.astype(np.float64), scale


def choose_pairs(
    image_records: Sequence[ImageRecord],
    fixed_modality: Optional[str],
    moving_modality: Optional[str],
) -> List[Tuple[ImageRecord, ImageRecord]]:
    by_case: Dict[str, List[ImageRecord]] = {}
    for record in image_records:
        by_case.setdefault(record.case_id, []).append(record)

    pairs: List[Tuple[ImageRecord, ImageRecord]] = []
    for case_id, records in sorted(by_case.items()):
        if len(records) < 2:
            continue

        if fixed_modality and moving_modality:
            movings = [r for r in records if r.modality.upper() == moving_modality.upper()]
            fixeds = [r for r in records if r.modality.upper() == fixed_modality.upper()]
            pairs.extend(itertools.product(movings, fixeds))
            continue

        if fixed_modality:
            fixeds = [r for r in records if r.modality.upper() == fixed_modality.upper()]
            movings = [r for r in records if r.modality.upper() != fixed_modality.upper()]
            pairs.extend((moving, fixed) for moving in movings for fixed in fixeds)
            continue

        distinct = []
        seen = set()
        for record in sorted(records, key=lambda x: (x.modality, x.path.name)):
            key = (record.modality, record.path.name)
            if key not in seen:
                seen.add(key)
                distinct.append(record)
        for left, right in itertools.combinations(distinct, 2):
            pairs.append((left, right))

    unique_pairs: List[Tuple[ImageRecord, ImageRecord]] = []
    seen_keys = set()
    for moving, fixed in pairs:
        key = (moving.path, fixed.path)
        if moving.path == fixed.path or key in seen_keys:
            continue
        seen_keys.add(key)
        unique_pairs.append((moving, fixed))
    return unique_pairs


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_pair_outputs(
    output_dir: Path,
    pair_name: str,
    strategy: str,
    moving_points: np.ndarray,
    fixed_points: np.ndarray,
    center_fixed: np.ndarray,
    scale: float,
    moving_record: ImageRecord,
    fixed_record: ImageRecord,
    moving_label: Optional[ImageRecord],
    fixed_label: Optional[ImageRecord],
    moving_source: str,
    fixed_source: str,
    moving_labels_used: Sequence[int],
    fixed_labels_used: Sequence[int],
) -> None:
    npz_path = output_dir / f"{pair_name}_{strategy}.npz"
    mat_path = output_dir / f"{pair_name}_{strategy}.mat"

    np.savez(
        npz_path,
        X=moving_points.astype(np.float64),
        Y=fixed_points.astype(np.float64),
        center_Y=center_fixed.astype(np.float64),
        scale=float(scale),
        strategy=strategy,
        pair_name=pair_name,
        moving_modality=moving_record.modality,
        fixed_modality=fixed_record.modality,
        moving_image_file=str(moving_record.path),
        fixed_image_file=str(fixed_record.path),
        moving_label_file=str(moving_label.path) if moving_label else "",
        fixed_label_file=str(fixed_label.path) if fixed_label else "",
        moving_extraction_source=moving_source,
        fixed_extraction_source=fixed_source,
        moving_labels_used=np.asarray(list(moving_labels_used), dtype=np.int32),
        fixed_labels_used=np.asarray(list(fixed_labels_used), dtype=np.int32),
    )

    sio.savemat(
        mat_path,
        {
            "A": moving_points.T.astype(np.float64),
            "B": fixed_points.T.astype(np.float64),
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build normalized point-cloud pairs for multimodal medical registration."
    )
    parser.add_argument("--images-dir", default="imagesTr", help="Folder containing medical images.")
    parser.add_argument("--labels-dir", default="labelsTr", help="Optional folder containing masks/labels.")
    parser.add_argument("--output-dir", default="datas", help="Folder to save .mat and .npz outputs.")
    parser.add_argument(
        "--strategies",
        default="mixed",
        help="Comma-separated extraction strategies: all,boundary,mixed. Default: mixed",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10000,
        help="Approximate number of samples per image for non-'all' strategies.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260716,
        help="Random seed for point sampling.",
    )
    parser.add_argument(
        "--fixed-modality",
        default=None,
        help="Use this modality as fixed/target. Example: MRI, CT, PET.",
    )
    parser.add_argument(
        "--moving-modality",
        default=None,
        help="Optional explicit moving/source modality when fixed modality is given.",
    )
    parser.add_argument(
        "--channel-map",
        nargs="*",
        default=[],
        help="Optional nnUNet-style channel mapping, e.g. 0000=CT 0001=MRI 0002=PET",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Optional subset of label ids to keep. Example: --labels 1 2 3",
    )
    parser.add_argument(
        "--processing-mode",
        choices=("fast", "smart"),
        default="fast",
        help="fast: prioritize throughput; smart: use stronger fallback for unknown modalities.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    channel_map = parse_mapping_items(args.channel_map)
    strategies = [item.strip().lower() for item in args.strategies.split(",") if item.strip()]
    valid_strategies = {"all", "boundary", "mixed"}
    bad_strategies = [item for item in strategies if item not in valid_strategies]
    if bad_strategies:
        raise ValueError(f"Unsupported strategies: {bad_strategies}")

    images_dir = Path(args.images_dir)
    labels_dir = Path(args.labels_dir)
    output_dir = Path(args.output_dir)

    if not images_dir.exists():
        raise FileNotFoundError(f"Images folder not found: {images_dir}")
    ensure_dir(output_dir)

    print("=" * 72)
    print("Multi-modal medical image point-cloud builder")
    print("=" * 72)
    print(f"Images : {images_dir}")
    print(f"Labels : {labels_dir if labels_dir.exists() else 'not provided / not found'}")
    print(f"Output : {output_dir}")
    print(f"Mode   : {args.processing_mode}")

    image_records = build_records(images_dir, channel_map)
    if not image_records:
        raise RuntimeError(f"No supported medical image files found in {images_dir}")

    label_records = build_records(labels_dir, channel_map) if labels_dir.exists() else []
    label_indexes = index_label_records(label_records)

    pairs = choose_pairs(image_records, args.fixed_modality, args.moving_modality)
    if not pairs:
        raise RuntimeError("No valid multimodal image pairs were found. Check modality naming or channel mapping.")

    rng = np.random.default_rng(args.seed)
    labels_to_use = [int(x) for x in args.labels] if args.labels else None

    print(f"Discovered image files : {len(image_records)}")
    print(f"Discovered label files : {len(label_records)}")
    print(f"Discovered pair count  : {len(pairs)}")

    success_count = 0
    for moving_record, fixed_record in pairs:
        pair_name = f"{moving_record.case_id}__{moving_record.modality}_to_{fixed_record.modality}"
        print(f"\n>> Processing pair: {pair_name}")
        print(f"   Moving: {moving_record.path.name}")
        print(f"   Fixed : {fixed_record.path.name}")

        moving_label = match_label_record(moving_record, label_indexes) if label_records else None
        fixed_label = match_label_record(fixed_record, label_indexes) if label_records else None

        for strategy in strategies:
            print(f"   Strategy: {strategy}")
            try:
                moving_raw, moving_source, moving_labels_used = extract_point_cloud(
                    moving_record,
                    moving_label,
                    strategy,
                    None if strategy == "all" else args.num_samples,
                    labels_to_use,
                    args.processing_mode,
                    rng,
                )
                fixed_raw, fixed_source, fixed_labels_used = extract_point_cloud(
                    fixed_record,
                    fixed_label,
                    strategy,
                    None if strategy == "all" else args.num_samples,
                    labels_to_use,
                    args.processing_mode,
                    rng,
                )
            except Exception as exc:
                print(f"      [skip] extraction failed: {exc}")
                continue

            if moving_raw.shape[0] == 0 or fixed_raw.shape[0] == 0:
                print("      [skip] empty point cloud")
                continue

            try:
                moving_norm, fixed_norm, center_fixed, scale = normalize_pair(moving_raw, fixed_raw)
            except Exception as exc:
                print(f"      [skip] normalization failed: {exc}")
                continue

            save_pair_outputs(
                output_dir=output_dir,
                pair_name=pair_name,
                strategy=strategy,
                moving_points=moving_norm,
                fixed_points=fixed_norm,
                center_fixed=center_fixed,
                scale=scale,
                moving_record=moving_record,
                fixed_record=fixed_record,
                moving_label=moving_label,
                fixed_label=fixed_label,
                moving_source=moving_source,
                fixed_source=fixed_source,
                moving_labels_used=moving_labels_used,
                fixed_labels_used=fixed_labels_used,
            )
            success_count += 1
            print(
                f"      saved | moving_pts={moving_norm.shape[0]} "
                f"fixed_pts={fixed_norm.shape[0]} "
                f"moving_src={moving_source} fixed_src={fixed_source}"
            )

    print(f"\nDone. Saved {success_count} pair outputs into {output_dir}")


if __name__ == "__main__":
    main()
