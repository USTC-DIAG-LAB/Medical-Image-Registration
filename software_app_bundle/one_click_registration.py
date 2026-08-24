import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import SimpleITK as sitk

from batch_prepare_experiments import ensure_dir, reset_dir, stage_image_file
from step1 import SUPPORTED_SUFFIXES
from visualize_registration_result import (
    compute_view_crop_boxes,
    crop_views_with_boxes,
    extract_views,
    load_tm_and_runtime,
    npz_string,
    prepare_self_oriented_moving_after,
    prepare_visualization_case,
    resolve_tm_direction,
    rotation_angle_deg,
    save_image,
    save_named_views,
    tm_norm_to_physical,
)

MATLAB_METHOD = "main"
METHOD_NAME = "modified"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-step medical image registration: preprocess, run MATLAB registration, and export 3D + self-centered slices."
    )
    parser.add_argument("--moving-image", required=True, help="Path to the moving/source image.")
    parser.add_argument("--fixed-image", required=True, help="Path to the fixed/target image.")
    parser.add_argument("--moving-modality", required=True, help="Moving image modality, e.g. CT, MRI, PET.")
    parser.add_argument("--fixed-modality", required=True, help="Fixed image modality, e.g. MRI, CT, PET.")
    parser.add_argument("--output-dir", required=True, help="Root directory for all outputs.")
    parser.add_argument("--case-name", default=None, help="Optional custom case name.")
    parser.add_argument(
        "--strategy",
        choices=("mixed", "boundary", "all"),
        default="mixed",
        help="Point sampling strategy passed to step1.py.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10000,
        help="Approximate point sample count for step1.py when strategy is not 'all'.",
    )
    parser.add_argument(
        "--processing-mode",
        choices=("fast", "smart"),
        default="fast",
        help="Foreground/point extraction mode passed to step1.py.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260722,
        help="Random seed passed to step1.py.",
    )
    parser.add_argument(
        "--matlab-bin",
        default="matlab",
        help="MATLAB executable name or absolute path.",
    )
    parser.add_argument(
        "--canvas-pad-mm",
        type=float,
        default=40.0,
        help="Physical padding used when preparing fixed/self-centered visualization outputs.",
    )
    parser.add_argument(
        "--crop-pad-ratio",
        type=float,
        default=0.0,
        help="Optional 3D crop padding ratio for visualization.",
    )
    parser.add_argument(
        "--crop-min-pad",
        type=int,
        default=0,
        help="Optional minimum 3D crop padding for visualization.",
    )
    parser.add_argument(
        "--slice-mode",
        choices=("foreground_center", "image_center"),
        default="foreground_center",
        help="Representative slice selection mode.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the output directory first if it already exists.",
    )
    return parser.parse_args()


def sanitize_name(text: str) -> str:
    chars = []
    for ch in str(text):
        if ch.isalnum() or ch in {"_", "-", "."}:
            chars.append(ch)
        else:
            chars.append("_")
    cleaned = "".join(chars).strip("_")
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned or "case"


def strip_supported_suffix(path: Path) -> str:
    name = path.name
    lower = name.lower()
    for suffix in SUPPORTED_SUFFIXES:
        if lower.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def full_supported_suffix(path: Path) -> str:
    name = path.name
    lower = name.lower()
    for suffix in SUPPORTED_SUFFIXES:
        if lower.endswith(suffix):
            return name[-len(suffix) :]
    return "".join(path.suffixes) or path.suffix or ".nii.gz"


def matlab_quote(text: str) -> str:
    return str(text).replace("'", "''")


def run_command(cmd, cwd: Path) -> None:
    print("\n[RUN]", " ".join(str(part) for part in cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def stage_single_pair(
    moving_image: Path,
    fixed_image: Path,
    case_name: str,
    output_dir: Path,
) -> Path:
    images_dir = output_dir / "imagesTr"
    reset_dir(images_dir)

    moving_target = images_dir / f"{case_name}_0000{full_supported_suffix(moving_image)}"
    fixed_target = images_dir / f"{case_name}_0001{full_supported_suffix(fixed_image)}"
    stage_image_file(moving_image, moving_target)
    stage_image_file(fixed_image, fixed_target)
    return images_dir


def run_step1_single_pair(
    project_root: Path,
    images_dir: Path,
    step1_output_dir: Path,
    moving_modality: str,
    fixed_modality: str,
    strategy: str,
    num_samples: int,
    processing_mode: str,
    seed: int,
) -> Tuple[Path, Path]:
    reset_dir(step1_output_dir)

    cmd = [
        sys.executable,
        str(project_root / "step1.py"),
        "--images-dir",
        str(images_dir),
        "--output-dir",
        str(step1_output_dir),
        "--strategies",
        strategy,
        "--num-samples",
        str(num_samples),
        "--processing-mode",
        processing_mode,
        "--seed",
        str(seed),
        "--moving-modality",
        moving_modality,
        "--fixed-modality",
        fixed_modality,
        "--channel-map",
        f"0000={moving_modality}",
        f"0001={fixed_modality}",
    ]
    run_command(cmd, cwd=project_root)

    npz_files = sorted(step1_output_dir.glob("*.npz"))
    if len(npz_files) != 1:
        raise RuntimeError(
            f"Expected exactly 1 step1 .npz output in {step1_output_dir}, but found {len(npz_files)}."
        )

    pair_npz = npz_files[0]
    pair_mat = step1_output_dir / f"{pair_npz.stem}.mat"
    if not pair_mat.exists():
        raise FileNotFoundError(f"Matching step1 .mat file not found: {pair_mat}")
    return pair_npz, pair_mat


def run_matlab_registration(
    project_root: Path,
    pair_mat: Path,
    result_path: Path,
    matlab_bin: str,
) -> None:
    ensure_dir(result_path.parent)
    matlab_expr = (
        f"cd('{matlab_quote(project_root)}'); "
        f"addpath(genpath('{matlab_quote(project_root)}')); "
        f"{MATLAB_METHOD}('{matlab_quote(pair_mat)}','{matlab_quote(result_path)}','auto');"
    )
    cmd = [matlab_bin, "-batch", matlab_expr]
    run_command(cmd, cwd=project_root)

    if not result_path.exists():
        raise FileNotFoundError(f"MATLAB finished but result file was not created: {result_path}")


def build_self_views(
    fixed_arr: np.ndarray,
    fixed_support_mask: np.ndarray,
    fixed_center: Tuple[int, int, int],
    moving_after_arr: np.ndarray,
    moving_after_mask: np.ndarray,
    moving_after_center: Tuple[int, int, int],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    fixed_views = extract_views(fixed_arr, fixed_center)
    fixed_mask_views = extract_views(fixed_support_mask.astype(np.uint8), fixed_center)
    fixed_crop_boxes = compute_view_crop_boxes([fixed_mask_views], pad_ratio=0.30, min_pad=24)
    fixed_views = crop_views_with_boxes(fixed_views, fixed_crop_boxes)

    moving_after_views = extract_views(moving_after_arr, moving_after_center)
    moving_after_mask_views = extract_views(moving_after_mask.astype(np.uint8), moving_after_center)
    moving_after_crop_boxes = compute_view_crop_boxes([moving_after_mask_views], pad_ratio=0.30, min_pad=24)
    moving_after_views = crop_views_with_boxes(moving_after_views, moving_after_crop_boxes)

    return fixed_views, moving_after_views


def export_registration_outputs(
    pair_npz: Path,
    result_path: Path,
    output_dir: Path,
    slice_mode: str,
    canvas_pad_mm: float,
    crop_pad_ratio: float,
    crop_min_pad: int,
) -> None:
    ensure_dir(output_dir)

    pair_data = np.load(pair_npz, allow_pickle=True)
    tm_norm, elapsed_time = load_tm_and_runtime(result_path)

    moving_image_path = Path(npz_string(pair_data, "moving_image_file")).resolve()
    fixed_image_path = Path(npz_string(pair_data, "fixed_image_file")).resolve()
    pair_name = npz_string(pair_data, "pair_name", pair_npz.stem)

    moving_image = sitk.ReadImage(str(moving_image_path))
    fixed_image = sitk.ReadImage(str(fixed_image_path))

    center_y = np.asarray(pair_data["center_Y"], dtype=np.float64).reshape(3)
    scale = float(pair_data["scale"])
    tm_phys_raw = tm_norm_to_physical(tm_norm, center_y, scale)
    tm_phys, chosen_direction, rmse_forward, rmse_inverse = resolve_tm_direction(
        pair_data,
        tm_phys_raw,
        "auto",
    )

    prepared = prepare_visualization_case(
        moving_image=moving_image,
        fixed_image=fixed_image,
        tm_phys=tm_phys,
        slice_mode=slice_mode,
        canvas_pad_mm=canvas_pad_mm,
        crop_pad_ratio=crop_pad_ratio,
        crop_min_pad=crop_min_pad,
    )
    moving_after_self = prepare_self_oriented_moving_after(
        moving_image=moving_image,
        tm_phys=tm_phys,
        slice_mode=slice_mode,
        crop_pad_ratio=crop_pad_ratio,
        crop_min_pad=crop_min_pad,
    )

    fixed_views, moving_after_views = build_self_views(
        fixed_arr=prepared["fixed_arr"],
        fixed_support_mask=prepared["fixed_support_mask"],
        fixed_center=prepared["fixed_center"],
        moving_after_arr=moving_after_self["arr"],
        moving_after_mask=moving_after_self["support_mask"],
        moving_after_center=moving_after_self["center"],
    )

    volumes_dir = output_dir / "volumes"
    slices_dir = output_dir / "slices"
    ensure_dir(volumes_dir)
    ensure_dir(slices_dir)

    fixed_volume_path = volumes_dir / "fixed_reference.nii.gz"
    moving_after_volume_path = volumes_dir / "moving_registered_self.nii.gz"
    save_image(prepared["fixed_processed_image"], fixed_volume_path)
    save_image(moving_after_self["processed_image"], moving_after_volume_path)

    save_named_views(slices_dir, "fixed_reference_self", fixed_views, f"{pair_name} | fixed self")
    save_named_views(slices_dir, "moving_registered_self", moving_after_views, f"{pair_name} | registered self")

    summary = {
        "pair_name": pair_name,
        "method": METHOD_NAME,
        "pair_npz": str(pair_npz),
        "result_file": str(result_path),
        "moving_image_file": str(moving_image_path),
        "fixed_image_file": str(fixed_image_path),
        "elapsed_time_s": elapsed_time,
        "chosen_direction": chosen_direction,
        "point_rmse_forward_mm": rmse_forward,
        "point_rmse_inverse_mm": rmse_inverse,
        "rotation_angle_deg": rotation_angle_deg(tm_phys),
        "tm_physical": tm_phys.tolist(),
        "outputs": {
            "fixed_reference_3d": str(fixed_volume_path),
            "moving_registered_3d": str(moving_after_volume_path),
            "fixed_reference_slice_triptych": str(slices_dir / "fixed_reference_self_triptych.png"),
            "moving_registered_slice_triptych": str(slices_dir / "moving_registered_self_triptych.png"),
        },
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()

    project_root = Path(__file__).resolve().parent
    moving_image = Path(args.moving_image).resolve()
    fixed_image = Path(args.fixed_image).resolve()
    output_root = Path(args.output_dir).resolve()

    if not moving_image.exists():
        raise FileNotFoundError(f"Moving image not found: {moving_image}")
    if not fixed_image.exists():
        raise FileNotFoundError(f"Fixed image not found: {fixed_image}")

    if output_root.exists() and args.overwrite:
        reset_dir(output_root)
    else:
        ensure_dir(output_root)

    base_case_name = args.case_name
    if not base_case_name:
        base_case_name = f"{strip_supported_suffix(moving_image)}_to_{strip_supported_suffix(fixed_image)}"
    case_name = sanitize_name(base_case_name)

    work_root = output_root / "work"
    staging_images = stage_single_pair(
        moving_image=moving_image,
        fixed_image=fixed_image,
        case_name=case_name,
        output_dir=work_root / "staging",
    )

    pair_npz, pair_mat = run_step1_single_pair(
        project_root=project_root,
        images_dir=staging_images,
        step1_output_dir=work_root / "step1_pairs",
        moving_modality=args.moving_modality,
        fixed_modality=args.fixed_modality,
        strategy=args.strategy,
        num_samples=args.num_samples,
        processing_mode=args.processing_mode,
        seed=args.seed,
    )

    registration_dir = output_root / "registration_results" / METHOD_NAME / "data"
    result_path = registration_dir / f"{sanitize_name(pair_npz.stem)}_result.mat"
    run_matlab_registration(
        project_root=project_root,
        pair_mat=pair_mat,
        result_path=result_path,
        matlab_bin=args.matlab_bin,
    )

    visualization_dir = output_root / "visualization" / METHOD_NAME / sanitize_name(pair_npz.stem)
    export_registration_outputs(
        pair_npz=pair_npz,
        result_path=result_path,
        output_dir=visualization_dir,
        slice_mode=args.slice_mode,
        canvas_pad_mm=args.canvas_pad_mm,
        crop_pad_ratio=args.crop_pad_ratio,
        crop_min_pad=args.crop_min_pad,
    )

    print("\n============================================================")
    print("One-click registration finished")
    print("============================================================")
    print(f"Case                 : {case_name}")
    print(f"Method               : {METHOD_NAME}")
    print(f"Step1 pair npz       : {pair_npz}")
    print(f"MATLAB result        : {result_path}")
    print(f"Visualization output : {visualization_dir}")
    print(f"Registered 3D volume : {visualization_dir / 'volumes' / 'moving_registered_self.nii.gz'}")
    print(f"Registered slices    : {visualization_dir / 'slices' / 'moving_registered_self_triptych.png'}")
    print("============================================================")


if __name__ == "__main__":
    main()
