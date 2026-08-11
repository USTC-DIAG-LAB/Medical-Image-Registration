import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def safe_link(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        shutil.copy2(src, dst)


def stage_image_file(src: Path, dst: Path) -> None:
    if src.suffix.lower() == ".mhd":
        stage_metaimage_file(src, dst)
    else:
        safe_link(src, dst)


def stage_metaimage_file(src_mhd: Path, dst_mhd: Path) -> None:
    ensure_dir(dst_mhd.parent)
    text = src_mhd.read_text(encoding="utf-8", errors="ignore")

    match = re.search(r"^ElementDataFile\s*=\s*(.+?)\s*$", text, flags=re.MULTILINE)
    if not match:
        shutil.copy2(src_mhd, dst_mhd)
        return

    data_ref = match.group(1).strip()
    src_data = (src_mhd.parent / data_ref).resolve()
    if not src_data.exists():
        raise FileNotFoundError(f"MetaImage data file not found for {src_mhd}: {src_data}")

    data_target_name = f"{dst_mhd.stem}{src_data.suffix}"
    dst_data = dst_mhd.parent / data_target_name
    shutil.copy2(src_data, dst_data)

    rewritten = re.sub(
        r"^ElementDataFile\s*=\s*.+?\s*$",
        f"ElementDataFile = {data_target_name}",
        text,
        flags=re.MULTILINE,
    )
    dst_mhd.write_text(rewritten, encoding="utf-8")


def find_child_dir_case_insensitive(parent: Path, *candidate_names: str) -> Optional[Path]:
    if not parent.exists():
        return None

    lookup = {child.name.lower(): child for child in parent.iterdir() if child.is_dir()}
    for candidate_name in candidate_names:
        matched = lookup.get(candidate_name.lower())
        if matched is not None:
            return matched
    return None


def run_step1(
    project_root: Path,
    images_dir: Path,
    labels_dir: Optional[Path],
    output_dir: Path,
    fixed_modality: Optional[str],
    moving_modality: Optional[str],
    channel_map: Sequence[str],
    strategies: str,
    num_samples: int,
    processing_mode: str,
) -> None:
    cmd = [
        sys.executable,
        str(project_root / "step1.py"),
        "--images-dir",
        str(images_dir),
        "--output-dir",
        str(output_dir),
        "--strategies",
        strategies,
        "--num-samples",
        str(num_samples),
        "--processing-mode",
        processing_mode,
    ]

    if labels_dir is not None and labels_dir.exists():
        cmd.extend(["--labels-dir", str(labels_dir)])

    if fixed_modality:
        cmd.extend(["--fixed-modality", fixed_modality])
    if moving_modality:
        cmd.extend(["--moving-modality", moving_modality])
    if channel_map:
        cmd.append("--channel-map")
        cmd.extend(channel_map)

    print("\n[RUN]", " ".join(cmd))
    subprocess.run(cmd, cwd=project_root, check=True)


def choose_preferred_rire_mri(patient_dir: Path) -> List[Tuple[str, Path]]:
    priorities = [
        ("MRI_T1", ["mr_t1_rectified", "mr_t1"]),
        ("MRI_T2", ["mr_t2_rectified", "mr_t2"]),
        ("MRI_PD", ["mr_pd_rectified", "mr_pd"]),
    ]
    results: List[Tuple[str, Path]] = []
    for modality_name, candidate_dirs in priorities:
        image_dir = find_child_dir_case_insensitive(patient_dir, *candidate_dirs)
        if image_dir is None:
            continue
        mhd_files = sorted(image_dir.glob("*.mhd"))
        if mhd_files:
            results.append((modality_name, mhd_files[0]))
    return results


def stage_rire_ct_mri_pairs(project_root: Path, staging_root: Path) -> Optional[Path]:
    rire_root = project_root / "archive" / "RIRE" / "RIRE"
    ct_mri_images = staging_root / "rire_ct_mri" / "imagesTr"
    reset_dir(ct_mri_images)

    ct_mri_count = 0

    for patient_dir in sorted(rire_root.glob("patient_*")):
        ct_dir = find_child_dir_case_insensitive(patient_dir, "ct")
        if ct_dir is None:
            continue
        ct_files = sorted(ct_dir.glob("*.mhd"))
        if not ct_files:
            continue
        ct_file = ct_files[0]
        patient_id = patient_dir.name

        for modality_name, mri_file in choose_preferred_rire_mri(patient_dir):
            variant_name = modality_name.replace("MRI_", "RIRE")
            case_id = f"{patient_id}_{variant_name}"
            stage_image_file(ct_file, ct_mri_images / f"{case_id}_0000.mhd")
            stage_image_file(mri_file, ct_mri_images / f"{case_id}_0001.mhd")
            ct_mri_count += 1

    print(f"[INFO] RIRE staged CT-MRI pair count: {ct_mri_count}")
    return ct_mri_images if ct_mri_count > 0 else None


def stage_lungct_tre_cases(project_root: Path, staging_root: Path) -> Tuple[Path, Path]:
    lung_root = project_root / "LungCT" / "LungCT"
    images_stage = staging_root / "lungct_tre" / "imagesTr"
    labels_stage = staging_root / "lungct_tre" / "labelsTr"
    reset_dir(images_stage)
    reset_dir(labels_stage)

    tre_case_ids = set()
    landmarks_dir = lung_root / "landmarksTr"
    if landmarks_dir.exists():
        for csv_file in landmarks_dir.glob("*.csv"):
            parts = csv_file.stem.split("_")
            if len(parts) >= 2:
                tre_case_ids.add("_".join(parts[:2]))

    copied = 0
    image_dir = lung_root / "imagesTr"
    mask_dir = lung_root / "masksTr"
    if image_dir.exists() and mask_dir.exists():
        for case_id in sorted(tre_case_ids):
            for channel in ("0000", "0001"):
                image_name = f"{case_id}_{channel}.nii.gz"
                image_file = image_dir / image_name
                mask_file = mask_dir / image_name
                if not image_file.exists() or not mask_file.exists():
                    continue
                safe_link(image_file, images_stage / image_name)
                safe_link(mask_file, labels_stage / image_name)
                copied += 1

    print(f"[INFO] LungCT staged image count with landmark-based TRE annotations: {copied}")
    return images_stage, labels_stage


def stage_abdomenmrct_labeled_cases(project_root: Path, staging_root: Path) -> Tuple[Path, Path]:
    abdomen_root = project_root / "AbdomenMRCT" / "AbdomenMRCT"
    source_images = abdomen_root / "imagesTr"
    source_labels = abdomen_root / "labelsTr"
    images_stage = staging_root / "abdomenmrct_labeled" / "imagesTr"
    labels_stage = staging_root / "abdomenmrct_labeled" / "labelsTr"
    reset_dir(images_stage)
    reset_dir(labels_stage)

    image_names = {path.name for path in source_images.glob("*.nii.gz")}
    label_names = {path.name for path in source_labels.glob("*.nii.gz")}
    case_ids = sorted({"_".join(name.split("_")[:2]) for name in image_names & label_names})

    copied = 0
    kept_cases = 0
    for case_id in case_ids:
        required_names = {f"{case_id}_0000.nii.gz", f"{case_id}_0001.nii.gz"}
        if not required_names.issubset(image_names) or not required_names.issubset(label_names):
            continue
        for image_name in sorted(required_names):
            safe_link(source_images / image_name, images_stage / image_name)
            safe_link(source_labels / image_name, labels_stage / image_name)
            copied += 1
        kept_cases += 1

    print(f"[INFO] AbdomenMRCT staged labeled train case count: {kept_cases}")
    print(f"[INFO] AbdomenMRCT staged image count: {copied}")
    return images_stage, labels_stage


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-build experiment inputs with step1.py")
    parser.add_argument("--project-root", default=".", help="Project root containing step1.py and datasets.")
    parser.add_argument("--output-root", default="experiment_batches", help="Root folder for staged data and outputs.")
    parser.add_argument("--strategies", default="mixed", help="Sampling strategies passed to step1.py")
    parser.add_argument("--num-samples", type=int, default=10000, help="Sampling budget passed to step1.py")
    parser.add_argument(
        "--processing-mode",
        choices=("fast", "smart"),
        default="fast",
        help="Processing mode passed to step1.py",
    )
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    output_root = (project_root / args.output_root).resolve()
    staging_root = output_root / "staging"
    processed_root = output_root / "processed"
    ensure_dir(staging_root)
    ensure_dir(processed_root)

    # 1. RIRE: only CT-MRI pairs.
    rire_images_stage = stage_rire_ct_mri_pairs(project_root, staging_root)
    if rire_images_stage is not None:
        run_step1(
            project_root=project_root,
            images_dir=rire_images_stage,
            labels_dir=None,
            output_dir=processed_root / "RIRE_CT_MRI",
            fixed_modality="MRI",
            moving_modality="CT",
            channel_map=["0000=CT", "0001=MRI"],
            strategies=args.strategies,
            num_samples=args.num_samples,
            processing_mode=args.processing_mode,
        )
    else:
        print("[WARN] No RIRE CT-MRI pairs were staged.")

    # 2. AbdomenMRCT: only fully labeled CT-MRI training cases.
    abdomen_images_stage, abdomen_labels_stage = stage_abdomenmrct_labeled_cases(project_root, staging_root)
    run_step1(
        project_root=project_root,
        images_dir=abdomen_images_stage,
        labels_dir=abdomen_labels_stage,
        output_dir=processed_root / "AbdomenMRCT_CT_MRI",
        fixed_modality="MRI",
        moving_modality="CT",
        channel_map=["0000=CT", "0001=MRI"],
        strategies=args.strategies,
        num_samples=args.num_samples,
        processing_mode=args.processing_mode,
    )

    # 3. LungCT: only landmark-based TRE training cases.
    lung_images_stage, lung_labels_stage = stage_lungct_tre_cases(project_root, staging_root)
    run_step1(
        project_root=project_root,
        images_dir=lung_images_stage,
        labels_dir=lung_labels_stage,
        output_dir=processed_root / "LungCT_CT_CT_TRE",
        fixed_modality=None,
        moving_modality=None,
        channel_map=["0000=CT", "0001=CT"],
        strategies=args.strategies,
        num_samples=args.num_samples,
        processing_mode=args.processing_mode,
    )

    print("\n[DONE] Batch preparation finished.")
    print(f"[DONE] Outputs saved under: {processed_root}")


if __name__ == "__main__":
    main()
