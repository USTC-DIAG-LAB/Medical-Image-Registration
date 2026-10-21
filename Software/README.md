# FOTMIR software

Version 1.0.0. Keep this directory inside the release package so that the shared `VERSION` file remains available.

## Installation

From the package root:

```sh
python -m pip install -r Software/requirements.txt
python Software/doctor_registration_gui.py
```

The English desktop interface uses Tkinter. Install a Python distribution with Tcl/Tk support. The numerical dependencies are NumPy 2.0.2, SciPy 1.13.1, SimpleITK 2.5.3 and Matplotlib 3.9.2.

## Inputs and settings

Select scalar 3D moving and fixed image files. NIfTI (`.nii`, `.nii.gz`), MetaImage (`.mha`, `.mhd`) and other formats supported by SimpleITK can be read. Keep the voxel sidecar referenced by an `.mhd` file beside its header. Convert a DICOM series into a 3D volume before selecting it.

Choose the corresponding image modalities and an empty output directory. Defaults are 10,000 points per image, one numerical thread, seed 20260722, transport parameter `tau=0.05`, coreset ratio `0.05`, and extraction profile `structure`.

## Command-line registration

Run the included real RIRE head CT-MRI example (`patient_001`, CT to rectified PD-weighted MRI) from the package root:

```sh
python Software/one_click_registration.py \
  --moving-image Software/examples/data/moving.nii.gz \
  --fixed-image Software/examples/data/fixed.nii.gz \
  --moving-modality CT --fixed-modality MRI \
  --num-samples 10000 --threads 1 \
  --output-dir run_outputs/rire_example
```

Dataset source and preparation are documented in [the example README](examples/README.md).

For another image pair, replace the two paths and modalities. Use `--num-samples 10000`, `--seed`, `--tau` and `--ratio` to set registration parameters. `register.py` is an alternative entry point for the same application.

The launcher sets BLAS, OpenMP and ITK thread environment variables before starting its numerical worker. The application uses this process-based thread control.

## Outputs

| File | Contents |
| --- | --- |
| `moving_registered.nii.gz` | Moving volume resampled onto the fixed image grid |
| `fixed_reference.nii.gz` | Fixed reference volume |
| `fixed_z.png`, `registered_z.png` | Separate grayscale central slices along the native z axis |
| `fixed_y.png`, `registered_y.png` | Separate grayscale central slices along the native y axis |
| `fixed_x.png`, `registered_x.png` | Separate grayscale central slices along the native x axis |
| `registration.npz` | Core registration output and normalized transform |
| `points.npz` | Point sets and normalization metadata |
| `summary.json` | Parameters, physical transform, extraction information and timings |

The application exports six independent grayscale PNGs: a fixed image and a registered moving image for each of the three directions. Each pair uses the same central slice on the fixed grid. Image proportions follow the in-plane voxel spacing. The names x, y and z identify the native image-grid axes; anatomical directions depend on the image orientation.

Display limits use the first and 99th intensity percentiles of each volume independently, with the same limits across its three views. For the registered image, only source-covered voxels contribute to these limits; areas outside source coverage appear black. These display settings leave the saved image values and registration transform unchanged.

`TM_physical` maps moving points to fixed physical coordinates in millimetres. Image resampling uses its inverse. `elapsed_registration_s` measures the core registration stage; `elapsed_pipeline_s` additionally covers the worker's image processing and output generation. Process startup and imports precede these timers.

## Point-cloud API and batch entry point

`fotmir_core/main.py` accepts NPZ files containing `A` and `B`, each a `3 x N` point array. To process one file or a directory of point-cloud files:

```sh
python Software/fotmir_core/run_registration.py \
  --input /path/to/point_inputs \
  --output-dir run_outputs/point_registration \
  --seed 20260722 --threads 1
```

The batch entry point writes per-case results and `summary.csv`, including the package version. Python callers may import `register_points` or `main` from `fotmir_core.main`.

## Summary fields

| Field | Meaning |
| --- | --- |
| `version`, `implementation` | Package version and Python implementation identifier |
| `thread_policy`, `thread_environment`, `threads` | Worker launch policy, numerical thread environment and requested count |
| `extraction_profile`, `extraction_diagnostics`, `labels_used` | Point extraction recipe, diagnostics and annotation-use flag |
| `points`, `seed` | Actual moving/fixed point counts and random seed |
| `tau`, `ratio` | Partial transport mass parameter and coreset ratio |
| `TM_physical`, `transform_convention` | Moving-to-fixed physical transform and its convention |
| `elapsed_registration_s`, `elapsed_pipeline_s` | Core and image-worker wall times in seconds |
| `grayscale_views` | PNG filenames, native axes, slice indices, fixed-grid direction, physical dimensions, pixel dimensions and display limits |
| `outputs` | Names of the generated registration and visualization files |

The `version` field identifies FOTMIR 1.0.0. Extraction-policy identifiers and dependency versions identify their own components and retain their independent names.
