# FOTMIR: A Fast Optimal Transport-Based Tool for Multimodal Medical Image Registration

FOTMIR is an open-source tool for rigid multimodal medical image registration. It converts moving and fixed medical images into three-dimensional point sets, compresses those sets with weighted `k`-center coresets, and estimates a rigid transformation using entropy-regularized optimal transport.

The repository provides an end-to-end workflow for:

1. Loading moving and fixed medical images
2. Extracting and sampling three-dimensional points
3. Constructing weighted `k`-center coresets
4. Computing optimal-transport-based soft correspondences
5. Estimating and applying a rigid transformation
6. Exporting registered volumes, slice visualizations, and a JSON summary

> **Name migration:** The tool was previously documented as **FastMedOT**. Its current public name is **FOTMIR**, derived from **Fast Optimal Transport-based tool for multimodal Medical Image Registration**. The legacy identifier `modified` is still used internally by some scripts and output directories for compatibility; it refers to the FOTMIR method and is not a separate algorithm.

## Key Features

- Multimodal rigid registration, including CT-MRI workflows
- Weighted `k`-center coreset compression for reduced transport cost
- Entropy-regularized optimal transport with Sinkhorn optimization
- Rigid rotation and translation estimation from soft correspondences
- One-click command-line execution
- A graphical interface for non-technical users
- Registered 3D NIfTI output and axial, coronal, and sagittal visualizations
- Local processing with no external service required

## Method Overview

Given a moving image and a fixed image, FOTMIR performs the following steps:

1. **Image preprocessing and point extraction**  
   The input images are loaded with SimpleITK. Foreground structures are identified and converted into three-dimensional point-set representations.

2. **Weighted coreset construction**  
   Each point set is compressed with weighted `k`-center sampling. Every selected center carries the total mass of its assigned cluster, retaining information from the original point distribution in a smaller representation.

3. **Optimal transport matching**  
   The MATLAB backend computes an entropy-regularized transport plan between the moving and fixed coresets using Sinkhorn optimization.

4. **Rigid transform estimation**  
   The soft correspondences induced by the transport plan are used to estimate a three-dimensional rotation and translation.

5. **Reconstruction and export**  
   The estimated transform is mapped back to physical image coordinates and applied to the original moving image. FOTMIR then saves the registered volume, fixed reference volume, representative slices, and transformation metadata.

## Main Files

### User Entry Points

- `doctor_registration_gui.py` - graphical interface for interactive registration
- `one_click_registration.py` - end-to-end command-line and backend entry point

### Python Modules

- `step1.py` - image loading, foreground preprocessing, point extraction, and pair preparation
- `visualize_registration_result.py` - transform conversion, image reconstruction, and visualization
- `batch_prepare_experiments.py` - file-staging utilities reused by the one-click workflow

### MATLAB Registration Core

- `main.m` - FOTMIR rigid registration entry point
- `KCenter.m` - weighted `k`-center coreset construction
- `SinkhornInit.m` - initialization and rigid matching routine
- `Sinkhorn.m` - entropy-regularized optimal transport solver
- `distance.m` - point-set distance computation
- `Transport.m` - lower-level transport optimization routine

## Requirements

### Python

- Python 3
- NumPy
- SciPy
- SimpleITK
- Matplotlib
- Tkinter, only when using the graphical interface

Install the required Python packages with:

```bash
python -m pip install numpy scipy SimpleITK matplotlib
```

Tkinter is commonly included with Python. If it is unavailable, install the Tk package supplied by your operating system or Python distribution.

### MATLAB

MATLAB is required for the current registration backend. Confirm that it can be launched from a terminal:

```bash
matlab
```

If MATLAB is not available on the system path, pass the executable path with `--matlab-bin` or enter it in the graphical interface.

## Supported Inputs

FOTMIR accepts one moving image and one fixed image. The preprocessing code recognizes the following medical-image extensions:

- `.nii.gz`, `.nii`
- `.mha`, `.mhd`
- `.nrrd`
- `.img`, `.hdr`
- `.mnc`, `.mnc2`
- `.dcm`, `.ima`

The graphical interface provides the following modality labels:

- CT
- MRI
- PET
- SPECT
- CBCT
- US
- XRAY
- OCT
- MG

The **moving image** is transformed into the coordinate system of the **fixed image**.

## Quick Start

Run all commands from the directory containing this README and the FOTMIR source files.

### Graphical Interface

Start the interface with:

```bash
python doctor_registration_gui.py
```

Then:

1. Select the moving image.
2. Select the fixed image.
3. Choose the modality of each image.
4. Select an output directory.
5. Optionally enter a case name or a custom MATLAB executable.
6. Click **Start Registration**.

### Command-Line Interface

Run an example CT-to-MRI registration with:

```bash
python one_click_registration.py \
  --moving-image /path/to/moving_ct.nii.gz \
  --fixed-image /path/to/fixed_mri.nii.gz \
  --moving-modality CT \
  --fixed-modality MRI \
  --output-dir /path/to/output_case \
  --overwrite
```

Use `--matlab-bin` when MATLAB is not available under the default `matlab` command:

```bash
python one_click_registration.py \
  --moving-image /path/to/moving.nii.gz \
  --fixed-image /path/to/fixed.nii.gz \
  --moving-modality CT \
  --fixed-modality MRI \
  --output-dir /path/to/output_case \
  --matlab-bin /absolute/path/to/matlab
```

### Useful Command-Line Options

| Option | Default | Description |
| --- | --- | --- |
| `--case-name` | generated from filenames | Custom case identifier |
| `--strategy` | `mixed` | Point sampling strategy: `mixed`, `boundary`, or `all` |
| `--num-samples` | `10000` | Approximate sample count when the strategy is not `all` |
| `--processing-mode` | `fast` | Foreground extraction mode: `fast` or `smart` |
| `--seed` | `20260722` | Random seed used during point sampling |
| `--matlab-bin` | `matlab` | MATLAB command or absolute executable path |
| `--slice-mode` | `foreground_center` | Representative slice selection: `foreground_center` or `image_center` |
| `--overwrite` | disabled | Remove an existing output directory before running |

Display the complete command-line help with:

```bash
python one_click_registration.py --help
```

## Output Structure

A successful one-click run creates an output structure similar to:

```text
output_case/
|-- registration_results/
|   `-- modified/
|       `-- data/
|           `-- <case>_result.mat
|-- visualization/
|   `-- modified/
|       `-- <case>/
|           |-- volumes/
|           |   |-- fixed_reference.nii.gz
|           |   `-- moving_registered_self.nii.gz
|           |-- slices/
|           |   |-- fixed_reference_self_triptych.png
|           |   |-- moving_registered_self_triptych.png
|           |   `-- individual axial, coronal, and sagittal PNG files
|           `-- summary.json
`-- work/
    |-- staging/
    `-- step1_pairs/
```

Here, `modified` is the legacy internal method identifier for FOTMIR. It remains in the output paths so that existing experiment and visualization scripts continue to work.

The main user-facing outputs are:

- `moving_registered_self.nii.gz` - registered moving image
- `fixed_reference.nii.gz` - processed fixed reference image
- `moving_registered_self_triptych.png` - registered moving image in three orthogonal views
- `fixed_reference_self_triptych.png` - fixed reference image in three orthogonal views
- `summary.json` - runtime, transform, direction selection, point-based error values, and output paths

The raw MATLAB result file stores the estimated transformation and registration diagnostics.

## Current Scope

The current FOTMIR release supports:

- Three-dimensional rigid registration
- Single moving/fixed image-pair execution
- Multimodal medical-image preprocessing
- Coreset-based optimal transport alignment
- Local GUI and command-line workflows

It does not currently provide deformable registration. Batch experiments, baseline comparisons, benchmark evaluation, and paper-figure generation are available elsewhere in the full research codebase and are not required for the minimal software workflow.

## Citation

If you use FOTMIR in academic work, please cite:

> Feng Bian, Lin Chen, and Hu Ding. **FOTMIR: A Fast Optimal Transport-Based Tool for Multimodal Medical Image Registration.** *Frontiers of Computer Science*, 2026.

Please update the volume, issue, page/article number, and DOI after the final bibliographic record becomes available.

## License

Original FOTMIR source code and documentation are released under the [Apache License 2.0](LICENSE). Copyright and attribution information is provided in [NOTICE](NOTICE). Third-party code, datasets, templates, and external dependencies remain subject to their own terms; see [Third-Party Notices](THIRD_PARTY_NOTICES.md) for details.
