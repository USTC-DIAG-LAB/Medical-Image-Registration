# FastMedOT: Medical Image Registration Software Delivery Package

This folder contains the **minimum set of code required to run FastMedOT**, an optimal transport-based tool for rigid medical image registration, with a primary focus on multimodal medical images.

The purpose of this delivery package is not to preserve the entire experimental workflow, but to retain the components required for the deployable FastMedOT workflow:

1. Moving and fixed medical image input
2. Point extraction and preprocessing
3. Weighted `k`-center coreset compression
4. Optimal transport-based rigid registration
5. Registered 3D image reconstruction and slice visualization
6. Graphical and command-line interfaces

> **Naming note:** `modified` is retained only as an internal implementation name in the current codebase. In documentation and user-facing descriptions, the registration tool should be referred to as **FastMedOT**.

---

## 1. What This Folder Contains

### Python Entry Points

- `doctor_registration_gui.py`  
  Graphical user interface entry point for FastMedOT. It is recommended for doctors and general users who prefer an interactive workflow.

- `one_click_registration.py`  
  One-click command-line/backend entry point that connects the complete FastMedOT registration workflow.

### Python Functional Modules

- `step1.py`  
  Loads the original medical images and performs preprocessing and 3D point extraction for registration.

- `visualize_registration_result.py`  
  Reconstructs and visualizes registration results, including registered 3D volumes and slice outputs.

- `batch_prepare_experiments.py`  
  This delivery package retains only the file-organization functionality reused by the one-click FastMedOT workflow.

### MATLAB Registration Core

- `main.m`  
  The main MATLAB registration function used by FastMedOT. It performs the core rigid registration workflow based on weighted coreset compression, entropy-regularized optimal transport, and rigid transform estimation. In the current codebase, this implementation is internally referred to as `modified`.

- `KCenter.m`  
  Weighted `k`-center point-set compression module used to construct compact coresets while representing the contribution of the original point clusters.

- `SinkhornInit.m`  
  Initialization module used by the rigid matching backend.

- `Sinkhorn.m`  
  Computes the entropy-regularized optimal transport/Sinkhorn solution and transport-related matrices.

- `distance.m`  
  Computes point-set distance matrices.

- `Transport.m`  
  Low-level function used by the Sinkhorn/optimal transport solver.

---

## 2. FastMedOT Registration Workflow

FastMedOT processes a moving image and a fixed image through the following stages:

1. **Image loading and point extraction**  
   The moving and fixed medical images are loaded and converted into three-dimensional point-set representations.

2. **Weighted `k`-center coreset construction**  
   Each point set is compressed into a smaller weighted coreset. Each center represents its assigned cluster so that the contribution of the original points is retained in a compact representation.

3. **Optimal transport alignment**  
   The MATLAB backend computes an entropy-regularized optimal transport plan between the weighted coresets using Sinkhorn iterations.

4. **Rigid transform estimation**  
   Soft correspondences from the transport plan are used to estimate the rigid rotation and translation.

5. **Transformation and reconstruction**  
   The estimated transform is applied to the original moving data, and the registered image is reconstructed.

6. **Result visualization and export**  
   FastMedOT generates registered 3D outputs, slice visualizations, and a `summary.json` record.

This coreset-based design reduces the size of the pairwise transport problem while preserving the contribution of the original point clusters.

---

## 3. Why Only These Files Are Included

The deployable FastMedOT package currently focuses on:

- Single moving/fixed image-pair input
- Automatic medical image preprocessing
- Weighted `k`-center coreset compression
- Optimal transport-based rigid alignment
- Automatic reconstruction of registered results
- Automatic generation of `self` slice visualizations
- Local execution through graphical or command-line interfaces

Therefore, the following types of research and evaluation files are **not included** in this delivery folder:

- Batch experiment scripts
- Comparative algorithm scripts
- Error-analysis scripts
- Experimental MATLAB versions
- Paper reproduction experiment code
- Additional scripts for dataset organization and benchmark evaluation

The benefits of this packaging approach are:

- A cleaner FastMedOT software structure
- Better suitability for software delivery and demonstration
- Lower risk of accidentally using experimental scripts
- Easier future packaging as a standalone GUI or desktop application

---

## 4. Recommended Startup Methods

### For Doctors or General Users

Run:

```bash
python doctor_registration_gui.py
```

Then, in the interface:

1. Select the `Moving` image
2. Select the `Fixed` image
3. Select the modality of each image
4. Select the output folder
5. Click **Start Registration**

### For Technical Users

For command-line execution, run:

```bash
python one_click_registration.py \
  --moving-image /path/to/moving.nii.gz \
  --fixed-image /path/to/fixed.nii.gz \
  --moving-modality CT \
  --fixed-modality MRI \
  --output-dir /path/to/output_case \
  --overwrite
```

---

## 5. Output Results

FastMedOT outputs:

- The registered 3D moving image
- The corresponding fixed reference 3D image
- `self` slice visualizations of the registered moving image
- `self` slice visualizations of the fixed image
- A result summary in `summary.json`

All registration processing is performed locally.

---

## 6. Dependencies

### Python

At minimum, the following Python packages are required:

- `numpy`
- `scipy`
- `SimpleITK`
- `matplotlib`

### MATLAB

MATLAB must be installed and callable from the command line:

```bash
matlab
```

If MATLAB is not available through the default system path, specify its executable using `--matlab-bin` in the command-line entry point.

---

## 7. Current Software Scope

The current FastMedOT delivery package is suitable for:

- Rigid medical image registration
- Multimodal medical image registration, including CT-MRI workflows
- Inspection of registered 3D volumes and slice visualizations
- Clinical or research demonstrations
- Future packaging as a standalone GUI or desktop application

The current release supports **rigid registration only** and requires MATLAB for the registration backend.

If you later need to perform:

- Batch experiments
- Multi-algorithm comparisons
- Benchmark evaluation and error analysis
- Paper reproduction experiments

it is recommended to use the complete project codebase rather than relying only on this minimal FastMedOT delivery package.
