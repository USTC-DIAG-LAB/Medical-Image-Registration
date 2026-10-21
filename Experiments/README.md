# Registration experiments

FOTMIR package version: 1.0.0. Run `python Experiments/run_experiments.py --version` to display it.

The present evaluation is restricted to rigid registration. Although the Learn2Reg AbdomenMRCT and LungCT benchmarks primarily target deformable registration, all methods in this evaluation, including ANTs, are configured for rigid alignment. LungCT evaluates respiratory CT-to-CT registration.

## Dataset sources and case selection

| Dataset | Source | Pairs | Sampling region | Primary surface evaluation |
| --- | --- | ---: | --- | --- |
| RIRE | [RIRE downloads](https://rire.insight-journal.org/download), [data format](https://rire.insight-journal.org/data_format) | 45 | Image-inferred foreground | Warped moving foreground versus fixed foreground |
| AbdomenMRCT | [Learn2Reg datasets](https://learn2reg.grand-challenge.org/Datasets/), [Learn2Reg 2021](https://learn2reg.grand-challenge.org/Learn2Reg2021/) | 8 | Organ annotations | Union of positive organ labels |
| LungCT | [Learn2Reg datasets](https://learn2reg.grand-challenge.org/Datasets/), [Learn2Reg 2021](https://learn2reg.grand-challenge.org/Learn2Reg2021/) | 30 | Supplied automatic lung masks | Warped moving lung mask versus fixed lung mask |

Follow the data providers' download and citation terms. `manifests/cases83.csv` gives the case list. `manifests/main83.json` provides image, label, mask and landmark paths, point-array identifiers, normalization, registration seeds and initial center indices for each case.

RIRE pairs use CT as moving and MRI as fixed. AbdomenMRCT cases 0001-0008 use moving MRI (`0000`) and fixed CT (`0001`); the manifest's modality fields specify the direction. LungCT is a CT-to-CT respiratory registration benchmark. The same eight abdominal pairs are used for accuracy and runtime summaries. Three lung pairs have landmark measurements.

## Data layout

Arrange data in a separate directory such as:

```text
experiment_data/
  prepared_inputs/
    RIRE_CT_MRI/<case>.npz
    AbdomenMRCT_CT_MRI/<case>.npz
    LungCT_CT_CT/<case>.npz
  external_data/
    RIRE_preprocessed/imagesTr/...
    AbdomenMRCT/imagesTr/...
    AbdomenMRCT/labelsTr/...
    AbdomenMRCT/masksTr/...
    LungCT/images/...
    LungCT/masksTr/...
    LungCT/landmarksTr/...
```

Use the exact relative filenames specified by `main83.json`. Preserve origin, spacing and direction when converting image formats; place MetaImage voxel sidecars alongside their headers. Landmark CSV rows store matched continuous voxel coordinates in `x,y,z` order.

To arrange retained experiment inputs in this layout:

```sh
python Experiments/prepare_data.py \
  --source-prepared /path/to/archived/point_inputs \
  --source-data /path/to/mapped/image_data \
  --output-dir /path/to/experiment_data
```

The source point-input directory follows the `prepared_input` paths in the manifest. `--source-data` follows the image paths shown above; it may be omitted when image paths embedded in the retained NPZ files already resolve on the current machine.

## Point-set generation

The region sampler is `preprocessing/benchmark_sampling.py`. Each image requests 10,000 distinct voxel locations: 80% on region boundaries and 20% in interiors. Boundary and interior quotas are allocated in proportion to their available voxel counts. Capacity limits redistribute spare quota. Each positive organ label is sampled separately; selected coordinates are combined into a point cloud. Lung sampling uses the positive mask regions. The organ labels and lung masks also define the corresponding evaluation regions described in the Results README.

RIRE foreground generation thresholds CT above `max(-700, percentile_5)` and applies Otsu thresholding to MRI after clipping the first and 99th percentiles. One closing and one opening with 3D face connectivity, hole filling and largest-component selection follow thresholding. Empty threshold results use an 80th-percentile fallback. Boundary extraction uses a one-voxel erosion with 3D face connectivity. The image geometry maps sampled voxel indices into physical coordinates.

For an abdominal pair:

```sh
python Experiments/preprocessing/benchmark_sampling.py \
  --moving-image /path/to/moving_mri.nii.gz \
  --fixed-image /path/to/fixed_ct.nii.gz \
  --moving-region /path/to/moving_organs.nii.gz \
  --fixed-region /path/to/fixed_organs.nii.gz \
  --moving-modality MRI --fixed-modality CT \
  --mode organ --num-samples 10000 --seed 20261010 \
  --output /path/to/prepared_pair.npz
```

For LungCT, select `--mode mask`, supply the two lung masks and use CT for both modalities. For RIRE, select `--mode intensity`, supply CT/MRI images and omit the region arguments. The generation command uses a separate same-seed random generator for each image. The default sampling seed of 20261010 belongs to this generation recipe; the `seed` fields in `main83.json` belong to registration initialization.

For the recorded 83 cases, select the retained point arrays identified by `main83.json`. A newly sampled pair has its own arrays and sampling metadata. To run newly generated pairs, create a case manifest with their relative paths, array metadata and chosen registration initialization.

The prepared NPZ schema contains `X` and `Y` (`N x 3` moving and fixed coordinates), `center_Y`, `scale`, image paths, modalities and optional annotation paths. The sampler also writes `A=X.T`, `B=Y.T`, the sampling seed and requested point count. Normalization subtracts the fixed-point mean `c` from both clouds and divides both by the largest fixed-point radius `s`:

```text
X = (moving_points_mm - c) / s
Y = (fixed_points_mm  - c) / s
```

Manifest `X_sha256`, `Y_sha256`, `center_Y_sha256` and `scale_sha256` values are SHA-256 digests of the corresponding NumPy array bytes. `initial_indices` contains two one-based starting indices. Set `prepared_input`, `dataset`, `case`, `group`, image/reference paths, modalities, `seed`, `initial_indices`, `n_points`, `center_Y` and `scale` for each new entry; use an existing entry as the schema example. `input_sha256` records the historical NPZ file identity.

## Installation and execution

The recorded environment used Python 3.9.23 on Linux. Install the experiment dependencies and run from the package root:

```sh
python -m pip install -r Experiments/requirements.txt
python Experiments/run_experiments.py \
  --prepared-root /path/to/experiment_data/prepared_inputs \
  --data-root /path/to/experiment_data/external_data \
  --output-dir run_outputs/benchmark
```

Select methods using `--methods python icp fgr ants`; `python` identifies FOTMIR. Select individual case identifiers with `--cases`, or a different case manifest with `--manifest`. Use a new or empty output directory for each run.

`run_experiments.py` runs all selected registrations and then computes their metrics. New run configuration, summary and metric records include the package version. It writes per-case transformation NPZ files, ANTs transformation files, `metrics.csv`, `metrics.json`, `run_configuration.json` and `run_summary.json`. JSON records retain physical transformations; the CSV contains scalar results. Metric definitions are in [Results/README.md](../Results/README.md).

## Method configuration

`configs/benchmark.json` records the method settings. The run entry point reads runtime arguments from this file; Sinkhorn and outer-loop constants documented there are implemented in `fotmir_core/Transport.py` and `fotmir_core/SinkhornInit.py`.

| Method | Settings |
| --- | --- |
| FOTMIR | `tau=0.05`; coreset ratio `0.05`; `k=max(50, floor(N*0.05+0.5))`, giving 500 centers for 10,000 points; streaming coreset distances; one initialization using manifest center indices; zero refinement iterations |
| Transport | `lambda=10/median(augmented_squared_cost)`; entropy coefficient `epsilon=1/lambda`; marginal-difference tolerance `0.005`; solver counter limit `5000`; the counter also increments after convergence evaluations |
| Outer iterations | Stop when `abs(previous_cost-current_cost) <= 1e-6*current_cost`; at most 51 orthogonal updates |
| ICP | Identity starting transform; nearest-neighbor correspondences; keep distances below twice their mean; maximum 100 iterations; stop when `max(norm(R-I), norm(t)) < 1e-6` |
| FGR | Open3D 0.18.0; seed 20260716; voxel size `0.05`; normal radius `0.10`, maximum 30 neighbors; FPFH radius `0.20`, maximum 100 neighbors; maximum correspondence distance `0.05`; other FGR options use Open3D 0.18.0 defaults |
| ANTs | ANTsPyX 0.6.3; `type_of_transform=Rigid`; `aff_metric=mattes`; 32 histogram bins; random sampling rate `0.2`; seed 12345; single precision |
| ANTs pyramid | Iterations `(2100,1200,1200,10)`; shrink factors `(6,4,2,1)`; smoothing sigmas `(3,2,1,0)` |
| ANTs initialization | `initial_transform` uses the ANTsPy 0.6.3 default center-of-mass translation for Rigid registration; the registration call supplies moving and fixed image intensities |

FOTMIR, ICP and FGR receive the same normalized point arrays. FGR radii and voxel size are in those normalized coordinate units. ANTs uses the physical image geometry. Its Mattes metric is the optimizer's objective; the reported HD95 and histogram MI are calculated during evaluation. The ANTs call is implemented in `run_experiments.py`; ICP and FGR helpers are in `baselines/`.

## Environment and timing

Recorded CPU: two Intel Xeon E5-2680 v4 processors at 2.40 GHz, 28 physical cores and 56 logical CPUs. The environment record is `configs/environment_observed.json` (Linux 5.4.0-216-generic, x86_64, glibc 2.31). Dependency pins are NumPy 2.0.2, SciPy 1.13.1, SimpleITK 2.5.3, ANTsPyX 0.6.3, Open3D 0.18.0, Matplotlib 3.9.2, pandas 2.3.2 and threadpoolctl 3.6.0.

The recorded timing run processed cases and methods sequentially on 2026-10-11, with one timed invocation per case and method. FOTMIR, ICP and FGR used one numerical thread; ANTs used 28 ITK threads. Threadpoolctl is retained by this experiment runner for its recorded numerical-thread policy.

Runtime is registration-stage wall time in seconds. FOTMIR includes coreset construction, transport/alignment updates and its final nearest-neighbor score. FGR includes downsampling, normals, FPFH and registration in the main experiment runner. ANTs time surrounds `ants.registration`. Image loading, point extraction and evaluation occur outside the recorded timers. The figures summarize the per-case recorded times.

## Core implementation and entry points

The benchmark entry point uses `Experiments/fotmir_core`; the desktop and image command-line application use `Software/fotmir_core`. Both are distributed as FOTMIR 1.0.0. The experiment core retains the recorded benchmark behavior, including its coordinate-sum support filter. The application core uses positive-mass support filtering, restores the full transport-flow shape, caps the coreset size at the available point count and rejects degenerate input geometry. Its thread limits are applied before worker startup; the benchmark runner retains threadpoolctl for the recorded timing policy.

Both cores center the weighted coresets, alternate transport with SVD-based orthogonal updates, project the final matrix to a proper rotation, and recover translation from the transported correspondences. Translation is recovered after the alternating loop. The transport mass parameter is `tau=0.05`. The solver's limit of 5000 refers to its internal counter, which also advances at convergence checks; it is separate from the maximum of 51 outer updates.

The result tables in `Results/` retain the recorded benchmark measurements. Use the experiment entry point and the manifest's point arrays and initialization when comparing against those tables. The application entry points provide the interactive image workflow.
