# Experimental results and metric definitions

## Files and measurement coverage

`raw/case_metrics.csv` contains the retained original case-level result table: 332 rows for 83 image pairs and four methods. Method code `python` denotes FOTMIR; the other codes are `icp`, `fgr` and `ants`.

`summary/metric_summary.csv` gives the number of finite observations, arithmetic mean, median, minimum and maximum for each dataset, method and metric. `figures/` contains one PNG per metric. Bars show the mean, whiskers show the actual minimum and maximum, and short white lines with black outlines mark medians. Each panel has a linear axis starting at zero; dataset panels may use different upper limits.

| Metric | Unit | RIRE pairs per method | AbdomenMRCT pairs per method | LungCT pairs per method | Preferred direction |
| --- | --- | ---: | ---: | ---: | --- |
| HD95 | mm | 45 | 8 | 30 | Lower |
| ASD | mm | 45 | 8 | 30 | Lower |
| ASSD | mm | 45 | 8 | 30 | Lower |
| Point-set RMSE | mm | 45 | 8 | 30 | Lower |
| Registration runtime | s | 45 | 8 | 30 | Lower |
| Organ-union Dice | unitless | 0 | 8 | 0 | Higher |
| Organ-region MI | nats | 0 | 8 | 0 | Higher |
| Mean TRE | mm | 0 | 0 | 3 | Lower |
| TRE95 | mm | 0 | 0 | 3 | Lower |

Each case has equal weight within its dataset/method summary. Blank or nonfinite measurements are omitted from the corresponding metric's aggregation. Surface-distance and RMSE summaries use all cases listed in the table; optional Dice, MI and TRE use the indicated subsets.

## Coordinate conventions

Transforms map moving points into fixed physical coordinates. If the normalized point coordinates use center `c` and scale `s`, the normalized transform `(R,t)` becomes:

```text
T_physical(p) = R p + (s t + c - R c)
```

Physical positions use the image origin, spacing and direction, and distances are in millimetres. Image resampling uses the inverse transformation because the resampler maps fixed-grid positions back into the moving image. Masks and labels use nearest-neighbor interpolation with a zero background.

## HD95, ASD and ASSD

The evaluated regions are:

- **RIRE:** moving and fixed foreground masks inferred from their image intensities. CT uses `intensity > max(-700, percentile_5)`. MRI uses Otsu thresholding after clipping intensities to the first/99th percentiles. Hole filling and largest-component selection follow thresholding; an empty initial mask uses an 80th-percentile fallback.
- **AbdomenMRCT:** the union of positive organ labels in each image. These are the `organ_*` measurements for cases 0001-0008.
- **LungCT:** the supplied automatic lung masks, converted to positive-voxel binary regions.

Resample the moving region onto the fixed region's grid. For each binary region `M`, define its discrete boundary as `M XOR binary_erosion(M)` using one erosion with 3D face connectivity. Let `S_A` denote the warped moving boundary and `S_B` the fixed boundary. Define the two directed distance arrays:

```text
D_AB = { min over b in S_B of ||a-b|| : a in S_A }
D_BA = { min over a in S_A of ||b-a|| : b in S_B }

ASD  = mean(D_AB)
ASSD = (mean(D_AB) + mean(D_BA)) / 2
HD95 = percentile(concatenate(D_AB, D_BA), 95)
```

The implementation uses SciPy Euclidean distance transforms with the fixed grid's physical voxel spacing in `z,y,x` order. HD95 uses NumPy's default linear percentile interpolation on the pooled arrays. ASSD gives equal weight to the two directional means. An empty surface gives a missing measurement.

The plotted columns are `primary_hd95_mm`, `primary_asd_mm` and `primary_assd_mm`. For AbdomenMRCT these select `organ_hd95_mm`, `organ_asd_mm` and `organ_assd_mm`; for RIRE and LungCT they select the corresponding unprefixed columns. The original abdominal unprefixed columns record the broader dataset mask region and remain identifiable in the raw table.

## Point-set RMSE

Let `P` be the sampled moving points and `Q` the sampled fixed points, expressed in physical millimetres. Apply the estimated physical transform to `P`, query the nearest neighbor in `Q`, and calculate:

```text
RMSE = sqrt(mean over p in P of [min over q in Q of ||T(p)-q||]^2)
```

This is a directed point-set nearest-neighbor error. The CSV column is `rmse_mm`. Anatomical landmark correspondence is used separately by TRE.

## Dice

For the warped moving organ union `A` and fixed organ union `B` on the same grid:

```text
Dice = 2 * count(A intersection B) / (count(A) + count(B))
```

Positive organ labels are combined before calculation. The CSV column `dice` represents the union score. A zero denominator gives a missing value.

## Mutual information

MI is calculated for organ-annotated cases. Cast moving intensities to float32 and linearly resample them onto the fixed image grid. The evaluation region is the intersection of valid warped image support with the union of the warped moving organ labels and fixed organ labels; require finite intensities in both images.

Build a `64 x 64` joint histogram using `numpy.histogram2d`, with per-axis ranges determined from the selected intensities. Normalize the histogram into probabilities and calculate:

```text
MI = sum over occupied bins (i,j) of p(i,j) * log[p(i,j)/(p(i)*p(j))]
```

The logarithm is natural, so MI is expressed in nats. The CSV column is `mi`; `mi_voxel_count` records the evaluated voxel count. This histogram evaluation metric has its own definition, separate from ANTs' internal Mattes optimization metric.

## Landmark TRE

Read paired landmark coordinates in continuous voxel-index `x,y,z` order. Convert each side to physical positions using its own image geometry. For corresponding moving and fixed landmarks `p_i,q_i`:

```text
TRE_i = ||T(p_i) - q_i||
tre_mean_mm = mean(TRE_i)
tre95_mm = percentile(TRE_i, 95)
```

`landmark_count` gives the number of matched landmarks. TRE95 uses linear percentile interpolation. Dataset plots summarize these per-case quantities, giving each of the three landmark-bearing lung pairs equal weight.

## Runtime

`elapsed_time_s` is registration-stage wall time from the recorded sequential timing run. It includes FOTMIR coreset construction, transport/alignment and the final nearest-neighbor score; ICP registration; FGR downsampling, normals, FPFH and registration; or the ANTs registration call. Image loading, point extraction and metric computation take place outside these timers.

There is one timed invocation per case and method. FOTMIR, ICP and FGR use one numerical thread, and ANTs uses 28 ITK threads. Recorded hardware: two Intel Xeon E5-2680 v4 CPUs at 2.40 GHz, 28 physical cores and 56 logical CPUs. Full method settings and software versions are in [Experiments/README.md](../Experiments/README.md).

## Original table fields

| Field or family | Meaning |
| --- | --- |
| `dataset`, `case`, `method` | Measurement identifiers |
| `group`, `primary_evaluation_region` | Dataset-specific primary evaluation region |
| `primary_*_mm` | Selected primary surface measurements used in the figures |
| `organ_*_mm` | Abdominal positive-organ-union surface measurements |
| `asd_mm`, `assd_mm`, `hd95_mm` | Foreground or dataset-mask surface measurements |
| `rmse_mm`, `dice`, `mi`, `tre_mean_mm`, `tre95_mm`, `elapsed_time_s` | Metrics defined above |
| `surface_source`, `mi_roi_source` | Region provenance |
| `mi_voxel_count`, `landmark_count` | Evaluation support size |
| `status`, `*_status` | Per-case and per-metric availability/status |
| `runtime_provenance`, `metric_source` | Timing record and original metric-generation provenance |
| `max_TM_physical_difference_from_prior`, `original` | Retained historical transformation and case-selection metadata |

## Plot generation

Run from the package root:

```sh
python -m pip install -r Results/requirements.txt
python Results/plot_results.py
```

The script reads the supplied measurement CSV, creates the summary table and renders the nine metric images at 300 dpi. The metric implementation used during experiment runs is `Experiments/evaluation/metrics.py`.
