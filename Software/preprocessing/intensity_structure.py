"""Generic image-only 3D structure sampling; no modality or anatomy argument.

The same physical-scale filters and spatial balancing apply to every image.
This supplies geometric candidates, not organ segmentation or correspondences.
"""
import numpy as np
from scipy import ndimage as ndi

VERSION = 'image_structure_multiscale_v2'


def normalize_intensity(array):
    arr = np.asarray(array, dtype=np.float32)
    finite = np.isfinite(arr)
    if arr.ndim != 3 or min(arr.shape) < 3 or not finite.any():
        raise ValueError('Expected a nonempty 3D image, at least 3 voxels per axis')
    lo, hi = np.percentile(arr[finite], [.5, 99.5])
    if hi <= lo: lo, hi = float(arr[finite].min()), float(arr[finite].max())
    if hi <= lo: raise ValueError('Constant image has no usable structure')
    normalized = np.clip((np.where(finite, arr, lo) - lo) / (hi - lo), 0, 1)
    return normalized, finite, (float(lo), float(hi))


def background_from_border(image, finite):
    """Estimate a dominant border background class, including its noise spread.

    Handles dark or bright padding equally; makes no HU or tissue assumption.
    """
    faces = np.zeros(image.shape, dtype=bool)
    for axis in range(3):
        for value in [0, -1]:
            sl = [slice(None)] * 3; sl[axis] = value; faces[tuple(sl)] = True
    values = image[faces & finite]
    if not len(values): return ~finite, {'border_background_detected': False}
    hist, bins = np.histogram(values, bins=128, range=(0, 1))
    smoothed_hist = ndi.gaussian_filter1d(hist.astype(float), 1.0)
    peak = int(smoothed_hist.argmax())
    lower = upper = peak
    cutoff = .1 * smoothed_hist[peak]
    while lower > 0 and smoothed_hist[lower-1] >= cutoff: lower -= 1
    while upper < len(hist)-1 and smoothed_hist[upper+1] >= cutoff: upper += 1
    # A broad border distribution is not identifiable as a background mode.
    if bins[upper+1]-bins[lower] > .15:
        return ~finite, {'border_background_detected': False}
    cluster = values[(values >= bins[lower]) & (values <= bins[upper+1])]
    center = float(np.median(cluster))
    tolerance = max(.005, float(np.percentile(np.abs(cluster-center), 99.5)) * 1.25)
    flat_class = finite & (np.abs(image - center) <= tolerance)
    border_fraction = float(flat_class[faces & finite].mean())
    if border_fraction < .2:
        return ~finite, {'border_background_detected': False}
    connected = ndi.binary_propagation(faces & flat_class, mask=flat_class)
    # Isolated residual speckles after removing the connected background are
    # not reliable structures. Keep all substantial components, not one organ.
    labels, _ = ndi.label(finite & ~connected)
    sizes = np.bincount(labels.ravel()); sizes[0] = 0
    speckles = (labels > 0) & (sizes[labels] < 32)
    return connected | speckles | ~finite, {'border_background_detected': True,
                                'normalized_background_value': center,
                                'normalized_background_tolerance': tolerance,
                                'background_voxels': int(connected.sum()),
                                'isolated_background_speckle_voxels': int(speckles.sum())}


def structure_candidates(array, spacing_zyx, scales_mm=(1.5, 3.0), margin_mm=3.0, focus='balanced'):
    if focus not in {'balanced', 'interior'}: raise ValueError('Unknown structure focus')
    spacing = np.asarray(spacing_zyx, dtype=float)
    if spacing.shape != (3,) or not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError('Spacing must contain three positive finite physical lengths')
    scales = np.asarray(scales_mm, dtype=float)
    if len(scales) < 2 or np.any(~np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError('At least two positive Gaussian scales are required')
    if not np.isfinite(margin_mm) or margin_mm < 0: raise ValueError('Invalid field margin')
    image, finite, intensity_range = normalize_intensity(array)
    background, background_info = background_from_border(image, finite)
    usable = finite & ~background
    if not usable.any(): raise ValueError('Only flat background was found')
    # Guard both the array edge and constant-padded acquisition extent. This is
    # a rejection mask only: its boundary never creates artificial surface points.
    bounds = []
    for axis in range(3):
        active = np.flatnonzero(np.any(usable, axis=tuple(i for i in range(3) if i != axis)))
        n = max(1, int(np.ceil(margin_mm / spacing[axis])))
        lower, upper = int(active[0] + n), int(active[-1] - n + 1)
        if lower >= upper: raise ValueError('Image support too thin for the requested physical margin')
        bounds.append((lower, upper))
    field = np.zeros(image.shape, dtype=bool)
    field[tuple(slice(lo, hi) for lo, hi in bounds)] = True
    usable &= field
    if not usable.any(): raise ValueError('No image support after field margin')
    score = np.ones(image.shape, dtype=np.float32)
    scale_stats = []
    for scale in scales:
        sigma = np.maximum(scale / spacing, .5)
        smoothed = ndi.gaussian_filter(image, sigma=sigma, mode='nearest')
        variance = np.maximum(ndi.gaussian_filter(image * image, sigma=sigma, mode='nearest') - smoothed * smoothed, 0)
        contrast = np.sqrt(variance)
        gradient_sq = np.zeros(image.shape, dtype=np.float32)
        for axis in range(3):
            order = [0, 0, 0]; order[axis] = 1
            derivative = ndi.gaussian_filter(image, sigma=sigma, order=order, mode='nearest') / spacing[axis]
            gradient_sq += derivative * derivative
        gradient = np.sqrt(gradient_sq)
        positive = gradient[usable & (gradient > 1e-7)]
        if not len(positive): raise ValueError('No nonconstant image structure')
        noise_floor = max(1e-6, float(np.percentile(positive, 90)) * .03)
        reliability = np.clip(gradient / (noise_floor * 3), 0, 1)
        response = scale * gradient / (contrast + .03)
        response *= reliability
        reference = max(float(np.percentile(response[usable], 90)), 1e-6)
        score *= np.clip(response / reference, 0, 1)
        scale_stats.append({'scale_mm': float(scale), 'noise_floor': noise_floor,
                            'response_p90': reference})
    # Geometric mean rewards structures that persist at both scales.
    score **= 1 / len(scales)
    focus_applied = focus == 'interior' and background_info['border_background_detected']
    if focus_applied:
        # A generic soft preference for internal structures, not organ recognition.
        # The envelope comes only from image-estimated background. No annotation
        # or paired-image result enters this weight; the 6 mm ramp is shared.
        depth = ndi.distance_transform_edt(np.pad(finite & ~background, 1), sampling=spacing)[1:-1,1:-1,1:-1]
        score *= .1 + .9 * np.clip(depth / 6.0, 0, 1)
    score[~usable] = 0
    valid_score = score[usable & (score > .02)]
    if not len(valid_score): raise ValueError('No reliable multi-scale structure')
    threshold = max(.05, float(np.percentile(valid_score, 30)))
    candidate = usable & (score >= threshold)
    strong_threshold = float(np.percentile(score[candidate], 65))
    strong = candidate & (score >= strong_threshold)
    weak = candidate & ~strong
    report = {'version': VERSION if focus == 'balanced' else 'image_structure_interior_v3', 'profile': 'structure', 'intensity_range': intensity_range,
              'focus': focus, 'interior_focus_applied': focus_applied, 'interior_ramp_mm': 6.0 if focus == 'interior' else None,
              'scales': scale_stats, 'margin_mm': float(margin_mm), 'support_bounds_zyx': bounds,
              'candidate_voxels': int(candidate.sum()), 'strong_voxels': int(strong.sum()),
              'weak_voxels': int(weak.sum()), 'candidate_threshold': threshold,
              'strong_threshold': strong_threshold, 'labels_used': False, 'roi_used': False,
              'anatomy_or_modality_used': False, **background_info}
    return strong, weak, score, report


def balanced_indices(mask, score, count, spacing_zyx, cell_mm, rng):
    flat = np.flatnonzero(mask)
    count = min(int(count), len(flat))
    if count <= 0: return np.empty((0, 3), dtype=np.int64)
    coords = np.column_stack(np.unravel_index(flat, mask.shape))
    if count == len(flat): return coords
    spacing = np.asarray(spacing_zyx)
    cells = np.floor(coords * spacing / cell_mm).astype(np.int64)
    ids = np.ravel_multi_index(cells.T, tuple(cells.max(axis=0) + 1))
    # Weighted random ordering within cells, then one point per occupied cell
    # before any cell gets a second point. No duplication to fill the budget.
    priority = -np.log(np.maximum(rng.random(len(flat)), 1e-12)) / (.1 + score.ravel()[flat])
    order = np.lexsort((priority, ids))
    ordered_ids = ids[order]
    starts = np.r_[0, np.flatnonzero(np.diff(ordered_ids)) + 1]
    rank = np.arange(len(order)) - np.repeat(starts, np.diff(np.r_[starts, len(order)]))
    rounds = np.lexsort((rng.random(len(order)), rank))[:count]
    return coords[order[rounds]]


def sample_structure(array, spacing_zyx, strategy, count, rng, scales_mm=(1.5, 3.0), margin_mm=3.0, cell_mm=8.0, focus='balanced'):
    if not np.isfinite(cell_mm) or cell_mm <= 0: raise ValueError('Cell size must be positive')
    if count is not None and count <= 0: raise ValueError('Sample count must be positive')
    strong, weak, score, report = structure_candidates(array, spacing_zyx, scales_mm, margin_mm, focus)
    if strategy == 'all':
        masks = [strong | weak]; budgets = [int(masks[0].sum())]
    elif strategy == 'boundary':
        masks = [strong]; budgets = [int(strong.sum()) if count is None else min(count, int(strong.sum()))]
    elif strategy == 'mixed':
        total = int(strong.sum() + weak.sum()) if count is None else min(count, int(strong.sum() + weak.sum()))
        nstrong = min(int(round(total * .8)), int(strong.sum()))
        nweak = min(total - nstrong, int(weak.sum()))
        nstrong = min(total - nweak, int(strong.sum()))
        masks = [strong, weak]; budgets = [nstrong, nweak]
    else: raise ValueError('Unknown sampling strategy: ' + strategy)
    indices = np.concatenate([balanced_indices(m, score, n, spacing_zyx, cell_mm, rng) for m, n in zip(masks, budgets)])
    occupied = np.unique(np.floor(indices * np.asarray(spacing_zyx) / cell_mm).astype(int), axis=0)
    report.update(strategy=strategy, point_count=len(indices), cell_mm=cell_mm,
                  occupied_cells=len(occupied), sample_mean_structure_score=float(score[tuple(indices.T)].mean()),
                  boundary_semantics='persistent_image_structure', requested_count=count)
    return indices, report
