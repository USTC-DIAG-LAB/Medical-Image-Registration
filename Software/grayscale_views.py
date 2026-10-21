"""Export paired grayscale slices in the fixed image's native grid directions."""
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg


def export_grayscale_views(fixed, registered, output_dir, valid_support=None):
    """Write six PNGs with matched slices and physical in-plane aspect ratios.

    Native x/y/z are image-grid axes, not inferred anatomical directions.
    valid_support marks where the resampled moving image has source coverage.
    Display normalization never changes either input image.
    """
    for image in (fixed, registered):
        if image.GetDimension() != 3 or image.GetNumberOfComponentsPerPixel() != 1:
            raise ValueError('Grayscale views require scalar 3D images.')
    if fixed.GetSize() != registered.GetSize() or any(
        not np.allclose(getattr(fixed, key)(), getattr(registered, key)(), rtol=0, atol=1e-6)
        for key in ('GetSpacing', 'GetOrigin', 'GetDirection')
    ):
        raise ValueError('Grayscale views require both images on the fixed image grid.')
    arrays = {'fixed': sitk.GetArrayFromImage(fixed),
              'registered': sitk.GetArrayFromImage(registered)}
    support = np.ones(arrays['fixed'].shape, dtype=bool)
    if valid_support is not None:
        support = np.asarray(valid_support, dtype=bool)
        if support.shape != arrays['fixed'].shape:
            raise ValueError('Moving-image coverage must match the fixed image grid.')
    valid = {'fixed': np.isfinite(arrays['fixed']),
             'registered': support & np.isfinite(arrays['registered'])}
    windows = {}
    for role, array in arrays.items():
        values = array[valid[role]]
        low, high = np.percentile(values, [1, 99]) if values.size else (0., 1.)
        if high <= low:
            high = low + 1.
        windows[role] = {'lower': float(low), 'upper': float(high)}
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    spacing_zyx = tuple(reversed(fixed.GetSpacing()))
    views = []
    files = []
    for axis, name in enumerate(('z', 'y', 'x')):
        index = arrays['fixed'].shape[axis] // 2
        row_axis, column_axis = [i for i in range(3) if i != axis]
        rows, columns = [arrays['fixed'].shape[i] for i in (row_axis, column_axis)]
        height_mm = rows * spacing_zyx[row_axis]
        width_mm = columns * spacing_zyx[column_axis]
        scale = 800. / max(width_mm, height_mm)
        width_px = max(1, round(width_mm * scale))
        height_px = max(1, round(height_mm * scale))
        filenames = {}
        for role, array in arrays.items():
            plane = np.take(array, index, axis=axis).astype(np.float64)
            covered = np.take(valid[role], index, axis=axis)
            window = windows[role]
            normalized = np.zeros_like(plane)
            normalized[covered] = np.clip(
                (plane[covered] - window['lower']) / (window['upper'] - window['lower']), 0, 1)
            filename = f'{role}_{name}.png'
            fig = Figure(figsize=(width_px / 100., height_px / 100.), dpi=100)
            FigureCanvasAgg(fig)
            ax = fig.add_axes((0, 0, 1, 1))
            ax.imshow(normalized, cmap='gray', vmin=0, vmax=1, origin='lower',
                      extent=(0, width_mm, 0, height_mm), aspect='auto', interpolation='nearest')
            ax.set_axis_off()
            fig.savefig(output / filename, dpi=100, facecolor='black', transparent=False)
            fig.clear()
            filenames[role] = filename
            files.append(filename)
        views.append({'native_axis': name, 'slice_index_zero_based': index,
                      'in_plane_size_mm': [width_mm, height_mm],
                      'png_size_pixels': [width_px, height_px], 'files': filenames})
    return {'files': files, 'coordinate_system': 'fixed image native grid',
            'fixed_direction': list(fixed.GetDirection()),
            'display_limits': windows, 'display_percentiles': [1, 99],
            'normalization': 'independent volume-level window for each image; moving coverage only',
            'origin': 'lower', 'interpolation': 'nearest', 'views': views}
