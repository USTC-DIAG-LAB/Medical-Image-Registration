"""Generate normalized point sets with boundary/interior region sampling."""
import argparse
import json
from pathlib import Path
import numpy as np
import SimpleITK as sitk
if __package__:
    from .image_points import extract_points_from_masks, infer_foreground_mask, normalize_pair
else:
    from image_points import extract_points_from_masks, infer_foreground_mask, normalize_pair


def sample_image(image_path, region_path, modality, mode, count=10000, seed=20261010):
    image = sitk.ReadImage(str(image_path))
    if image.GetDimension() != 3 or image.GetNumberOfComponentsPerPixel() != 1:
        raise ValueError('Expected a scalar 3D image')
    if mode not in ('organ', 'mask', 'intensity'):
        raise ValueError('mode must be organ, mask or intensity')
    if count < 3: raise ValueError('At least three points are required')
    if mode == 'intensity':
        if region_path is not None: raise ValueError('Intensity mode must not receive an annotation')
        source = image
        masks = [infer_foreground_mask(image, modality, 'fast')]
        label_ids = []
    else:
        if region_path is None: raise ValueError('Explicit region required; no silent intensity fallback')
        source = sitk.ReadImage(str(region_path))
        if source.GetDimension() != 3 or source.GetNumberOfComponentsPerPixel() != 1:
            raise ValueError('Expected a scalar 3D label/mask image')
        array = sitk.GetArrayFromImage(source)
        if not np.isfinite(array).all() or np.any(array < 0) or not np.equal(array, np.floor(array)).all():
            raise ValueError('Regions must contain finite nonnegative integer labels')
        label_ids = [int(v) for v in np.unique(array) if v > 0]
        masks = [array == value for value in label_ids]
    if not masks or not any(m.any() for m in masks): raise ValueError('No usable region voxels')
    # Separate same-seed RNG for each side matches the added-case recipe.
    points = extract_points_from_masks(source, masks, 'mixed', count, np.random.default_rng(seed))
    if len(points) < 3: raise ValueError('Too few unique region points')
    return points, label_ids


def prepare_pair(moving_image, fixed_image, moving_region, fixed_region, moving_modality,
                 fixed_modality, mode, count=10000, seed=20261010):
    points, ids = {}, {}
    sides = [('moving', moving_image, moving_region, moving_modality),
             ('fixed', fixed_image, fixed_region, fixed_modality)]
    for side, image, region, modality in sides:
        points[side], ids[side] = sample_image(image, region, modality, mode, count, seed)
    X, Y, center, scale = normalize_pair(points['moving'], points['fixed'])
    result = dict(A=X.T, B=Y.T, X=X, Y=Y, center_Y=center, scale=np.array(scale),
                  evaluation_schema=np.array('explicit_optional_references_v2'),
                  point_extraction_policy=np.array('region_mixed_80_20' if mode != 'intensity' else 'intensity_mixed_80_20'),
                  sampling_seed=np.array(seed), requested_points=np.array(count),
                  labels_role=np.array('sampling_and_evaluation' if mode == 'organ' else 'region_sampling_or_no_labels'))
    for side, image, region, modality in sides:
        result[side+'_image_file'] = np.array(str(Path(image).resolve()))
        result[side+'_modality'] = np.array(modality)
        result[side+'_label_file'] = np.array(str(Path(region).resolve()) if mode == 'organ' else '')
        result[side+'_mask_file'] = np.array(str(Path(region).resolve()) if mode == 'mask' else '')
        result[side+'_landmark_file'] = np.array('')
        result[side+'_labels_used'] = np.asarray(ids[side], dtype=np.int32)
    return result


def cli():
    p = argparse.ArgumentParser(description=__doc__)
    for side in ('moving', 'fixed'):
        p.add_argument('--'+side+'-image', required=True, type=Path)
        p.add_argument('--'+side+'-region', type=Path)
        p.add_argument('--'+side+'-modality', required=True)
    p.add_argument('--mode', choices=('organ','mask','intensity'), required=True)
    p.add_argument('--num-samples', type=int, default=10000)
    p.add_argument('--seed', type=int, default=20261010)
    p.add_argument('--output', required=True, type=Path)
    a=p.parse_args()
    if a.output.exists(): p.error('Output exists; choose a new NPZ file')
    if a.output.suffix.lower() != '.npz': p.error('Output must end with .npz')
    result=prepare_pair(a.moving_image,a.fixed_image,a.moving_region,a.fixed_region,
                        a.moving_modality,a.fixed_modality,a.mode,a.num_samples,a.seed)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('xb') as stream: np.savez_compressed(stream,**result)
    print(json.dumps({'output':str(a.output),'points':[len(result['X']),len(result['Y'])],
                      'sampling_seed':a.seed}))

if __name__ == '__main__': cli()
