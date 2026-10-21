"""Load point arrays and resolve the image references in a case manifest."""
import hashlib, json
from pathlib import Path
ROOT = Path(__file__).resolve().parent
ARRAY_KEYS = ('X', 'Y', 'center_Y', 'scale')
def safe_path(root, relative):
    root = Path(root).resolve()
    rel = Path(relative)
    # External datasets may intentionally be linked into this layout.
    # Reject unsafe manifest spelling, while allowing these read-only links.
    path = root / rel
    if rel.is_absolute() or '..' in rel.parts:
        raise ValueError('Manifest path escapes data root: ' + relative)
    return path.resolve()
def jobs_for(manifest, cases=None):
    jobs = json.loads(Path(manifest).read_text())
    keys = [(j['dataset'], j['case']) for j in jobs]
    if not jobs or len(set(keys)) != len(keys):
        raise ValueError('Manifest must contain nonempty, unique dataset/case entries')
    for j in jobs:
        for key in ('dataset', 'case'):
            if not j[key] or Path(j[key]).name != j[key] or j[key] in ('.', '..'):
                raise ValueError('Unsafe dataset/case name: ' + j[key])
    if cases:
        unknown = set(cases) - {j['case'] for j in jobs}
        if unknown: raise ValueError('Unknown cases: ' + str(sorted(unknown)))
        jobs = [j for j in jobs if j['case'] in cases]
    return jobs
def load_pair(job, prepared_root, data_root):
    import numpy as np
    with np.load(safe_path(prepared_root, job['prepared_input']), allow_pickle=False) as z:
        pair = {k: z[k] for k in z.files}
    for key in ARRAY_KEYS:
        if hashlib.sha256(np.asarray(pair[key]).tobytes()).hexdigest() != job[key + '_sha256']:
            raise ValueError('Array hash mismatch: ' + job['case'] + ' ' + key)
    for key in ('X', 'Y'):
        points = np.asarray(pair[key])
        if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
            raise ValueError('Expected finite nonempty Nx3 array: ' + key)
    if np.asarray(pair['center_Y']).size != 3 or not np.isfinite(pair['center_Y']).all():
        raise ValueError('Expected finite 3D normalization center')
    if np.asarray(pair['scale']).size != 1 or not np.isfinite(pair['scale']).all() or float(pair['scale']) <= 0:
        raise ValueError('Expected positive finite normalization scale')
    for key, value in job.items():
        if key.endswith('_file'):
            path = safe_path(data_root, value) if value else None
            if path and not path.is_file(): raise FileNotFoundError(path)
            pair[key] = np.array(str(path) if path else '')
        elif key.endswith('_modality'): pair[key] = np.array(value)
    # Explicit portable references override legacy label-as-mask conventions.
    if any(k.endswith('_mask_file') for k in job):
        pair['evaluation_schema'] = np.array('explicit_optional_references_v2')
    return pair
