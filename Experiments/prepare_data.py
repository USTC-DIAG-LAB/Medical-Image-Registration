"""Arrange archived point arrays and image references in a portable data directory."""
import argparse, hashlib, json, shutil
from pathlib import Path
from input_data import ROOT, ARRAY_KEYS, jobs_for, safe_path

def cli():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-prepared',type=Path,required=True); p.add_argument('--source-data',type=Path)
    p.add_argument('--output-dir',type=Path,required=True); p.add_argument('--cases',nargs='+'); a=p.parse_args()
    if a.output_dir.exists() and any(a.output_dir.iterdir()): p.error('Output directory must be new or empty')
    import numpy as np
    import SimpleITK as sitk
    jobs=jobs_for(ROOT/'manifests/main83.json',a.cases); arrays=[]; sources={}
    for job in jobs:
        with np.load(safe_path(a.source_prepared,job['prepared_input']),allow_pickle=False) as z: pair={k:z[k] for k in z.files}
        for key in ARRAY_KEYS:
            if hashlib.sha256(pair[key].tobytes()).hexdigest()!=job[key+'_sha256']: raise ValueError('Array hash mismatch: '+job['case']+' '+key)
        for key,value in job.items():
            if key.endswith('_file'):
                if value:
                    if a.source_data: src=safe_path(a.source_data,value)
                    else:
                        raw=str(np.asarray(pair.get(key,'')).item())
                        if not raw and key.endswith('_mask_file'):
                            raw=str(np.asarray(pair.get(key.replace('_mask_file','_label_file'),'')).item())
                        src=Path(raw)
                    if not src.is_file(): raise FileNotFoundError(str(src)+' for '+job['case']+' '+key)
                    safe_path(a.output_dir/'external_data',value)
                    if value in sources and sources[value].resolve()!=src.resolve(): raise ValueError('Conflicting sources: '+value)
                    sources[value]=src
                pair[key]=np.array(value)
            elif key.endswith('_modality'): pair[key]=np.array(value)
        arrays.append((job,pair))
    for relative,src in sorted(sources.items()):
        dst=safe_path(a.output_dir/'external_data',relative); dst.parent.mkdir(parents=True,exist_ok=True)
        if dst.suffix.lower()=='.mhd':
            im=sitk.ReadImage(str(src)); sitk.WriteImage(im,str(dst)); copy=sitk.ReadImage(str(dst))
            if im.GetSpacing()!=copy.GetSpacing() or im.GetOrigin()!=copy.GetOrigin() or im.GetDirection()!=copy.GetDirection() or not np.array_equal(sitk.GetArrayViewFromImage(im),sitk.GetArrayViewFromImage(copy)): raise ValueError('MetaImage copy mismatch: '+relative)
        else: shutil.copy2(src,dst)
    for job,pair in arrays:
        dst=safe_path(a.output_dir/'prepared_inputs',job['prepared_input']); dst.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(dst,**pair)
    print(json.dumps({'output_dir': str(a.output_dir.resolve()), 'cases': len(jobs)}, indent=2))
if __name__=='__main__': cli()
