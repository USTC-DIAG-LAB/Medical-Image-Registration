"""Image-only Python CLI added for the revision candidate, 2026-10-12."""
import argparse,json,os,time
from pathlib import Path

def cli():
 p=argparse.ArgumentParser(description=__doc__)
 for side in ['moving','fixed']:
  p.add_argument('--'+side+'-image',required=True,type=Path)
  p.add_argument('--'+side+'-modality',required=True)
 p.add_argument('--output-dir',required=True,type=Path)
 p.add_argument('--num-samples',default=10000,type=int);p.add_argument('--seed',default=20260722,type=int)
 p.add_argument('--threads',default=1,type=int);p.add_argument('--profile',choices=['structure','legacy'],default='structure')
 p.add_argument('--tau',default=.05,type=float);p.add_argument('--ratio',default=.05,type=float)
 a=p.parse_args()
 if os.environ.get('FOTMIR_IMAGE_WORKER') != str(a.threads):p.error('Use one_click_registration.py or register.py to launch this worker.')
 if a.threads<1 or a.num_samples<50:p.error('threads>=1 and num-samples>=50 required')
 for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]=str(a.threads)
 import numpy as np
 import SimpleITK as sitk
 from preprocessing.image_points import ImageRecord,extract_point_cloud,normalize_pair
 from fotmir_core.main import main, PACKAGE_VERSION
 from geometry import tm_norm_to_physical,resample_moving_to_fixed
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(a.threads)
 for f in [a.moving_image,a.fixed_image]:
  if not f.is_file():p.error('Missing image: '+str(f))
 out=a.output_dir.resolve()
 if out.exists() and any(out.iterdir()):p.error('Output directory must be new or empty')
 out.mkdir(parents=True,exist_ok=True);started=time.perf_counter();rng=np.random.default_rng(a.seed)
 points={};diag={}
 for side in ['moving','fixed']:
  path=getattr(a,side+'_image').resolve();mod=getattr(a,side+'_modality')
  record=ImageRecord(path,path.name,'user_pair',mod,None,None,'','')
  diag[side]={};points[side],_,_=extract_point_cloud(record,'mixed',a.num_samples,'fast',rng,extraction_profile=a.profile,diagnostics=diag[side])
 X,Y,c,s=normalize_pair(points['moving'],points['fixed'])
 np.savez(out/'points.npz',A=X.T,B=Y.T,X=X,Y=Y,center_Y=c,scale=s)
 r=main(out/'points.npz',out/'registration.npz','manual',a.tau,a.ratio,seed=a.seed,threads=None)
 tm=tm_norm_to_physical(r['TM'],c,s)
 moving=sitk.Cast(sitk.ReadImage(str(a.moving_image)),sitk.sitkFloat32);fixed=sitk.ReadImage(str(a.fixed_image))
 registered=resample_moving_to_fixed(moving,fixed,tm,sitk.sitkLinear,0)
 sitk.WriteImage(registered,str(out/'moving_registered.nii.gz'));sitk.WriteImage(fixed,str(out/'fixed_reference.nii.gz'))
 af=sitk.GetArrayFromImage(fixed).astype(float);ar=sitk.GetArrayFromImage(registered).astype(float)
 def norm(v):
  lo,hi=np.percentile(v,[1,99]);return np.clip((v-lo)/max(hi-lo,1e-8),0,1)
 af,ar=norm(af),norm(ar);fig,axes=plt.subplots(1,3,figsize=(12,4))
 for axis,ax in enumerate(axes):
  idx=af.shape[axis]//2;f=np.take(af,idx,axis=axis);r_=np.take(ar,idx,axis=axis)
  ax.imshow(np.stack([f,r_,f],axis=-1),origin='lower');ax.set_title(['Native z slice','Native y slice','Native x slice'][axis]);ax.axis('off')
 fig.tight_layout();fig.savefig(out/'overlay.png',dpi=160);plt.close(fig)
 summary={'thread_policy':'environment_before_process_start','thread_environment':{k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS','ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS']},'implementation':'python','version':PACKAGE_VERSION,'labels_used':False,'extraction_profile':a.profile,'extraction_diagnostics':diag,'points':[len(X),len(Y)],'seed':a.seed,'threads':a.threads,'tau':a.tau,'ratio':a.ratio,'elapsed_registration_s':float(r['elapsed_time']),'elapsed_pipeline_s':time.perf_counter()-started,'transform_convention':'moving-to-fixed physical point transform in mm; inverse used for image resampling','TM_physical':tm.tolist(),'outputs':['registration.npz','points.npz','moving_registered.nii.gz','fixed_reference.nii.gz','overlay.png']}
 (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps({'output_dir':str(out),'elapsed_registration_s':summary['elapsed_registration_s']}))
 return 0
if __name__=='__main__':raise SystemExit(cli())
