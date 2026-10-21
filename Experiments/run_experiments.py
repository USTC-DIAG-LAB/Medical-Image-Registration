"""Run FOTMIR, ICP, FGR and rigid ANTs on paired medical images and point sets."""
import argparse,os,json,time,hashlib,csv
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PACKAGE_VERSION=(ROOT.parent/'VERSION').read_text(encoding='utf-8').strip()
from input_data import jobs_for, load_pair
from evaluation.reporting import primary_metrics, save_rows

def cli():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--version',action='version',version='FOTMIR '+PACKAGE_VERSION)
 p.add_argument('--manifest',type=Path,default=ROOT/'manifests/main83.json');p.add_argument('--prepared-root',type=Path,required=True);p.add_argument('--data-root',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
 p.add_argument('--methods',nargs='+',choices=['python','icp','fgr','ants'],default=['python','icp','fgr','ants']);p.add_argument('--cases',nargs='+');a=p.parse_args()
 for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
 os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS']='28'
 import numpy as np
 import SimpleITK as sitk
 from threadpoolctl import threadpool_limits
 from fotmir_core.main import main
 from evaluation import metrics as ev
 configs=json.loads((ROOT/'configs/benchmark.json').read_text());jobs=jobs_for(a.manifest,a.cases)
 if a.cases:
  unknown=set(a.cases)-{j['case'] for j in jobs}
  if unknown:p.error('Unknown cases: '+str(sorted(unknown)))
  jobs=[j for j in jobs if j['case'] in a.cases]
 out=a.output_dir.resolve()
 if out.exists() and (not out.is_dir() or any(out.iterdir())):p.error('Output directory must be new or empty')
 if len(set(a.methods)) != len(a.methods): p.error('Methods must be unique')
 out.mkdir(parents=True,exist_ok=True);rows=[]
 (out/'run_configuration.json').write_text(json.dumps(dict(version=PACKAGE_VERSION,configs=configs,methods=a.methods,cases=[j['case'] for j in jobs],manifest_sha256=hashlib.sha256(a.manifest.read_bytes()).hexdigest(),timing_scope='registration; excludes input reading and evaluation'),indent=2))
 for j in jobs:
  pair=load_pair(j,a.prepared_root,a.data_root)
  folder=out/j['dataset']/j['case'];folder.mkdir(parents=True)
  points_file=folder/'points.npz';np.savez(points_file,A=pair['X'].T,B=pair['Y'].T);measured=[]
  for method in a.methods:
   result=folder/(method+'.npz')
   try:
    with threadpool_limits(limits=1):
     if method=='python':
      cfg=configs['python'];r=main(points_file,result,'manual',cfg['tau'],cfg['ratio'],seed=j['seed'],initial_indices=j['initial_indices'],threads=cfg['threads']);tm=ev.tm_norm_to_physical(r['TM'],pair['center_Y'],pair['scale']);seconds=r['elapsed_time']
     elif method=='icp':
      from baselines.icp import robust_icp_3d
      cfg=configs['icp'];beg=time.perf_counter();_,norm,h=robust_icp_3d(pair['X'],pair['Y'],max_iter=cfg['max_iter'],outlier_scale=cfg['outlier_scale'],tolerance=cfg['tolerance'],return_details=True);seconds=time.perf_counter()-beg;tm=ev.tm_norm_to_physical(norm,pair['center_Y'],pair['scale']);np.savez(result,TM=norm,elapsed_time=seconds)
     elif method=='fgr':
      import open3d as o3d
      from baselines.fgr import set_seed,preprocess,to_pcd
      cfg=configs['fgr'];set_seed(cfg['seed']);beg=time.perf_counter();src,sf=preprocess(to_pcd(pair['X']),cfg['voxel_size'],cfg['normal_radius'],cfg['fpfh_radius']);dst,df=preprocess(to_pcd(pair['Y']),cfg['voxel_size'],cfg['normal_radius'],cfg['fpfh_radius']);opt=o3d.pipelines.registration.FastGlobalRegistrationOption(maximum_correspondence_distance=cfg['maximum_correspondence_distance']);norm=np.asarray(o3d.pipelines.registration.registration_fgr_based_on_feature_matching(src,dst,sf,df,opt).transformation);seconds=time.perf_counter()-beg;tm=ev.tm_norm_to_physical(norm,pair['center_Y'],pair['scale']);np.savez(result,TM=norm,elapsed_time=seconds)
     else:
      import ants
      cfg=configs['ants'];sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(cfg['threads']);ants.config.set_ants_deterministic(on=False,seed_value=cfg['seed'])
      fixed=ants.image_read(str(pair['fixed_image_file'].item()));moving=ants.image_read(str(pair['moving_image_file'].item()))
      kw={k:v for k,v in cfg.items() if k in ['type_of_transform','aff_metric','aff_sampling','aff_random_sampling_rate','singleprecision']}
      for key in ['aff_iterations','aff_shrink_factors','aff_smoothing_sigmas']:kw[key]=tuple(cfg[key])
      beg=time.perf_counter();r=ants.registration(fixed=fixed,moving=moving,outprefix=str(folder/'ants_'),**kw);seconds=time.perf_counter()-beg
      push=sitk.ReadTransform(r['fwdtransforms'][0]).GetInverse();t=np.array(push.TransformPoint((0.,0.,0.)));tm=np.eye(4);tm[:3,3]=t;tm[:3,:3]=np.column_stack([np.array(push.TransformPoint(tuple(v)))-t for v in np.eye(3)]);np.savez(result,TM_physical=tm,elapsed_time=seconds)
   except Exception as exc:
    rows.append(dict(version=PACKAGE_VERSION,dataset=j['dataset'],case=j['case'],method=method,group=j['group'],status='registration_failed',message=type(exc).__name__+': '+str(exc)))
    save_rows(out,rows)
    print(j['case'],method,'FAILED:',exc,flush=True)
    continue
   measured.append((method,seconds,tm));print(j['case'],method,seconds,flush=True)
  sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
  for method,seconds,tm in measured:
   row=dict(version=PACKAGE_VERSION,dataset=j['dataset'],case=j['case'],method=method,group=j['group'],elapsed_time_s=float(seconds),TM_physical=tm.tolist())
   row.update(ev.evaluate_case(j['dataset'],pair,tm,a.data_root));primary_metrics(row,j['group']);rows.append(row)
  save_rows(out,rows)
 valid=sum(r.get('status')=='success' and r.get('primary_hd95_mm') is not None for r in rows)
 summary=dict(version=PACKAGE_VERSION,expected_rows=len(jobs)*len(a.methods),recorded_rows=len(rows),successful_primary_rows=valid,
              failed_or_incomplete_rows=len(rows)-valid)
 (out/'run_summary.json').write_text(json.dumps(summary,indent=2))
 return 0 if valid==summary['expected_rows'] else 1
if __name__=='__main__':raise SystemExit(cli())
