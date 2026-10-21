"""FOTMIR image registration. Standard-library launcher."""
import argparse
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

THREAD_KEYS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS','ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS')
MODULES=('numpy','scipy','SimpleITK','matplotlib')
ROOT=Path(__file__).resolve().parent
PACKAGE_VERSION=(ROOT.parent/'VERSION').read_text(encoding='utf-8').strip()

def thread_environment(threads):
    if threads < 1: raise ValueError('threads must be positive')
    env=os.environ.copy()
    for key in THREAD_KEYS: env[key]=str(threads)
    env['FOTMIR_IMAGE_WORKER']=str(threads)
    env['FOTMIR_CORE_THREADS']=str(threads)
    env['PYTHONUNBUFFERED']='1'
    return env

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--version',action='version',version='FOTMIR '+PACKAGE_VERSION)
    p.add_argument('--check-environment',action='store_true')
    for side in ('moving','fixed'):
        p.add_argument('--'+side+'-image',type=Path)
        p.add_argument('--'+side+'-modality',default='MRI' if side=='moving' else 'CT')
    p.add_argument('--output-dir',type=Path)
    p.add_argument('--threads',type=int,default=1)
    p.add_argument('--num-samples',type=int,default=10000)
    p.add_argument('--seed',type=int,default=20260722)
    p.add_argument('--tau',type=float,default=.05)
    p.add_argument('--ratio',type=float,default=.05)
    p.add_argument('--profile',choices=('structure','legacy'),default='structure')
    return p

def cli(argv=None):
    p=parser();a=p.parse_args(argv)
    if a.threads<1 or a.num_samples<50:p.error('threads must be >=1; num-samples must be >=50')
    if not 0<a.tau<1 or not 0<a.ratio<=1:p.error('0<tau<1 and 0<ratio<=1 required')
    if not 0<=a.seed<=2**31-1:p.error('seed must be between 0 and 2**31-1')
    env=thread_environment(a.threads)
    if a.check_environment:
        code="import importlib,json; names="+repr(MODULES)+"; print(json.dumps({n:getattr(importlib.import_module(n),'__version__','available') for n in names},indent=2))"
        return subprocess.run([sys.executable,'-c',code],env=env).returncode
    for name in ('moving_image','fixed_image','output_dir'):
        if getattr(a,name) is None:p.error('--'+name.replace('_','-')+' is required')
    for name in ('moving_image','fixed_image','output_dir'):setattr(a,name,getattr(a,name).expanduser().resolve())
    for f in (a.moving_image,a.fixed_image):
        if not f.is_file():p.error('Input image not found: '+str(f))
    out=a.output_dir.expanduser().resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):p.error('Use a new or empty output directory; existing results will not be overwritten.')
    cmd=[sys.executable,'-u',str(ROOT/'_image_worker.py')]
    for name in ('moving_image','fixed_image','moving_modality','fixed_modality','output_dir','threads','num_samples','seed','tau','ratio','profile'):
        cmd+=['--'+name.replace('_','-'),str(getattr(a,name))]
    print('FOTMIR '+PACKAGE_VERSION+': starting registration; threads='+str(a.threads),flush=True)
    proc=subprocess.Popen(cmd,env=env)
    try:return proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        try:proc.wait(timeout=5)
        except subprocess.TimeoutExpired:proc.kill();proc.wait()
        return 130

if __name__=='__main__':raise SystemExit(cli())
