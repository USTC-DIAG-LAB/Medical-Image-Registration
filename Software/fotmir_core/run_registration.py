"""Run single or batch registration on NPZ point-cloud inputs."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import uuid


PACKAGE_VERSION = (Path(__file__).resolve().parents[2] / 'VERSION').read_text(encoding='utf-8').strip()


FIELDS = ['version','dataset','case_name','method','implementation','status','input_path','output_path',
          'seed','tau','ratio','elapsed_time','wall_seconds','rmse','message']


def configure_threads(threads):
    if threads < 1:
        raise ValueError('threads must be a positive integer.')
    # Called before importing NumPy. This controls BLAS threads, not case concurrency.
    for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
        os.environ[name] = str(threads)


def _write_summary(path, rows):
    with tempfile.NamedTemporaryFile(mode='w',dir=path.parent,suffix='.csv',delete=False,encoding='utf-8',newline='') as f:
        temporary = Path(f.name)
        writer=csv.DictWriter(f,fieldnames=FIELDS);writer.writeheader();writer.writerows(rows)
    os.replace(temporary,path)


def run(input_path, output_dir, *, tau=.05, ratio=.05, seed=20260716, overwrite=False,
        initial_indices=None, fail_fast=False, threads=1):
    """Return per-case records; do not overwrite existing .npz files by default.

    Directory runs derive each case's seed from base seed + relative path so the
    result is independent of traversal order or resuming a partial run.
    """
    import numpy as np
    if int(threads) != threads or threads < 1:
        raise ValueError('threads must be a positive integer.')
    if __package__:
        from .main import main
    else:
        from main import main
    source, target = Path(input_path).expanduser().resolve(), Path(output_dir).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    if not np.isfinite(tau) or not 0 < tau < 1 or not np.isfinite(ratio) or not 0 < ratio <= 1:
        raise ValueError('Require 0 < tau < 1 and 0 < ratio <= 1.')
    if int(seed)!=seed or not 0<=seed<=2**31-1:
        raise ValueError('seed must be an integer in [0, 2**31-1].')
    single=source.is_file()
    if single and source.suffix.lower()!='.npz':
        raise ValueError('Input must be an NPZ point-cloud file or a directory of these files.')
    if not single and (target==source or source in target.parents):
        raise ValueError('Place the output directory outside the batch input directory.')
    if not single and initial_indices is not None:
        raise ValueError('Use --initial-indices for a single file; use --seed for a batch.')
    files=[source] if single else sorted(p for p in source.rglob('*') if p.is_file() and p.suffix.lower()=='.npz')
    if not files:
        raise ValueError('No NPZ files found.')
    jobs=[];seen=set()
    for path in files:
        dataset,case=path.parent.name,path.stem
        out=target/'python'/dataset/f'{case}_result.npz'
        if out in seen:
            raise ValueError(f'Duplicate dataset and case names would produce the same output: {out}')
        seen.add(out)
        key=path.name if single else path.relative_to(source).as_posix()
        case_seed=int(seed) if single else int.from_bytes(hashlib.sha256(f'{seed}:{key}'.encode()).digest()[:8],'big')%(2**31-1)+1
        jobs.append((path,out,dataset,case,case_seed))
    target.mkdir(parents=True,exist_ok=True)
    run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
    logs=target/'logs';logs.mkdir(exist_ok=True)
    metadata={'version':PACKAGE_VERSION,'run_id':run_id,'input':str(source),'output_dir':str(target),'tau':tau,'ratio':ratio,
              'base_seed':seed,'overwrite':overwrite,'initial_indices':initial_indices,'threads':threads,
              'coreset_distance_mode':'streaming',
              'core_source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).resolve().parent.glob('*.py')},
              'note':'method=python identifies this implementation.'}
    (logs/f'{run_id}.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    rows=[]
    for path,out,dataset,case,case_seed in jobs:
        row=dict.fromkeys(FIELDS,'')
        row.update(version=PACKAGE_VERSION,dataset=dataset,case_name=case,method='python',implementation='python',
                   input_path=str(path),output_path=str(out),seed=case_seed,tau=tau,ratio=ratio)
        started=time.perf_counter();temporary=None
        try:
            with np.load(path, allow_pickle=False) as archive:
                names=set(archive.files)
            if not {'A','B'}<=names:
                row.update(status='ignored',message='The file has no A/B point-cloud fields.',output_path='')
            elif out.exists() and not overwrite:
                # Do not claim an old file matches the requested parameters.
                row.update(status='skipped_existing',message='Existing output retained.')
            else:
                out.parent.mkdir(parents=True,exist_ok=True)
                fd,name=tempfile.mkstemp(dir=out.parent,suffix='.npz');os.close(fd);temporary=Path(name)
                result=main(path,temporary,'manual',tau,ratio,seed=case_seed,initial_indices=initial_indices,threads=threads)
                # Publish only a fully written result file.
                os.replace(temporary,out);temporary=None
                row.update(status='success',elapsed_time=result['elapsed_time'],rmse=result['run_scores'])
        except Exception as exc:
            row.update(status='failed',message=f'{type(exc).__name__}: {exc}',output_path='')
        finally:
            if temporary is not None:temporary.unlink(missing_ok=True)
        row['wall_seconds']=time.perf_counter()-started
        rows.append(row)
        _write_summary(logs/f'{run_id}.csv',rows)
        _write_summary(target/'summary.csv',rows)
        print(f"[{len(rows)}/{len(jobs)}] {dataset}/{case}: {row['status']}"+ (f" — {row['message']}" if row['message'] else ''),flush=True)
        if fail_fast and row['status']=='failed':break
    return rows


def cli():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--version',action='version',version='FOTMIR '+PACKAGE_VERSION)
    p.add_argument('--input',required=True,type=Path,help='An NPZ file or a directory searched recursively for point clouds.')
    p.add_argument('--output-dir',required=True,type=Path,help='Output directory for registration results.')
    p.add_argument('--tau',type=float,default=.05)
    p.add_argument('--ratio',type=float,default=.05)
    p.add_argument('--seed',type=int,default=20260716)
    p.add_argument('--threads',type=int,default=1,help='BLAS/OpenMP thread count; cases run sequentially; default: 1.')
    p.add_argument('--overwrite',action='store_true',help='Recalculate and overwrite existing outputs.')
    p.add_argument('--fail-fast',action='store_true')
    p.add_argument('--initial-indices',nargs=2,type=int,help='Two one-based initial center indices for a single input file.')
    a=p.parse_args()
    try:
        configure_threads(a.threads)
        rows=run(a.input,a.output_dir,tau=a.tau,ratio=a.ratio,seed=a.seed,overwrite=a.overwrite,
                 initial_indices=a.initial_indices,fail_fast=a.fail_fast,threads=a.threads)
    except (ValueError,FileNotFoundError) as exc:
        p.error(str(exc))
    counts={status:sum(row['status']==status for row in rows) for status in ('success','failed','ignored','skipped_existing')}
    print(json.dumps(counts,ensure_ascii=False))
    print(f'Summary file: {a.output_dir.resolve()/"summary.csv"}')
    return 1 if counts['failed'] or not (counts['success']+counts['skipped_existing']) else 0


if __name__=='__main__':
    raise SystemExit(cli())
