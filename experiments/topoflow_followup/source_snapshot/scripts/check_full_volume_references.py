"""Compare two exact full image domains to published header reference values.

The reference boundary settings are incompletely documented; results are a
cross-method consistency check, not a certified matched DNS validation.
"""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def worker(args):
    import numpy as np
    import torch
    import psutil
    from scripts.prepare_locked_external import read_source
    from src.research_pnm import extract_pilot_graph
    from scripts.run_experiments import write_json
    torch.set_num_threads(1)
    with zipfile.ZipFile(ROOT/'data/public/ImagesICPSC2007.zip') as z:
        header,raw,voxel=read_source(z,args.worker)
    image=(raw==0);del raw
    graph,diag=extract_pilot_graph(image,voxel)
    directory=ROOT/'data/full_volume_reference';directory.mkdir(exist_ok=True,parents=True)
    torch.save(graph,directory/f'{args.worker}.pt')
    write_json(Path(args.result),{'name':args.worker,'shape_zyx':list(image.shape),'voxel_size_m':voxel,
               'porosity':float(image.mean()),'num_nodes':graph.num_nodes,'num_directed_edges':graph.num_edges,
               'pnm_k_m2':float(10**graph.y.item()),'graph_path':str(directory/f'{args.worker}.pt'),
               'process_peak_rss_bytes':getattr(psutil.Process().memory_info(),'peak_wset',None),
               'diagnostics':diag})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='experiments/topoflow_followup/full_reference')
    parser.add_argument('--worker',choices=['S1','S5'])
    parser.add_argument('--result')
    args=parser.parse_args()
    if args.worker:return worker(args)
    import psutil
    from src.experiment_data import sha256
    from scripts.run_experiments import write_json
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    write_json(out/'protocol.json',{'created_utc':datetime.now(timezone.utc).isoformat(),
       'preselected_full_scans':['S1','S5'],'source':'official ImageICPSC2007 exact binary volumes',
       'nominal_direction':'numpy axis 0 is source z, compare published Kz',
       'reference_conditions':'Full voxelized-image and network K values reported separately in archive headers; exact boundary convention not independently matched.',
       'not_a_crop_reference':True,'worker_memory_limit_bytes':2_500_000_000,'worker_timeout_seconds':300,
       'source_sha256':sha256(__file__)})
    results=[]
    for name in ['S1','S5']:
        result=out/f'{name}_pnm.json'
        command=[sys.executable,str(Path(__file__).resolve()),'--worker',name,'--result',str(result.resolve())]
        with (out/f'{name}_worker.log').open('w',encoding='utf-8') as log:
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT,
                    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
            started=time.perf_counter();peak=0;failure=None
            while process.poll() is None:
                try:peak=max(peak,psutil.Process(process.pid).memory_info().rss)
                except psutil.NoSuchProcess:break
                if peak>2_500_000_000:failure='worker_memory_limit';process.kill();break
                if time.perf_counter()-started>300:failure='worker_timeout';process.kill();break
                time.sleep(.2)
            code=process.wait()
        row={'name':name,'exit_code':code,'observed_peak_rss_bytes':peak,
             'elapsed_seconds':time.perf_counter()-started,'failure':failure,
             'status':'complete' if code==0 and result.exists() else 'not_completed'}
        results.append(row);write_json(out/'execution.json',results)
        print(f'Full-volume reference extraction {name}: {row["status"]}, peak MiB={peak/1024**2:.1f}',flush=True)


if __name__=='__main__':main()
