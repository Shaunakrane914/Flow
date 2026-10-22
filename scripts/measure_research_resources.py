"""Measure fresh extraction/solver times and trained-model forward latencies.

Fresh worker processes measure one predeclared crop per natural parent. Runtime
results never use historical estimates. CPU RSS includes the Python runtime and
libraries and is explicitly distinct from model weights or GPU activation memory.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def worker(args):
    import numpy as np
    import psutil
    import torch
    from src.research_pnm import extract_pilot_graph
    torch.set_num_threads(1)
    image=np.load(args.chunk).astype(bool)
    process=psutil.Process()
    before=process.memory_info().rss
    graph,diag=extract_pilot_graph(image,args.voxel_size)
    memory=process.memory_info()
    result={'sample_id':Path(args.chunk).stem,'shape':list(image.shape),
            'porosity':float(image.mean()),'num_nodes':graph.num_nodes,'num_directed_edges':graph.num_edges,
            'fresh_log10_k_m2':graph.y.item(),'rss_before_bytes':before,'rss_after_bytes':memory.rss,
            'process_lifetime_peak_rss_bytes':getattr(memory,'peak_wset',None),
            'diagnostics':diag}
    Path(args.worker_output).write_text(json.dumps(result,indent=2),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='experiments/topoflow_research_v2')
    parser.add_argument('--manifest',default='data/benchmark_dataset/manifest.json')
    parser.add_argument('--checkpoint-dir')
    parser.add_argument('--resource-directory',default='resources')
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--chunk')
    parser.add_argument('--voxel-size',type=float)
    parser.add_argument('--worker-output')
    args=parser.parse_args()
    if args.worker:return worker(args)
    import numpy as np
    import torch
    import psutil
    from torch_geometric.loader import DataLoader
    from src.experiment_data import load_manifest,sha256
    from src.research_evaluation import ScaleAwareRegressor
    from scripts.run_research_v2 import NEURAL
    from scripts.run_experiments import write_json,write_csv
    torch.set_num_threads(1)
    output=Path(args.output)
    resources=output/args.resource_directory
    resources.mkdir(parents=True,exist_ok=False)
    manifest,records,graphs=load_manifest(args.manifest)
    natural=[i for i,r in enumerate(records) if r['formation']!='Synthetic']
    parents=sorted({records[i]['parent_id'] for i in natural})
    prepared=[]
    # Selection is lexicographic by ID, fixed before any timing observation.
    for parent in parents:
        i=min((i for i in natural if records[i]['parent_id']==parent),key=lambda i:records[i]['id'])
        record=records[i]
        chunk=Path(args.manifest).resolve().parent/'chunks'/f"{record['id']}.npy"
        resultfile=resources/f"{record['id']}_extraction.json"
        command=[sys.executable,str(Path(__file__).resolve()),'--worker','--chunk',str(chunk),
                 '--voxel-size',str(record['voxel_size_m']),'--worker-output',str(resultfile)]
        env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        print(f"Measuring fresh extraction: {record['id']}",flush=True)
        completed=subprocess.run(command,env=env,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                  text=True,encoding='utf-8',errors='replace')
        (resources/f"{record['id']}_worker.log").write_text(completed.stdout,encoding='utf-8')
        if completed.returncode or not resultfile.exists():
            prepared.append({'sample_id':record['id'],'status':'failed','exit_code':completed.returncode})
            continue
        value=json.loads(resultfile.read_text(encoding='utf-8'))
        difference=abs(value['fresh_log10_k_m2']-record['true_log10_k_m2'])
        value.update(status='measured',original_log10_k_m2=record['true_log10_k_m2'],
                     absolute_label_difference=difference,agrees_with_saved_label=difference<1e-4)
        write_json(resultfile,value)
        prepared.append(value)
    write_json(resources/'extraction_summary.json',prepared)
    timings=[]
    for stage in ['snow2','pnm_solve','graph_conversion','total_extraction_and_reference']:
        values=[r['diagnostics']['timing_seconds'][stage]*1000 for r in prepared if r['status']=='measured']
        if values:
            timings.append({'stage':stage,'measured_crops':len(values),'measured_forward_calls':None,
                            'median_ms':float(np.median(values)),'p95_ms':float(np.percentile(values,95)),
                            'minimum_ms':float(min(values)),'maximum_ms':float(max(values)),
                            'scope':'one fresh predeclared crop per natural parent; includes reference where named'})
    ckdir=Path(args.checkpoint_dir) if args.checkpoint_dir else output/'checkpoints'
    forward_rows=[]
    raw_forward=[]
    # Trained fold checkpoints, seed 11, and each of their held-out crops.
    for name,(architecture,physics) in NEURAL.items():
        all_latencies=[]
        for parent in parents:
            files=sorted(ckdir.glob(f'{parent}_{name}_seed*.pth'))
            if not files:continue
            checkpoint=files[0]
            ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
            model=ScaleAwareRegressor(architecture,physics=physics,dropout=ck['config']['dropout']).eval()
            model.load_state_dict(ck['state_dict'])
            transform=ck['transform']
            for i in natural:
                if records[i]['parent_id']!=parent:continue
                batch=next(iter(DataLoader([transform.apply(records[i],graphs[i])],batch_size=1)))
                values=[]
                with torch.inference_mode():
                    for _ in range(10):model(batch)
                    for _ in range(30):
                        started=time.perf_counter_ns();model(batch)
                        values.append((time.perf_counter_ns()-started)/1e6)
                all_latencies.extend(values)
                raw_forward.extend({'model':name,'sample_id':records[i]['id'],'repeat':repeat,
                                    'latency_ms':latency} for repeat,latency in enumerate(values))
                forward_rows.append({'model':name,'parent_id':parent,'sample_id':records[i]['id'],
                                     'checkpoint':checkpoint.name,'checkpoint_sha256':sha256(checkpoint),
                                     'nodes':graphs[i].num_nodes,'directed_edges':graphs[i].num_edges,
                                     'forward_repeats':len(values),'median_ms':float(np.median(values)),
                                     'p95_ms':float(np.percentile(values,95)),
                                     'parameters':sum(p.numel() for p in model.parameters()),
                                     'weight_bytes':sum(p.numel()*p.element_size() for p in model.parameters()),
                                     'graph_input_tensor_bytes':sum(t.numel()*t.element_size()
                                                                  for t in [batch.x,batch.edge_index,batch.geometry])})
        if all_latencies:
            timings.append({'stage':f'{name}_forward','measured_crops':sum(r['model']==name for r in forward_rows),
                            'measured_forward_calls':len(all_latencies),
                            'median_ms':float(np.median(all_latencies)),
                            'p95_ms':float(np.percentile(all_latencies,95)),
                            'minimum_ms':float(min(all_latencies)),'maximum_ms':float(max(all_latencies)),
                            'scope':'trained model forward only; precomputed normalized graph; 10 warmups/crop'})
    write_csv(resources/'timing_results.csv',timings)
    write_csv(resources/'per_crop_forward.csv',forward_rows)
    write_csv(resources/'raw_forward_repeats.csv',raw_forward)
    paired=[]
    for extraction in prepared:
        if extraction['status']!='measured':continue
        for forward in forward_rows:
            if forward['sample_id']!=extraction['sample_id']:continue
            solver=extraction['diagnostics']['timing_seconds']['pnm_solve']*1000
            paired.append({'sample_id':extraction['sample_id'],'model':forward['model'],
                           'reference_solve_ms':solver,'median_forward_ms':forward['median_ms'],
                           'solve_to_forward_ratio':solver/forward['median_ms']})
    write_csv(resources/'paired_solve_forward.csv',paired)
    write_json(resources/'environment.json',{'created_utc':datetime.now(timezone.utc).isoformat(),
               'python':platform.python_version(),'platform':platform.platform(),'cpu':platform.processor(),
               'cpu_threads_used':1,'torch':torch.__version__,'ram_bytes':psutil.virtual_memory().total,
               'psutil':psutil.__version__,'measurement_source_sha256':sha256(__file__),
               'extractor_source_sha256':sha256(ROOT/'src/research_pnm.py'),
               'limitation':'Process lifetime RSS includes imports. No CNN benchmark or GPU peak measurement. '
               'SNOW2 already includes pore-network construction. Graph conversion only means tensor packaging. '
               'Reference solve is a label-generation cost, not required for deployed GNN prediction.'})
    lines=['# Measured compute costs','','| Stage | Crops | Median ms | P95 ms |',
           '|---|---:|---:|---:|']
    lines += [f"| {r['stage']} | {r['measured_crops']} | {r['median_ms']:.3f} | {r['p95_ms']:.3f} |" for r in timings]
    lines += ['','Measurements are serial CPU runs at one Torch/BLAS thread. Extraction uses one fixed crop '
              'per natural parent; p95 across four crops is descriptive and unstable. Model forwarding uses '
              '30 repeated calls per held-out crop after 10 warmups. Input file loading and feature normalization '
              'are excluded from forward latency. The complete extraction-and-reference row includes an '
              'unnecessary PNM label solve for deployment; it is not an end-to-end inference measurement.', '',
              'Peak process RSS, actual parameter bytes and graph tensor bytes are different quantities. '
              'No dense CNN was measured, so these data do not establish a CNN memory advantage.']
    (resources/'resource_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(f'Measurements complete: {resources}',flush=True)


if __name__=='__main__':main()
