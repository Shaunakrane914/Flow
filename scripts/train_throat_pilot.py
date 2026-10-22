"""Frozen-parameter critical-throat pilot with external-parent evaluation."""
from copy import deepcopy
import argparse
import csv
from datetime import datetime,timezone
from pathlib import Path
import json
import sys
import time
import numpy as np
from scipy.stats import spearmanr
import torch
from torch_geometric.loader import DataLoader

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.critical_throat_model import CriticalThroatRegressor,fractional_response
from src.experiment_data import sha256
from scripts.run_experiments import set_seed,write_json,write_csv

MODELS=['edge_gat','guided_edge_gat','edge_sage','edge_mlp','guided_edge_sage','guided_edge_mlp']


def score(model,graphs,indices):
    model.eval();values=[]
    with torch.inference_mode():
        for i in indices:
            batch=next(iter(DataLoader([graphs[i]],batch_size=1)))
            s,l=model(batch);response=fractional_response(s,l)
            values.append(float(((response[batch.selected_mask]-batch.response_target[batch.selected_mask])**2).mean()))
    return float(np.mean(values))


def train(name,graphs,train_indices,validation,seed,fixed_epochs=None):
    set_seed(seed)
    model=CriticalThroatRegressor(name.removeprefix('guided_'))
    opt=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=1e-4)
    loader=DataLoader([graphs[i] for i in train_indices],batch_size=2,shuffle=True,
                       generator=torch.Generator().manual_seed(seed))
    best=float('inf');best_epoch=0;state=None;stale=0
    for epoch in range(1,(fixed_epochs or 80)+1):
        model.train()
        for batch in loader:
            opt.zero_grad(set_to_none=True)
            s,l=model(batch);response=fractional_response(s,l)
            # Each graph contributes six selected throats at each of three doses.
            loss=((response[batch.selected_mask]-batch.response_target[batch.selected_mask])**2).mean()
            if name.startswith('guided_'):
                group=batch.batch[batch.throat_index[0]]
                terms=-batch.sensitivity*torch.log(s.clamp_min(1e-12))
                guidance=torch.stack([terms[group==g].sum() for g in range(batch.num_graphs)]).mean()
                loss=loss+.05*guidance
            if not torch.isfinite(loss):raise ValueError('Nonfinite pilot loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
        if fixed_epochs:continue
        error=score(model,graphs,validation)
        if error<best:
            best=error;best_epoch=epoch;state=deepcopy(model.state_dict());stale=0
        else:stale+=1
        if stale>=15:break
    if fixed_epochs is None:model.load_state_dict(state)
    return model,(fixed_epochs or best_epoch),best


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(1)
    root=ROOT/'experiments/topoflow_followup/throat_aware_pilot'
    out=root/'training';out.mkdir(parents=True,exist_ok=args.resume)
    (out/'checkpoints').mkdir(exist_ok=args.resume)
    inventory=json.loads((ROOT/'data/throat_pilot/manifest.json').read_text())
    graphs=[torch.load(r['graph'],map_location='cpu',weights_only=False) for r in inventory]
    development=[i for i,r in enumerate(inventory) if r['role']=='development']
    external=[i for i,r in enumerate(inventory) if r['role']=='external']
    parents=sorted({inventory[i]['parent_id'] for i in development})
    assert not set(parents)&{inventory[i]['parent_id'] for i in external}
    seeds=[11,22,33,44,55]
    protocol={'status':'running','locked_utc':datetime.now(timezone.utc).isoformat(),
       'models':MODELS,'seeds':seeds,'learning_rate':.003,'weight_decay':1e-4,'epochs_max':80,'patience':15,
       'guidance_weight':.05,'primary_model':'guided_edge_gat',
       'primary_metric':'per-graph mean absolute fractional permeability-loss error over three high and three random internal throats at 10/25/50 percent diameter restriction',
       'epoch_selection':'leave-one-development-parent-out, selection seed 2026; median selected epoch for final full-development fit',
       'development_graphs':len(development),'development_parents':parents,
       'external_graphs':len(external),'external_parents':[inventory[i]['parent_id'] for i in external],
       'target':'PNM counterfactual response; no voxel-flow or real-clogging validation',
       'model_output':'global hydraulic importance head and bounded edge leverage; not native GAT attention coefficients',
       'readout':'loss=-(r_g-1)*s/(1+(r_g-1)*l); r_g=(1-diameter_restriction)^4',
       'input':'geometry/boundary/connectivity only; pressures/flows/sensitivities never read during forward',
       'manifest_sha256':sha256(ROOT/'data/throat_pilot/manifest.json'),
       'source_hashes':{str(p.relative_to(ROOT)):sha256(p) for p in [Path(__file__),ROOT/'src/critical_throat_model.py']}}
    if args.resume and (out/'protocol.json').exists():
        previous=json.loads((out/'protocol.json').read_text())
        if previous['manifest_sha256']!=protocol['manifest_sha256']:raise ValueError('Resume manifest changed')
        previous['resume_source_hashes']=protocol['source_hashes']
        previous['resume_reason']='Corrected unavailable torch.flatnonzero export API to torch.nonzero; model and selected epochs unchanged.'
        if previous.get('status')=='complete':
            previous['posthoc_controls']=['guided_edge_sage','guided_edge_mlp']
            previous['posthoc_control_scope']='Added after the first external pilot results to separate guidance from architecture; exploratory test-set reuse, no hyperparameter changes.'
            previous['models']=MODELS
        previous['status']='running'
        protocol=previous
    write_json(out/'protocol.json',protocol)
    selected=json.loads((out/'locked_epochs.json').read_text()) if args.resume and (out/'locked_epochs.json').exists() else {}
    validation_rows=list(csv.DictReader((out/'development_validation.csv').open(encoding='utf-8'))) if args.resume and (out/'development_validation.csv').exists() else []
    for name in MODELS:
        if name in selected:continue
        epochs=[]
        for vp in parents:
            train_indices=[i for i in development if inventory[i]['parent_id']!=vp]
            valid=[i for i in development if inventory[i]['parent_id']==vp]
            model,epoch,error=train(name,graphs,train_indices,valid,2026)
            epochs.append(epoch);validation_rows.append({'model':name,'validation_parent':vp,'best_epoch':epoch,'validation_mse':error})
        selected[name]=max(1,int(np.rint(np.median(epochs))))
        write_csv(out/'development_validation.csv',validation_rows)
        print(f'Chosen without external labels: {name}, epochs={selected[name]}',flush=True)
    write_json(out/'locked_epochs.json',selected)
    rows=[];rankings=[];latencies=[]
    for name in MODELS+['no_response','conductance_proxy']:
        for seed in seeds:
            if name in MODELS:
                checkpoint=out/'checkpoints'/f'{name}_seed{seed}.pth'
                if args.resume and checkpoint.exists():
                    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
                    if ck['epochs']!=selected[name]:raise ValueError('Cached checkpoint epochs differ')
                    model=CriticalThroatRegressor(name.removeprefix('guided_'));model.load_state_dict(ck['state_dict'])
                else:model,_,_=train(name,graphs,development,[],seed,fixed_epochs=selected[name])
                model.eval()
                torch.save({'model':name,'state_dict':model.state_dict(),'epochs':selected[name],'seed':seed,
                            'parameters':sum(p.numel() for p in model.parameters())},checkpoint)
            for i in external:
                graph=graphs[i];batch=next(iter(DataLoader([graph],batch_size=1)))
                with torch.inference_mode():
                    if name in MODELS:
                        started=time.perf_counter_ns();s,l=model(batch);response=fractional_response(s,l)
                        latency=(time.perf_counter_ns()-started)/1e6
                        latencies.append({'model':name,'seed':seed,'sample_id':inventory[i]['sample_id'],'unwarmed_forward_ms':latency})
                    elif name=='conductance_proxy':
                        s=torch.softmax(graph.throat_attr[:,2]*np.log(10),dim=0);l=(1-s)*.5
                        response=fractional_response(s,l)
                    else:s=torch.ones_like(graph.sensitivity)/len(graph.sensitivity);response=torch.zeros_like(graph.response_target)
                internal=graph.internal_mask
                rho=spearmanr(s[internal].numpy(),graph.sensitivity[internal].numpy()).statistic
                n=max(1,int(np.ceil(int(internal.sum())*.1)))
                overlap=len(set(torch.argsort(s[internal])[-n:].tolist()) & set(torch.argsort(graph.sensitivity[internal])[-n:].tolist()))/n
                rankings.append({'model':name,'seed':seed,'sample_id':inventory[i]['sample_id'],
                                 'parent_id':inventory[i]['parent_id'],'spearman_hydraulic_importance':float(rho) if np.isfinite(rho) else None,
                                 'top_10_percent_overlap':float(overlap)})
                for edge in torch.nonzero(graph.selected_mask,as_tuple=True)[0].tolist():
                    for j,dose in enumerate([.1,.25,.5]):
                        rows.append({'model':name,'seed':seed,'sample_id':inventory[i]['sample_id'],
                                     'parent_id':inventory[i]['parent_id'],'throat_index':edge,'diameter_reduction':dose,
                                     'selection_group':'high_dissipation' if graph.high_mask[edge] else 'random',
                                     'true_fractional_permeability_loss':float(graph.response_target[edge,j]),
                                     'predicted_fractional_permeability_loss':float(response[edge,j])})
        write_csv(out/'response_predictions.csv',rows);write_csv(out/'importance_rankings.csv',rankings)
        write_csv(out/'forward_observations.csv',latencies)
        print('Completed external critical-throat inference:',name,flush=True)
    metrics=[]
    for name in MODELS+['no_response','conductance_proxy']:
        for group in ['all','high_dissipation','random']:
            values=[]
            for seed in seeds:
                subset=[r for r in rows if r['model']==name and r['seed']==seed and (group=='all' or r['selection_group']==group)]
                values.append(np.mean([abs(r['predicted_fractional_permeability_loss']-r['true_fractional_permeability_loss']) for r in subset]))
            rank=[r for r in rankings if r['model']==name and r['spearman_hydraulic_importance'] is not None]
            metrics.append({'model':name,'group':group,'external_parents':len(external),'external_graphs':len(external),
                            'mean_absolute_fractional_loss_error':float(np.mean(values)),
                            'seed_std':float(np.std(values,ddof=1)),
                            'mean_spearman_importance':float(np.mean([r['spearman_hydraulic_importance'] for r in rank])) if rank else None})
    write_csv(out/'metrics.csv',metrics)
    protocol.update(status='complete');write_json(out/'protocol.json',protocol)
    print('Critical-throat pilot complete.',flush=True)


if __name__=='__main__':main()
