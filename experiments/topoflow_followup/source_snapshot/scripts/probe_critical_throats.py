"""Counterfactual throat restrictions and attention-faithfulness feasibility study.

The physical reference is the declared PNM closure, not experimental clogging
data. Parent counts remain four; interventions are dependent observations.
"""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from scipy.stats import spearmanr
from torch_geometric.loader import DataLoader

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.experiment_data import load_manifest,sha256
from src.research_pnm import extract_pilot_graph,solve_permeability
from src.research_evaluation import ScaleAwareRegressor
from scripts.run_experiments import write_csv,write_json


def network_solve(net,diameter,length,shape,voxel,conns=None,fields=False):
    return solve_permeability(net['pore.coords'],net['throat.conns'] if conns is None else conns,
            diameter,length,net['pore.xmin'],net['pore.xmax'],shape,voxel,return_fields=fields)


def attention_scores(model,batch,physical_conns):
    from torch.nn import functional as F
    encoder=model.encoder
    with torch.inference_mode():
        x=encoder.dropout(F.elu(encoder.gat1(batch.x,batch.edge_index)))
        x=encoder.dropout(F.elu(encoder.gat2(x,batch.edge_index)))
        _,(edges,alpha)=encoder.gat3(x,batch.edge_index,return_attention_weights=True)
    values=alpha.mean(1).numpy();lookup={tuple(e):float(v) for e,v in zip(edges.T.tolist(),values)}
    learned=np.asarray([(lookup.get((int(a),int(b)),0)+lookup.get((int(b),int(a)),0))/2
                        for a,b in physical_conns])
    degree=np.bincount(batch.edge_index[1].numpy(),minlength=batch.num_nodes)+1
    uniform=np.asarray([(.5/degree[a]+.5/degree[b]) for a,b in physical_conns])
    return learned,uniform


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',default='data/benchmark_dataset/manifest.json')
    parser.add_argument('--checkpoints',default='experiments/topoflow_research_v2/checkpoints')
    parser.add_argument('--output',default='experiments/topoflow_followup/critical_throats')
    args=parser.parse_args()
    torch.set_num_threads(1)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    _,records,graphs=load_manifest(args.manifest)
    natural=[i for i,r in enumerate(records) if r['formation']!='Synthetic']
    parents=sorted({records[i]['parent_id'] for i in natural})
    chosen=[min((i for i in natural if records[i]['parent_id']==p),key=lambda i:records[i]['id']) for p in parents]
    protocol={'status':'running','created_utc':datetime.now(timezone.utc).isoformat(),
              'samples':[records[i]['id'] for i in chosen],'parents':parents,
              'selection':'lexicographically first accepted crop per natural parent',
              'high_selection':'top three internal physical throats by base PNM dissipation fraction',
              'random_selection':'three random other internal physical throats; fixed seed 20261022',
              'restrictions':[0.10,0.25,0.50,1.0],
              'full_blockage':'delete throat; disconnected spanning path means k=0, not an invalid log label',
              'reference':'same fixed-pressure cylindrical-throat PNM closure; length held constant',
              'sensitivity':'d log(k) / d log(g_e) = g_e*(p_i-p_j)^2/(Q*DeltaP)',
              'attention':'last GAT layer, mean of two directed coefficients, self loops excluded',
              'attention_limitation':'Local attention normalization differs from global physical dissipation; ranking is diagnostic, not a causal explanation.',
              'manifest_sha256':sha256(args.manifest),'independent_rocks':len(parents),
              'source_hashes':{str(p.relative_to(ROOT)):sha256(p) for p in [Path(__file__),ROOT/'src/research_pnm.py']}}
    write_json(out/'protocol.json',protocol)
    observations=[];ranking=[];checks=[];rng=np.random.default_rng(20261022)
    for i in chosen:
        record=records[i];sid=record['id'];parent=record['parent_id']
        image=np.load(Path(args.manifest).resolve().parent/'chunks'/f'{sid}.npy').astype(bool)
        graph,diag,net=extract_pilot_graph(image,record['voxel_size_m'],return_network=True)
        diameter=np.asarray(net['throat.inscribed_diameter'],dtype=np.float64).copy()
        length=np.asarray(net['throat.direct_length'],dtype=np.float64).copy()
        conns=np.asarray(net['throat.conns']);physical=~net['pore.boundary']
        eligible=np.flatnonzero(physical[conns].all(axis=1))
        base,details=network_solve(net,diameter,length,image.shape,record['voxel_size_m'],fields=True)
        sensitivity=details.pop('fields')['dissipation_fraction']
        assert abs(np.log10(base)-record['true_log10_k_m2'])<1e-4
        assert abs(sensitivity.sum()-1)<1e-6
        # Verify the analytic log-conductance sensitivity by a finite difference.
        edge=int(eligible[np.argmax(sensitivity[eligible])]);eps=1e-5
        modified=diameter.copy();modified[edge]*=np.exp(eps/4)
        shifted,_=network_solve(net,modified,length,image.shape,record['voxel_size_m'])
        finite=float(np.log(shifted/base)/eps)
        assert abs(finite-sensitivity[edge])<2e-4
        remap=np.full(len(physical),-1,dtype=int);remap[physical]=np.arange(physical.sum())
        pc=remap[conns[eligible]]
        models={}
        for name,physics in [('gat_standard',False),('gat_physics',True)]:
            ckpath=Path(args.checkpoints)/f'{parent}_{name}_seed11.pth'
            ck=torch.load(ckpath,map_location='cpu',weights_only=False)
            model=ScaleAwareRegressor('gat',physics=physics,dropout=ck['config']['dropout']).eval()
            model.load_state_dict(ck['state_dict']);transform=ck['transform']
            batch=next(iter(DataLoader([transform.apply(record,graph)],batch_size=1)))
            with torch.inference_mode():basepred=float(model(batch).item()*transform.target_std+transform.target_mean+batch.offset.item())
            models[name]=(model,transform,basepred)
            learned,uniform=attention_scores(model,batch,pc)
            for method,score in [('learned_attention',learned),('uniform_incoming',uniform),
                                 ('diameter_fourth_over_length',diameter[eligible]**4/length[eligible])]:
                physical_score=sensitivity[eligible];top=max(1,int(np.ceil(.1*len(eligible))))
                overlap=len(set(np.argsort(score)[-top:]) & set(np.argsort(physical_score)[-top:]))/top
                rho=spearmanr(score,physical_score).statistic
                ranking.append({'sample_id':sid,'parent_id':parent,'model':name,'method':method,
                                'internal_throats':len(eligible),'spearman_dissipation':float(rho) if np.isfinite(rho) else None,
                                'top_10_percent_overlap':float(overlap)})
        high=eligible[np.argsort(sensitivity[eligible],kind='stable')[-3:]]
        pool=np.setdiff1d(eligible,high);random=rng.choice(pool,min(3,len(pool)),replace=False)
        for group,edges in [('high_dissipation',high),('random',random)]:
            for edge in edges:
                for restriction in protocol['restrictions']:
                    changed=diameter.copy();changed[edge]*=1-restriction
                    predicted_graph=graph.clone()
                    if restriction==1.0:
                        keep=np.arange(len(conns))!=edge
                        try:
                            truth,changed_diag=network_solve(net,diameter[keep],length[keep],image.shape,
                                                            record['voxel_size_m'],conns=conns[keep])
                            disconnected=False
                        except ValueError as exc:
                            if str(exc)!='Non-percolating network':raise
                            truth=0.;disconnected=True
                        a,b=remap[conns[edge]]
                        keep_graph=~(((graph.edge_index[0]==a)&(graph.edge_index[1]==b))|
                                     ((graph.edge_index[0]==b)&(graph.edge_index[1]==a)))
                        predicted_graph.edge_index=graph.edge_index[:,keep_graph]
                    else:
                        truth,changed_diag=network_solve(net,changed,length,image.shape,record['voxel_size_m'])
                        disconnected=False
                    # Restricting/removing a passive conductance cannot increase effective permeability.
                    assert truth<=base*(1+1e-8)
                    for name,(model,transform,basepred) in models.items():
                        batch=next(iter(DataLoader([transform.apply(record,predicted_graph)],batch_size=1)))
                        with torch.inference_mode():pred=float(model(batch).item()*transform.target_std+transform.target_mean+batch.offset.item())
                        delta=pred-basepred
                        if restriction<1:assert delta==0.,'Node-only model unexpectedly reads throat diameter'
                        observations.append({'sample_id':sid,'parent_id':parent,'model':name,
                                             'selection_group':group,'throat_index':int(edge),
                                             'diameter_reduction_fraction':restriction,
                                             'base_dissipation_fraction':float(sensitivity[edge]),
                                             'reference_k_m2':base,'perturbed_k_m2':truth,
                                             'reference_fractional_permeability_loss':float(1-truth/base),
                                             'reference_delta_log10_k':float(np.log10(truth/base)) if truth>0 else None,
                                             'predicted_delta_log10_k':delta,
                                             'predicted_fractional_permeability_loss':float(1-10**delta),
                                             'nonpercolating_after_blockage':disconnected})
        checks.append({'sample_id':sid,'parent_id':parent,'base_log_label_agrees':True,
                       'sum_dissipation_fraction':float(sensitivity.sum()),'finite_difference_sensitivity':finite,
                       'analytic_sensitivity':float(sensitivity[eligible[np.argmax(sensitivity[eligible])]]),
                       'relative_mass_imbalance':details['relative_mass_imbalance']})
        write_csv(out/'interventions.csv',observations);write_csv(out/'attention_rankings.csv',ranking)
        write_json(out/'physics_checks.json',checks)
        print(f'Completed interventions and attention ranking: {sid}',flush=True)
    protocol['status']='complete';write_json(out/'protocol.json',protocol)
    summary=[]
    for parent in parents:
        for group in ['high_dissipation','random']:
            for restriction in protocol['restrictions']:
                rows=[r for r in observations if r['parent_id']==parent and r['model']=='gat_physics'
                      and r['selection_group']==group and r['diameter_reduction_fraction']==restriction]
                summary.append({'parent_id':parent,'selection_group':group,'diameter_reduction_fraction':restriction,
                                'throats':len(rows),'mean_fractional_permeability_loss':float(np.mean([r['reference_fractional_permeability_loss'] for r in rows])),
                                'mean_absolute_delta_log10_error':float(np.mean([abs(r['reference_delta_log10_k']-r['predicted_delta_log10_k']) for r in rows if r['reference_delta_log10_k'] is not None]))
                                 if any(r['reference_delta_log10_k'] is not None for r in rows) else None})
    write_csv(out/'parent_summary.csv',summary)
    lines=['# Critical throat feasibility experiment','','The existing GAT is exactly insensitive to partial throat '
           'diameter changes because those changes alter none of its inputs. The same limitation holds for '
           'the V2 physics-and-descriptor GAT. Full throat removal changes connectivity and is evaluated separately.', '',
           'One predeclared accepted crop per natural parent is tested. Within each graph, three high-dissipation '
           'and three other random internal throats are restricted by 10%, 25%, 50% and 100%. '
           'The 100% intervention deletes a throat; nonpercolating outcomes have k=0 and no log label.', '',
           'PNM hydraulic dissipation fractions sum to one. Analytic conductance sensitivities are checked '
           'by finite differences, all saved base labels are checked against fresh solves, and effective '
           'permeability is checked to be nonincreasing under conductance restrictions.', '',
           '| Parent | Selection | Diameter reduction | Mean permeability loss |',
           '|---|---|---:|---:|']
    for r in summary:lines.append(f"| {r['parent_id']} | {r['selection_group']} | {100*r['diameter_reduction_fraction']:.0f}% | {100*r['mean_fractional_permeability_loss']:.2f}% |")
    lines+=['','Attention rankings compare local GAT coefficients with global PNM sensitivity. These are '
            'different normalizations; correlation is an exploratory faithfulness diagnostic. Conductance-only '
            'and uniform-incoming rankings are controls. The high-dissipation intervention selector uses a '
            'base PNM solve and is an oracle experimental selector, not a solver-free prediction method.', '',
            'This is a four-graph, four-parent feasibility study. Perturbations do not increase the number '
            'of independent specimens, and all physical conclusions are conditional on the PNM closure. '
            'No real clogging experiment or voxel-resolved validation was performed.']
    (out/'feasibility_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':main()
