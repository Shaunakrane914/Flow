"""Complete a GAT 2x2 normalization/descriptor ablation with nested parent validation.

Reuse the two exact compatible V2 cells; train the two missing cells with the
same grid, inner folds, seeds and epoch rule. No new external labels are used.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.experiment_data import load_manifest,sha256
from src.research_evaluation import parent_macro_mse
from scripts.run_experiments import write_csv,write_json
from scripts.run_research_v2 import CONFIGS,NEURAL,fit_neural,predict,summarize

CELLS={'gat_standard':(False,False),'gat_physics':(True,True),
       'gat_normalized_only':(True,False),'gat_descriptors_only':(False,True)}


def main():
    import csv
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',default='data/benchmark_dataset/manifest.json')
    parser.add_argument('--source',default='experiments/topoflow_research_v2')
    parser.add_argument('--output',default='experiments/topoflow_followup/ablation')
    args=parser.parse_args()
    torch.set_num_threads(1)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False);(out/'checkpoints').mkdir()
    source=Path(args.source)
    old_protocol=json.loads((source/'protocol.json').read_text())
    _,records,graphs=load_manifest(args.manifest)
    if sha256(args.manifest)!=old_protocol['manifest_sha256']:raise ValueError('Ablation manifest differs')
    natural=[i for i,r in enumerate(records) if r['formation']!='Synthetic']
    parents=sorted({records[i]['parent_id'] for i in natural})
    seeds=old_protocol['args']['seeds'];epochs=old_protocol['args']['epochs'];patience=old_protocol['args']['patience']
    protocol={'status':'running','created_utc':datetime.now(timezone.utc).isoformat(),
              'source_protocol_sha256':sha256(source/'protocol.json'),'cells':CELLS,'configs':CONFIGS,
              'seeds':seeds,'epochs':epochs,'patience':patience,'selection_seed':2026,
              'purpose':'Exploratory 2x2 physical input/target normalization and graph-descriptor study',
              'reuse':['gat_standard','gat_physics'],'new':['gat_normalized_only','gat_descriptors_only'],
              'external_labels_used':False,'manifest_sha256':sha256(args.manifest),
              'source_hashes':{str(p.relative_to(ROOT)):sha256(p) for p in [Path(__file__),ROOT/'src/research_evaluation.py',ROOT/'scripts/run_research_v2.py']}}
    write_json(out/'protocol.json',protocol)
    predictions=[]
    with (source/'predictions.csv').open(encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row['model'] not in ['gat_standard','gat_physics','uniform_physics','pore_set_physics']:continue
            row['seed']=int(row['seed'])
            for k in ['true_log10_k_m2','predicted_log10_k_m2']:row[k]=float(row[k])
            predictions.append(row)
    selections=[];started=time.perf_counter()
    for name in ['gat_normalized_only','gat_descriptors_only']:
        physics,geometry=CELLS[name];NEURAL[name]=('gat',physics)
        for outer in parents:
            train_outer=[i for i in natural if records[i]['parent_id']!=outer]
            test=[i for i in natural if records[i]['parent_id']==outer]
            remaining=[p for p in parents if p!=outer]
            candidates=[]
            for ci,config in enumerate(CONFIGS):
                scores=[];selected_epochs=[]
                for vp in remaining:
                    train=[i for i in train_outer if records[i]['parent_id']!=vp]
                    valid=[i for i in train_outer if records[i]['parent_id']==vp]
                    model,transform,info=fit_neural(name,records,graphs,train,valid,2026,config,epochs,patience,geometry=geometry)
                    transformed={i:transform.apply(records[i],graphs[i]) for i in valid}
                    score=parent_macro_mse(records,valid,predict(model,transformed,valid,transform))
                    scores.append(score);selected_epochs.append(info['selected_epoch'])
                    selections.append({'model':name,'outer_test_parent':outer,'candidate':ci,
                                       'inner_validation_parent':vp,'validation_parent_mse':score,
                                       'best_epoch':info['selected_epoch'],'train_parents':'|'.join(p for p in remaining if p!=vp)})
                candidates.append((float(np.mean(scores)),ci,max(1,int(np.rint(np.median(selected_epochs))))))
            score,ci,fixed=min(candidates,key=lambda a:(a[0],a[1]))
            write_json(out/f'selection_{outer}_{name}.json',{'candidate':ci,'config':CONFIGS[ci],
                       'inner_parent_mse':score,'epochs':fixed,'physics':physics,'geometry':geometry})
            for seed in seeds:
                model,transform,_=fit_neural(name,records,graphs,train_outer,[],seed,CONFIGS[ci],epochs,patience,
                                             fixed_epochs=fixed,geometry=geometry)
                transformed={i:transform.apply(records[i],graphs[i]) for i in test}
                values=predict(model,transformed,test,transform)
                torch.save({'model':name,'physics':physics,'geometry':geometry,'state_dict':model.state_dict(),
                            'transform':transform,'config':CONFIGS[ci],'epochs':fixed,'seed':seed,
                            'outer_test_parent':outer,'train_parents':remaining},out/'checkpoints'/f'{outer}_{name}_seed{seed}.pth')
                predictions.extend({'sample_id':records[i]['id'],'parent_id':outer,'model':name,'seed':seed,
                                    'selection':name,'true_log10_k_m2':records[i]['true_log10_k_m2'],
                                    'predicted_log10_k_m2':float(v)} for i,v in zip(test,values))
            write_csv(out/'predictions.csv',predictions);write_csv(out/'inner_selection.csv',selections)
            print(f'{name}: held-out {outer}, selected epochs={fixed}, elapsed={time.perf_counter()-started:.1f}s',flush=True)
    summarize(predictions,out,seeds)
    effects=[]
    # Difference of MSE effects is a descriptive factorial interaction, not a causal population estimate.
    for parent in parents:
        def loss(name):
            rows=[r for r in predictions if r['parent_id']==parent and r['model']==name]
            return float(np.mean([(r['predicted_log10_k_m2']-r['true_log10_k_m2'])**2 for r in rows]))
        a,b,c,d=(loss(n) for n in ['gat_standard','gat_normalized_only','gat_descriptors_only','gat_physics'])
        effects.append({'parent_id':parent,'normalization_mse_change_no_descriptors':b-a,
                        'normalization_mse_change_with_descriptors':d-c,
                        'descriptors_mse_change_without_normalization':c-a,
                        'descriptors_mse_change_with_normalization':d-b,
                        'factorial_interaction_mse':d-c-b+a})
    write_csv(out/'factorial_effects.csv',effects)
    protocol.update(status='complete',elapsed_seconds=time.perf_counter()-started)
    write_json(out/'protocol.json',protocol)
    print(f'Completed controlled ablation: {out}',flush=True)


if __name__=='__main__':main()
