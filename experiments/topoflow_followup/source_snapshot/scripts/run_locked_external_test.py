"""Train on the four development scans and evaluate once on four locked new scans."""
import argparse
from datetime import datetime,timezone
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
from scripts.run_research_v2 import CONFIGS,NEURAL,BASELINES,fit_neural,predict,baseline,summarize

CELLS={'gat_standard':(False,False),'gat_physics':(True,True),
       'gat_normalized_only':(True,False),'gat_descriptors_only':(False,True),
       'uniform_physics':(True,True),'pore_set_physics':(True,True)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',default='data/locked_external/manifest.json')
    parser.add_argument('--output',default='experiments/topoflow_followup/external_test')
    args=parser.parse_args();torch.set_num_threads(1)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False);(out/'checkpoints').mkdir()
    _,records,graphs=load_manifest(args.manifest)
    development=[i for i,r in enumerate(records) if not r['parent_id'].startswith('external_')]
    test=[i for i,r in enumerate(records) if r['parent_id'].startswith('external_')]
    parents=sorted({records[i]['parent_id'] for i in development})
    testparents=sorted({records[i]['parent_id'] for i in test})
    lock=json.loads((ROOT/'experiments/topoflow_followup/external_protocol.json').read_text())
    assert testparents==['external_'+s for s in lock['external_parents']]
    seeds=[11,22,33,44,55]
    protocol={'status':'running','created_utc':datetime.now(timezone.utc).isoformat(),
              'development_parents':parents,'test_parents':testparents,'development_crops':len(development),'test_crops':len(test),
              'primary_fixed_model':'gat_physics; chosen using development results before external evaluation',
              'configs':CONFIGS,'cells':CELLS,'seeds':seeds,'epochs':120,'patience':20,'selection_seed':2026,
              'selection':'leave-one-development-parent-out; four validation folds; no external labels',
              'final_epoch_rule':'median of selected development-fold best epochs',
              'external_protocol_sha256':sha256(ROOT/'experiments/topoflow_followup/external_protocol.json'),
              'manifest_sha256':sha256(args.manifest),
              'source_hashes':{str(p.relative_to(ROOT)):sha256(p) for p in [Path(__file__),ROOT/'scripts/run_research_v2.py',ROOT/'src/research_evaluation.py']}}
    write_json(out/'protocol.json',protocol)
    selections=[];chosen={};started=time.perf_counter()
    for name in list(BASELINES)+list(CELLS):
        candidates=CONFIGS if name in CELLS else BASELINES[name]
        if name in CELLS:NEURAL[name]=('gat' if name.startswith('gat_') else 'uniform_gat' if name=='uniform_physics' else 'pore_set',CELLS[name][0])
        scores=[]
        for ci,config in enumerate(candidates):
            errors=[];epochs=[]
            for vp in parents:
                train=[i for i in development if records[i]['parent_id']!=vp]
                valid=[i for i in development if records[i]['parent_id']==vp]
                if name in CELLS:
                    model,transform,info=fit_neural(name,records,graphs,train,valid,2026,config,120,20,geometry=CELLS[name][1])
                    transformed={i:transform.apply(records[i],graphs[i]) for i in valid}
                    values=predict(model,transformed,valid,transform);epochs.append(info['selected_epoch'])
                else:values=baseline(name,config,records,graphs,train,valid)
                error=parent_macro_mse(records,valid,values);errors.append(error)
                selections.append({'model':name,'candidate':ci,'development_validation_parent':vp,
                                   'validation_parent_mse':error,'selected_epoch':epochs[-1] if epochs else None})
            scores.append((float(np.mean(errors)),ci,max(1,int(np.rint(np.median(epochs)))) if epochs else None))
        score,ci,epochs=min(scores,key=lambda x:(x[0],x[1]))
        chosen[name]={'config':candidates[ci],'epochs':epochs,'development_macro_mse':score}
        write_csv(out/'selection.csv',selections);write_json(out/'locked_model_choices.json',chosen)
        print(f'Selected on development data only: {name}, epochs={epochs}',flush=True)
    # Complete all model choices before obtaining any external predictions or test metrics.
    selected_gat=min((n for n in CELLS if n.startswith('gat_')),key=lambda n:chosen[n]['development_macro_mse'])
    selected_all=min(chosen,key=lambda n:chosen[n]['development_macro_mse'])
    predictions=[]
    for name,choice in chosen.items():
        for seed in seeds:
            if name in CELLS:
                model,transform,_=fit_neural(name,records,graphs,development,[],seed,choice['config'],120,20,
                                             fixed_epochs=choice['epochs'],geometry=CELLS[name][1])
                transformed={i:transform.apply(records[i],graphs[i]) for i in test}
                values=predict(model,transformed,test,transform)
                torch.save({'model':name,'physics':CELLS[name][0],'geometry':CELLS[name][1],
                            'state_dict':model.state_dict(),'transform':transform,'seed':seed,
                            'config':choice['config'],'epochs':choice['epochs'],'development_parents':parents},
                           out/'checkpoints'/f'{name}_seed{seed}.pth')
            else:values=baseline(name,choice['config'],records,graphs,development,test,seed)
            predictions.extend({'model':name,'sample_id':records[i]['id'],'parent_id':records[i]['parent_id'],
                                 'seed':seed,'selection':name,'true_log10_k_m2':records[i]['true_log10_k_m2'],
                                 'predicted_log10_k_m2':float(v)} for i,v in zip(test,values))
        write_csv(out/'predictions.csv',predictions)
        print(f'Completed frozen external inference: {name}',flush=True)
    for selector,name in [('selected_gat',selected_gat),('selected_all',selected_all)]:
        predictions.extend([dict(r,model=selector,selection=name) for r in predictions if r['model']==name])
    write_csv(out/'predictions.csv',predictions)
    summarize(predictions,out,seeds)
    protocol.update(status='complete',selected_gat=selected_gat,selected_all=selected_all,
                    elapsed_seconds=time.perf_counter()-started)
    write_json(out/'protocol.json',protocol)
    # summarize()'s generic prose describes nested development evaluation; replace it for the external design.
    (out/'research_report.md').write_text('''# Locked external scan evaluation

The model is fitted to all 24 development crops from Berea, C1, C2 and Estaillades.
Hyperparameters and epochs are selected by leaving out development parents only.
The four additional scans S1, S3, S5 and S9 supply the locked external test crops.
Synthetic media are excluded. See metrics.csv and per_parent_metrics.csv for the
complete results, including the fixed GAT and development-selected strategies.

The external scans are four named sandstone datasets from one public collection.
Their geological specimen independence has not been independently certified.
All labels remain the same cylindrical-throat PNM approximation; external scan
validation does not independently validate that physical approximation. Published
full-volume references are recorded separately and are not crop labels.
''',encoding='utf-8')
    print('Completed locked external evaluation.',flush=True)


if __name__=='__main__':main()
