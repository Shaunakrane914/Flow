"""Compare full-domain PNM and locked GAT predictions to published Kz metadata."""
from pathlib import Path
import json
import sys
import numpy as np
import torch
from torch_geometric.loader import DataLoader

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.research_evaluation import ScaleAwareRegressor
from scripts.run_experiments import write_csv,write_json


def main():
    torch.set_num_threads(1)
    out=ROOT/'experiments/topoflow_followup/full_reference'
    refs=json.loads((ROOT/'experiments/topoflow_followup/external_data/source_references.json').read_text())
    rows=[]
    for name in ['S1','S5']:
        file=out/f'{name}_pnm.json'
        if not file.exists():continue
        meta=json.loads(file.read_text());reference=next(r for r in refs if r['parent']==name)
        published=reference['published_full_volume_directional_k_mD']['z']*9.869233e-16
        graph=torch.load(meta['graph_path'],map_location='cpu',weights_only=False)
        diameter=10**graph.x[:,0].double().numpy();volume=10**graph.x[:,1].double().numpy()
        degree=np.bincount(graph.edge_index[1].numpy(),minlength=graph.num_nodes)
        descriptors=[meta['porosity'],np.log10(diameter).mean(),np.log10(diameter).std(),
                     np.log10(volume).mean(),np.log10(volume).std(),np.log10(graph.num_nodes),
                     np.log10(graph.num_edges),degree.mean(),degree.std(),diameter.std()/diameter.mean()]
        record={'porosity':meta['porosity'],'descriptors':descriptors,'true_log10_k_m2':float(graph.y.item())}
        rows.append({'sample':name,'method':'cylindrical_throat_pnm','seed':0,'k_m2':meta['pnm_k_m2'],
                     'published_full_volume_kz_m2':published,'ratio_to_published_kz':meta['pnm_k_m2']/published})
        for modelname,physics,geometry in [('gat_physics',True,True),('gat_normalized_only',True,False),('gat_descriptors_only',False,True)]:
            for seed in [11,22,33,44,55]:
                ck=torch.load(ROOT/f'experiments/topoflow_followup/external_test/checkpoints/{modelname}_seed{seed}.pth',map_location='cpu',weights_only=False)
                model=ScaleAwareRegressor('gat',physics=physics,geometry=geometry,dropout=ck['config']['dropout']).eval()
                model.load_state_dict(ck['state_dict']);transform=ck['transform']
                batch=next(iter(DataLoader([transform.apply(record,graph)],batch_size=1)))
                with torch.inference_mode():logk=model(batch).item()*transform.target_std+transform.target_mean+batch.offset.item()
                value=10**logk
                rows.append({'sample':name,'method':modelname,'seed':seed,'k_m2':value,
                             'published_full_volume_kz_m2':published,'ratio_to_published_kz':value/published})
    write_csv(out/'full_volume_comparison.csv',rows)
    note={'domain':'Exact full 300-cubed source binary, not crop targets','direction':'array axis 0 mapped to source z; nominal Kz comparison',
          'source':'official archive summary_results_2009; primary source Dong and Blunt (2009), doi:10.1103/PhysRevE.80.036307',
          'published_boundary_settings_fully_matched':False,
          'interpretation':'Cross-method consistency check. Differences combine closure, extraction and possible boundary conventions. '
                           'The 300-cube GAT inputs also exceed its 128-cube training size. Neither agreement nor disagreement alone certifies physical accuracy.',
          'published_network_values_not_used_as_direct_reference':True}
    write_json(out/'reference_scope.json',note)
    lines=['# Published full-volume reference comparison','','| Sample | Our full-volume PNM mD | Published full-volume Kz mD | Ratio |',
           '|---|---:|---:|---:|']
    for r in rows:
        if r['method']=='cylindrical_throat_pnm':
            lines.append(f"| {r['sample']} | {r['k_m2']/9.869233e-16:.2f} | {r['published_full_volume_kz_m2']/9.869233e-16:.2f} | {r['ratio_to_published_kz']:.3f} |")
    lines+=['','These comparisons use complete source image domains. Published full-image values are not '
            'assigned to extracted crops. The archive separates directional voxel-image permeability from '
            'its original extracted-network permeability. Full boundary settings have not been independently '
            'matched, and the learned models were trained on 128-cube crops rather than 300-cube full scans. '
            'Therefore this is a consistency check against an independent published numerical reference, '
            'not a certified matched DNS validation or a new LBM simulation.', '',
            'Source: [official Imperial College collection](https://www.imperial.ac.uk/earth-science/research/research-groups/pore-scale-modelling/micro-ct-images-and-networks/) '
            'and [Dong and Blunt (2009)](https://doi.org/10.1103/PhysRevE.80.036307).']
    (out/'reference_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Saved full-volume consistency comparison.',flush=True)


if __name__=='__main__':main()
