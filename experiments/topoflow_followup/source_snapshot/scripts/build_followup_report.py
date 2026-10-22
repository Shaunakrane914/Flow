"""Generate reviewable research follow-up tables and figures from measured artifacts."""
from pathlib import Path
import csv
import itertools
import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_experiments import write_json,write_csv


def read(path):
    with Path(path).open(encoding='utf-8') as f:return list(csv.DictReader(f))


def main():
    out=ROOT/'experiments/topoflow_followup'
    figures=out/'figures';figures.mkdir(exist_ok=True)
    ablation=read(out/'ablation/metrics.csv');external=read(out/'external_test/metrics.csv')
    throat=read(out/'throat_aware_pilot/training/metrics.csv')
    predictions=read(out/'throat_aware_pilot/training/response_predictions.csv')
    importance=read(out/'throat_aware_pilot/training/importance_rankings.csv')
    dose=read(out/'critical_throats_verified/parent_summary.csv')
    def metric(rows,name,key,group=None):
        return float(next(r for r in rows if r['model']==name and (group is None or r['group']==group))[key])
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
                         'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    def save(fig,name):
        fig.tight_layout()
        for ext in ['png','pdf','svg']:fig.savefig(figures/f'{name}.{ext}',dpi=300,bbox_inches='tight')
        plt.close(fig)
    names=[['gat_standard','gat_descriptors_only'],['gat_normalized_only','gat_physics']]
    matrix=np.array([[metric(ablation,n,'parent_macro_rmse') for n in row] for row in names])
    ablation_lines=['# Controlled GAT ablation','',
      'Four development scans are evaluated with nested parent-held-out validation. '
      'The physical-normalization factor changes both node/target dimensional scaling, '
      'while the descriptor factor independently enables the matched graph-level descriptor head. '
      'The original GAT encoder and candidate grid are held fixed. The two compatible '
      'V2 cells are reused and the two missing cells are newly trained.', '',
      '| Normalization | Descriptors | Parent-balanced log RMSE |','|---|---|---:|']
    for i in range(2):
        for j in range(2):ablation_lines.append(f"| {'Physical' if i else 'Standard'} | {'Yes' if j else 'No'} | {matrix[i,j]:.4f} |")
    ablation_lines+=['','The primary four-cell comparison concerns GAT design choices. '
      'Reused normalized descriptor controls are included in the prediction CSV for context. '
      'This is development cross-validation, not the separately locked external evaluation.']
    (out/'ablation/research_report.md').write_text('\n'.join(ablation_lines)+'\n',encoding='utf-8')
    fig,ax=plt.subplots(figsize=(5.5,3.7));im=ax.imshow(matrix,cmap='Blues',vmin=0)
    ax.set_xticks([0,1],['No descriptors','With descriptors']);ax.set_yticks([0,1],['Standardized original inputs','Physical normalization'])
    for (i,j),v in np.ndenumerate(matrix):ax.text(j,i,f'{v:.3f}',ha='center',va='center',color='white' if v>1.2 else 'black',fontsize=14)
    fig.colorbar(im,ax=ax,label='Parent-balanced log RMSE');ax.set_title('Controlled GAT design comparison\nDevelopment scans only; four held-out parent folds')
    save(fig,'gat_factorial_ablation')

    parents=sorted({r['parent_id'] for r in dose})
    fig,ax=plt.subplots(figsize=(6.5,4));colors=['#0072B2','#D55E00','#009E73','#CC79A7']
    for j,parent in enumerate(parents):
        for group,style in [('high_dissipation','-'),('random','--')]:
            rs=sorted([r for r in dose if r['parent_id']==parent and r['selection_group']==group],key=lambda r:float(r['diameter_reduction_fraction']))
            ax.plot([100*float(r['diameter_reduction_fraction']) for r in rs],
                    [100*float(r['mean_fractional_permeability_loss']) for r in rs],
                    marker='o',linestyle=style,color=colors[j],markersize=3,
                    label=parent.removeprefix('parent_')+' '+('critical' if style=='-' else 'random'))
    ax.axhline(0,color='black',linewidth=.6,label='Existing GAT: no partial response')
    ax.set(xlabel='Throat diameter reduction (%)',ylabel='PNM permeability loss (%)',
           title='Three critical and three random throats per development graph')
    ax.legend(fontsize=6,ncol=2);ax.grid(alpha=.2);save(fig,'throat_intervention_response')

    names=['edge_gat','guided_edge_gat','edge_sage','guided_edge_sage','edge_mlp','guided_edge_mlp','no_response','conductance_proxy']
    labels=['Edge GAT','Guided GAT','Edge mean','Guided edge mean*','Edge MLP','Guided edge MLP*','No response','Conductance proxy']
    fig,axes=plt.subplots(1,2,figsize=(10,4.8))
    vals=[100*metric(throat,n,'mean_absolute_fractional_loss_error','all') for n in names]
    sd=[100*metric(throat,n,'seed_std','all') for n in names]
    axes[0].barh(range(len(names)),vals,xerr=sd,capsize=3,
                 color=['#0072B2' if n=='guided_edge_gat' else '#999999' for n in names])
    axes[0].set_yticks(range(len(names)),labels);axes[0].invert_yaxis()
    axes[0].set_xlabel('Permeability-loss MAE (percentage points)');axes[0].grid(axis='x',alpha=.2);axes[0].set_axisbelow(True)
    ranking=[]
    for name in names:
        rows=[r for r in importance if r['model']==name and r['spearman_hydraulic_importance']]
        ranking.append(np.mean([float(r['spearman_hydraulic_importance']) for r in rows]) if rows else 0.)
    axes[1].barh(range(len(names)),ranking,color=['#0072B2' if n=='guided_edge_gat' else '#999999' for n in names])
    axes[1].set_yticks(range(len(names)),labels);axes[1].invert_yaxis();axes[1].set_xlabel('Mean Spearman correlation with PNM sensitivity')
    axes[1].text(.005,names.index('no_response'),'undefined',va='center',fontsize=8)
    axes[1].grid(axis='x',alpha=.2);axes[1].set_axisbelow(True)
    fig.suptitle('External critical-throat pilot: four graphs from four additional scans',fontsize=11)
    fig.text(.5,-.025,'Whiskers: training-seed SD. *Guided controls added after the first external pilot results.\nThe hydraulic-importance head is distinct from native GAT attention coefficients.',ha='center',fontsize=8)
    save(fig,'critical_throat_pilot')

    selected=['gat_standard','gat_physics','gat_normalized_only','gat_descriptors_only','pore_set_physics','uniform_physics','physics_ridge','scaled_kc','porosity_kc','descriptors']
    selected=sorted(selected,key=lambda n:metric(external,n,'parent_macro_rmse'))
    fig,ax=plt.subplots(figsize=(7.5,4.5));values=[metric(external,n,'parent_macro_rmse') for n in selected]
    ax.barh(range(len(selected)),values,xerr=[metric(external,n,'parent_macro_rmse_seed_std') for n in selected],capsize=3,
            color=['#0072B2' if n.startswith('gat_') else '#888888' for n in selected])
    ax.set_yticks(range(len(selected)),[n.replace('_',' ') for n in selected]);ax.invert_yaxis()
    ax.set(xlabel='Parent-balanced log10 RMSE',title='Locked external permeability test: 32 crops, four new scans')
    ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True);save(fig,'locked_external_permeability')

    # Use parent-level paired effects, not the thousands of dependent edge/dose/seed rows.
    comparisons=[]
    for comparator in [n for n in names if n!='guided_edge_gat']:
        effects=[]
        for parent in sorted({r['parent_id'] for r in predictions}):
            def error(model):
                rs=[r for r in predictions if r['model']==model and r['parent_id']==parent]
                return np.mean([abs(float(r['predicted_fractional_permeability_loss'])-float(r['true_fractional_permeability_loss'])) for r in rs])
            effects.append(float(error(comparator)-error('guided_edge_gat')))
        signs=np.asarray(list(itertools.product([-1,1],repeat=len(effects))))
        p=float(np.mean(np.abs(np.mean(signs*effects,axis=1))>=abs(np.mean(effects))-1e-14))
        comparisons.append({'comparator':comparator,'parent_mae_differences_positive_favors_guided_gat':effects,
                            'parent_mean_effect':float(np.mean(effects)),'exploratory_exact_sign_flip_p':p,
                            'independent_test_groups':len(effects)})
    write_json(out/'throat_aware_pilot/training/parent_comparisons.json',comparisons)
    per=[]
    for name in names:
        for parent in sorted({r['parent_id'] for r in predictions}):
            rs=[r for r in predictions if r['model']==name and r['parent_id']==parent]
            per.append({'model':name,'parent_id':parent,'physical_interventions':18,'training_seeds':5,
                        'mae_fractional_loss':float(np.mean([abs(float(r['predicted_fractional_permeability_loss'])-float(r['true_fractional_permeability_loss'])) for r in rs]))})
    write_csv(out/'throat_aware_pilot/training/per_parent_metrics.csv',per)

    lines=['# TopoFlow follow-up experiments','',
      'This study completes the controlled GAT ablation, a locked test on four additional natural scans, '
      'and a critical-throat intervention pilot. The measured gap is an input-identifiability limitation: '
      'a GAT that receives only node size/volume and adjacency cannot distinguish networks that differ '
      'only in throat diameter. The pilot extension supplies throat geometry and predicts counterfactual '
      'permeability loss with a separate learned hydraulic-importance head.', '',
      '## Controlled design comparison','',
      '| Physical normalization | Graph descriptors | Development parent-balanced log RMSE |',
      '|---|---|---:|']
    for i in range(2):
        for j in range(2):lines.append(f"| {'Yes' if i else 'No'} | {'Yes' if j else 'No'} | {matrix[i,j]:.4f} |")
    lines+=['','All cells use the same original GAT encoder, inner parent validation, candidate grid '
      'and five seeds. The normalization factor changes both dimensionless input and dimensionless '
      'target representation; it is not an isolated comparison of input z-scoring alone. Two existing '
      'cells are reused from their exact compatible V2 runs; two missing cells are newly trained.', '',
      '## Locked external permeability evaluation','',
      'The official archive supplies S1, S3, S5 and S9. All 32 predeclared corner crops are accepted. '
      'The scans are locked before crop preparation, and all model settings are selected on the four '
      'development scans only. These are four named sandstone datasets from the same collection; '
      'independent geological provenance beyond the specimen identifiers is not certified.', '',
      '| Model | Log RMSE | Log R2 | Factor-two accuracy |','|---|---:|---:|---:|']
    for r in sorted(external,key=lambda r:float(r['parent_macro_rmse'])):
        lines.append(f"| {r['model']} | {float(r['parent_macro_rmse']):.4f} | {float(r['r2_log10_m2']):.4f} | {100*float(r['factor_of_two_accuracy']):.1f}% |")
    lines+=['','Physics ridge remains the strongest permeability comparator in this external experiment. '
      'The primary fixed GAT and the GAT selected on development data do not beat it. These external '
      'permeability results were not used to adjust the original permeability models afterward.', '',
      '## Existing GAT counterfactual failure','',
      'On one predeclared accepted crop from each development scan, three highest-dissipation and three '
      'other random internal throats are narrowed by 10%, 25%, 50% or fully removed. Partial narrowing '
      'does not change node features, adjacency or the V2 graph descriptors. Both existing GAT '
      'formulations therefore emit exactly zero predicted change. Full removal alters adjacency and is '
      'reported separately. A disconnected full-blockage case has k=0 and no finite log label.', '',
      'At 50% diameter reduction, the mean loss across the three high-dissipation throats is '
      +', '.join(f"{100*float(next(r for r in dose if r['parent_id']==p and r['selection_group']=='high_dissipation' and float(r['diameter_reduction_fraction'])==.5)['mean_fractional_permeability_loss']):.1f}% ({p.removeprefix('parent_')})" for p in parents)
      +'. This is behavior under the declared PNM approximation, not a real clogging measurement.', '',
      '## Throat-aware prototype and external pilot','',
      'The prototype is trained on eight development graphs, two per development scan. One graph from '
      'each of the four additional scans is reserved for external response evaluation. The original '
      'primary guided-GAT and three unguided architectures are specified before evaluating those pilot '
      'responses. Equally guided mean-aggregation and edge-MLP controls are added after inspecting '
      'the first pilot results to test a supervision confound; those two comparisons are explicitly post hoc.', '',
      'Features comprise dimensionless pore geometry, coordinates, inlet/outlet flags, throat diameter, '
      'length, directional geometry and a conductance proxy. No pressure, flow, dissipation label or '
      'reference permeability enters the model forward pass. The guided model receives PNM hydraulic '
      'importance only as an auxiliary training target. Its global importance head is not a native '
      'GAT attention coefficient, so it does not establish that the attention coefficients are explanations.', '',
      '| Model | Permeability-loss MAE in percentage points | Seed SD in points | Importance Spearman |',
      '|---|---:|---:|---:|']
    for n in names:
        r=next(r for r in throat if r['model']==n and r['group']=='all')
        rho=r['mean_spearman_importance'] or 'undefined'
        lines.append(f"| {n} | {100*float(r['mean_absolute_fractional_loss_error']):.3f} | {100*float(r['seed_std']):.3f} | {float(rho):.3f} |" if rho!='undefined' else
                     f"| {n} | {100*float(r['mean_absolute_fractional_loss_error']):.3f} | {100*float(r['seed_std']):.3f} | undefined |")
    guided=metric(throat,'guided_edge_gat','mean_absolute_fractional_loss_error','all')
    zero=metric(throat,'no_response','mean_absolute_fractional_loss_error','all')
    lines+=['',f'The primary guided GAT reduces mean absolute response error by {100*(1-guided/zero):.1f}% relative '
      'to predicting no response. This is an exploratory four-graph result. Seed variation is not '
      'specimen uncertainty, and thousands of dependent edge/dose/seed rows do not create additional '
      'independent rocks. The exact parent sign-flip comparisons are exported; with four parents their '
      'minimum possible two-sided p-value is 0.125.', '',
      'The no-response control has lower error for random throats than the guided GAT. The gain is '
      'concentrated in the high-dissipation subset. Test throats are selected using an oracle PNM '
      'dissipation field to construct the experiment; this selector is not claimed as a solver-free '
      'deployment mechanism. Separately reported ranking metrics assess the learned geometry-only head.', '',
      '## Independent published full-volume consistency check','',
      'Complete 300-cubed S1 and S5 images are processed in isolated workers, rather than assigning '
      'published full-scan values to crops. Our PNM gives 342.49 and 1187.48 mD, compared with '
      'published Kz values of 1312 and 4440 mD. Boundary settings are incompletely matched and the '
      'GAT inputs are larger than their 128-cube training domains. These are cross-method consistency '
      'checks, not certified matched DNS validation or newly run LBM simulations. The approximately '
      'fourfold discrepancies prevent treating numerical mass conservation as physical accuracy.', '',
      '## Research positioning','',
      'The defensible question is whether geometry-only learned hydraulic importance predicts controlled '
      'single-throat restrictions better than equally informed controls on unseen scans. The current '
      'evidence supports a promising pilot, not a universal critical-throat detector or an established '
      'attention-superiority claim. The sensitivity and finite-update identities are known consequences '
      'of linear network physics; they are not presented as new mathematical results.', '',
      '[RTG-GNN (2024)](https://doi.org/10.1016/j.geoen.2024.213358) already uses pore/throat topology '
      'for permeability prediction. [Zhao and Xiao (2026)](https://doi.org/10.1016/j.jhydrol.2026.135602) '
      'already learns conductances inside a PNM and studies physically consistent geometric sensitivities. '
      'Therefore edge features and sensitivity analysis alone are not novelty. Controlled intervention '
      'prediction and comparison of learned importance with hydraulic sensitivity are the narrower '
      'contribution being investigated; a first-of-its-kind claim is not established.', '',
      'Sources: [official Imperial collection](https://www.imperial.ac.uk/earth-science/research/research-groups/pore-scale-modelling/micro-ct-images-and-networks/) '
      'and [Dong and Blunt (2009)](https://doi.org/10.1103/PhysRevE.80.036307).', '',
      '## Artifacts','',
      '- `ablation/`: controlled four-cell results, configurations and 40 new checkpoints.',
      '- `external_protocol.json` and `external_data/`: scan lock, 32 crop attempts, source headers and hashes.',
      '- `external_test/`: 30 trained models, prediction records, tuning records and per-scan results.',
      '- `critical_throats_verified/`: 96 physical interventions, two existing GAT evaluations each, attention rankings and physics checks.',
      '- `throat_aware_pilot/`: 12-graph inventory, 30 trained pilot models, response records and parent-level comparisons.',
      '- `full_reference/`: complete-domain reference calculations and published-value scope.',
      '- `figures/`: four figures in PNG, SVG and PDF.',
      '- Raw source archives, images and graph tensors remain in the ignored `data/` directory.']
    (out/'FOLLOWUP_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (out/'README.md').write_text('# TopoFlow follow-up research\n\nStart with [the complete follow-up report](FOLLOWUP_REPORT.md).\n\n'
      'This package includes controlled normalization/descriptor ablations, four locked external scans, '
      'a throat-intervention feasibility test, a trained throat-aware prototype with matched controls, '
      'and full-image consistency checks against published numerical references. All claims remain '
      'limited by small parent counts and the declared pore-network approximation.\n',encoding='utf-8')
    print('Generated follow-up report, parent comparisons and four vector figures.',flush=True)


if __name__=='__main__':main()
