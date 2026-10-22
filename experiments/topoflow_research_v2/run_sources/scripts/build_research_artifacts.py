"""Build paper tables and vector figures exclusively from completed V2 predictions."""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.experiment_metrics import regression_metrics
from scripts.run_experiments import write_csv,write_json

LABELS={'gat_standard':'GAT standardized','gat_physics':'GAT physics + geometry',
        'sage_physics':'SAGE physics + geometry','uniform_physics':'Uniform GAT + geometry',
        'pore_set_physics':'Pore-set + geometry','physics_ridge':'Physics ridge',
        'scaled_kc':'Diameter-scaled KC','porosity_kc':'Porosity KC','descriptors':'ExtraTrees descriptors',
        'constant':'Train mean','selected_gat':'Inner-selected GAT','selected_all':'Inner-selected model'}
COLORS=['#0072B2','#D55E00','#009E73','#CC79A7','#E69F00','#56B4E9','#666666']


def read_csv(path):
    with Path(path).open(encoding='utf-8') as f:return list(csv.DictReader(f))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='experiments/topoflow_research_v2')
    parser.add_argument('--resource-directory',default='resources_verified')
    args=parser.parse_args();output=Path(args.output)
    protocol=json.loads((output/'protocol.json').read_text(encoding='utf-8'))
    if protocol['status']!='complete':raise ValueError('Run must be complete before generating paper artifacts')
    figures=output/'figures';figures.mkdir(exist_ok=True)
    tables=output/'tables';tables.mkdir(exist_ok=True)
    metrics=read_csv(output/'metrics.csv')
    for r in metrics:
        for key in r:
            if key!='model' and r[key]!='':r[key]=float(r[key])
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
                         'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    def save(fig,name):
        fig.tight_layout()
        for ext in ['png','pdf','svg']:fig.savefig(figures/f'{name}.{ext}',dpi=300,bbox_inches='tight')
        plt.close(fig)
    order=sorted(metrics,key=lambda r:r['parent_macro_rmse'])
    fig,ax=plt.subplots(figsize=(8,4.9))
    ys=np.arange(len(order))
    ax.barh(ys,[r['parent_macro_rmse'] for r in order],
            xerr=[r['parent_macro_rmse_seed_std'] for r in order],capsize=3,
            color=['#0072B2' if 'gat' in r['model'] else '#888888' for r in order])
    ax.set_yticks(ys,[LABELS[r['model']] for r in order]);ax.invert_yaxis()
    ax.set(xlabel=r'Parent-balanced RMSE in $\log_{10}(k/\mathrm{m}^2)$',
           title='Natural-rock generalization across four held-out scans')
    ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
    fig.text(.99,.005,f"Bars: mean over five seeds; whiskers: seed SD. Four natural parents, {protocol['natural_crops']} crops.",
             ha='right',fontsize=8)
    save(fig,'natural_parent_comparison')

    ensemble=read_csv(output/'ensemble_predictions.csv')
    shown=['gat_standard','gat_physics','physics_ridge','pore_set_physics']
    parents=sorted({r['parent_id'] for r in ensemble})
    fig,axes=plt.subplots(2,2,figsize=(8,7),sharex=True,sharey=True)
    truth=[float(r['true_log10_k_m2']) for r in ensemble]
    pred=[float(r['predicted_log10_k_m2']) for r in ensemble if r['model'] in shown]
    lo=min(truth+pred)-.1;hi=max(truth+pred)+.1
    for ax,name in zip(axes.flat,shown):
        for j,parent in enumerate(parents):
            rs=[r for r in ensemble if r['model']==name and r['parent_id']==parent]
            ax.errorbar([float(r['true_log10_k_m2']) for r in rs],
                        [float(r['predicted_log10_k_m2']) for r in rs],
                        yerr=[float(r['prediction_seed_std']) for r in rs],
                        fmt='o',markersize=4,color=COLORS[j],alpha=.85,
                        label=parent.removeprefix('parent_').capitalize(),capsize=2)
        ax.plot([lo,hi],[lo,hi],color='black',linestyle='--',linewidth=.8)
        ax.set_title(LABELS[name]);ax.set_xlim(lo,hi);ax.set_ylim(lo,hi);ax.set_aspect('equal')
        ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7)
    fig.supxlabel(r'Reference $\log_{10}(k/\mathrm{m}^2)$')
    fig.supylabel(r'Predicted $\log_{10}(k/\mathrm{m}^2)$')
    fig.suptitle('Each crop predicted only while its source scan is held out',fontsize=11)
    save(fig,'natural_predicted_vs_reference')

    per=read_csv(output/'per_parent_metrics.csv')
    shown=['gat_standard','gat_physics','sage_physics','uniform_physics','pore_set_physics','physics_ridge','scaled_kc']
    values=np.asarray([[float(next(r for r in per if r['model']==m and r['parent_id']==p)['rmse_log10_m2'])
                        for p in parents] for m in shown])
    fig,ax=plt.subplots(figsize=(7.5,4.5))
    im=ax.imshow(values,cmap='Blues',aspect='auto',vmin=0)
    for (i,j),v in np.ndenumerate(values):
        ax.text(j,i,f'{v:.3f}',ha='center',va='center',color='white' if v>values.max()*.6 else 'black')
    ax.set_yticks(range(len(shown)),[LABELS[m] for m in shown])
    ax.set_xticks(range(len(parents)),[p.removeprefix('parent_').capitalize() for p in parents])
    ax.set_title('Per-scan RMSE (log10 permeability); average over five seeds')
    fig.colorbar(im,ax=ax,label='RMSE (lower is better)');save(fig,'per_scan_error')

    # Report ensemble metrics separately rather than mixing them with single-seed averages.
    em=[]
    for model in sorted({r['model'] for r in ensemble}):
        rs=[r for r in ensemble if r['model']==model]
        result=regression_metrics([float(r['true_log10_k_m2']) for r in rs],
                                  [float(r['predicted_log10_k_m2']) for r in rs])
        em.append({'model':model,'prediction_type':'mean log prediction across five seeds',**result})
    write_csv(tables/'ensemble_metrics.csv',em)
    lines=['# Paper comparison table','',f"Metrics use natural rocks only: four parent scans and {protocol['natural_crops']} accepted crops. "
           'Each crop appears in exactly one outer test fold. Neural settings and epoch counts are selected '
           'inside the training parents. All labels are PNM approximations.', '',
           '| Model | Parent RMSE mean +/- seed SD | Log MSE | Log RMSE | Log MAE | Log R2 | Linear R2 | Factor-two accuracy |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    # MSE mean is obtained from prediction rows, not square of the mean seed RMSE.
    predictions=read_csv(output/'predictions.csv')
    for r in order:
        selected=[p for p in predictions if p['model']==r['model']]
        mse=np.mean([(float(p['predicted_log10_k_m2'])-float(p['true_log10_k_m2']))**2 for p in selected])
        lines.append(f"| {LABELS[r['model']]} | {r['parent_macro_rmse']:.4f} +/- {r['parent_macro_rmse_seed_std']:.4f} "
                     f"| {mse:.4f} | {r['rmse_log10_m2']:.4f} | {r['mae_log10_m2']:.4f} "
                     f"| {r['r2_log10_m2']:.4f} | {r['r2_linear']:.4f} | {100*r['factor_of_two_accuracy']:.1f}% |")
    (tables/'paper_comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    latex=['% Four parent scans; five seeds; seed SD is not specimen uncertainty.',
           '\\begin{tabular}{lrrrr}', '\\hline',
           'Model & Parent RMSE & Log MAE & Log $R^2$ & Within $2\\times$ \\\\', '\\hline']
    for r in order:
        latex.append(f"{LABELS[r['model']]} & ${r['parent_macro_rmse']:.3f}\\pm {r['parent_macro_rmse_seed_std']:.3f}$ "
                     f"& {r['mae_log10_m2']:.3f} & {r['r2_log10_m2']:.3f} & {100*r['factor_of_two_accuracy']:.1f}\\% \\\\")
    latex+=['\\hline','\\end{tabular}']
    (tables/'paper_comparison.tex').write_text('\n'.join(latex)+'\n',encoding='utf-8')
    inventory=json.loads((output/'data_inventory.json').read_text(encoding='utf-8'))
    gat=next(r for r in metrics if r['model']=='selected_gat')
    physics=next(r for r in metrics if r['model']=='gat_physics')
    simpler=min((r for r in metrics if r['model'] in ['physics_ridge','scaled_kc','porosity_kc','descriptors','constant']),
                key=lambda r:r['parent_macro_rmse'])
    cv=[r['pore_radius_cv'] for r in inventory]
    passages=['# Manuscript replacement text','','## Evaluation protocol','',
      f"We evaluate TopoFlow on {protocol['natural_crops']} accepted, non-overlapping 128-cubed micro-CT crops from four natural "
      'parent scans: Berea sandstone and the C1, C2 and Estaillades carbonates. The accepted binary '
      'crops and voxel resolutions are verified against their source archives. We use nested '
      'leave-one-parent-out evaluation. Each outer fold holds out an entire scan; inner folds rotate '
      'the remaining scans to select model settings and training duration. We report five training '
      'seeds per outer fold. Synthetic media are excluded from this primary experiment. Permeability '
      'targets are calculated with a cylindrical-throat pore-network model and are not lattice-Boltzmann '
      'or experimental reference measurements.', '',
      'We compare the standardized original GAT with a scale-normalized formulation that predicts '
      'log10(k/d_g^2), where d_g is the geometric mean equivalent pore diameter. The latter uses '
      'dimensionless pore features and graph descriptors. GraphSAGE, uniform-attention GAT and '
      'an edge-free pore-set network receive the same descriptors. Normalization statistics are '
      'fitted exclusively to the training parents. Additional comparators are a training-mean '
      'predictor, calibrated porosity and diameter-scaled Kozeny-Carman surrogates, ExtraTrees '
      'descriptor regression and a dimensionally normalized ridge model.', '',
      '## Results','',
      f"The GAT formulation selected by inner validation achieves a parent-balanced log10 RMSE of "
      f"{gat['parent_macro_rmse']:.4f} +/- {gat['parent_macro_rmse_seed_std']:.4f} across five seeds, "
      f"with a crop-level log10 MAE of {gat['mae_log10_m2']:.4f}, pooled log-space R2 of "
      f"{gat['r2_log10_m2']:.4f}, and factor-of-two accuracy of {100*gat['factor_of_two_accuracy']:.1f}%. "
      f"The fixed physics-and-geometry GAT has parent-balanced RMSE {physics['parent_macro_rmse']:.4f}. "
      f"The {LABELS[simpler['model']]} comparator has parent-balanced RMSE {simpler['parent_macro_rmse']:.4f}. "
      'Per-scan results and the uniform-attention and edge-free controls are reported alongside '
      'aggregate metrics; aggregation does not establish a distinct advantage for learned attention.', '',
      '## Scope of conclusions','',
      f"Measured pore-radius CV spans {min(cv):.3f} to {max(cv):.3f}. The data do not test a threshold "
      'at CV=1.5. Numerical mass conservation assesses the PNM solve rather than its physical '
      'accuracy against a voxel-resolved reference. These four previously examined source scans '
      'support an exploratory comparison; they do not establish population-level geological '
      'generalization or an untouched external validation result. Seed variation measures '
      'optimization sensitivity and cannot replace independent scan replication.', '',
      'The scale-normalized formulation changes both the feature representation and the graph-level '
      'inputs. Its comparison with the standardized GAT is therefore a combined design comparison, '
      'not a causal isolation of dimensional scaling. Global descriptors are supplied equally to '
      'the physics-model controls to assess whether graph message passing adds information.']
    resource=output/args.resource_directory
    if (resource/'timing_results.csv').exists():
        timings=read_csv(resource/'timing_results.csv')
        chosen=['gat_physics_forward','sage_physics_forward','pore_set_physics_forward',
                'pnm_solve','graph_conversion','snow2']
        displayed=[next(t for t in timings if t['stage']==s) for s in chosen]
        labels=['GAT forward','SAGE forward','Pore-set forward','PNM reference solve',
                'Graph tensor packaging','SNOW2 extraction']
        med=[float(t['median_ms']) for t in displayed]
        fig,ax=plt.subplots(figsize=(7.5,4))
        ax.barh(range(len(med)),med,color=['#0072B2']*3+['#888888']*3)
        ax.set_yticks(range(len(labels)),labels);ax.invert_yaxis();ax.set_xscale('log')
        ax.set_xlabel('Measured milliseconds (log scale)');ax.set_title('CPU component timings; measured scopes differ')
        ax.set_xlim(min(med)*.5,max(med)*4)
        for i,m in enumerate(med):ax.text(m*1.15,i,f'{m:.3f} ms',va='center',fontsize=8)
        ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
        save(fig,'measured_compute_components')
        forward=read_csv(resource/'per_crop_forward.csv')
        table=['# Measured compute and storage','',
               '| Model | Parameters | Weight MiB | Median forward ms | P95 forward ms |',
               '|---|---:|---:|---:|---:|']
        for m in ['gat_standard','gat_physics','sage_physics','uniform_physics','pore_set_physics']:
            example=next(r for r in forward if r['model']==m)
            t=next(r for r in timings if r['stage']==m+'_forward')
            table.append(f"| {LABELS[m]} | {int(example['parameters']):,} | {int(example['weight_bytes'])/(1024**2):.4f} "
                         f"| {float(t['median_ms']):.4f} | {float(t['p95_ms']):.4f} |")
        table+=['','All 24 held-out natural crops are timed using trained fold checkpoints, 10 warmups '
                'and 30 forward repeats per crop. Forward measurements exclude graph extraction, '
                'normalization, loading and prediction inversion. Parameter storage excludes activations '
                'and Python/library memory. No CNN resource comparison is established.']
        (tables/'measured_compute.md').write_text('\n'.join(table)+'\n',encoding='utf-8')
        gat_time=next(t for t in timings if t['stage']=='gat_physics_forward')
        snow_time=next(t for t in timings if t['stage']=='snow2')
        pnm_time=next(t for t in timings if t['stage']=='pnm_solve')
        passages += ['', '## Computational measurements', '',
          f"On CPU, the trained scale-normalized GAT has median forward latency "
          f"{float(gat_time['median_ms']):.3f} ms and 95th-percentile latency "
          f"{float(gat_time['p95_ms']):.3f} ms across 24 held-out crops and 720 timed calls. "
          f"Fresh SNOW2 extraction on one predeclared crop per parent has median latency "
          f"{float(snow_time['median_ms'])/1000:.3f} s, while the cylindrical-throat PNM reference solve "
          f"has median latency {float(pnm_time['median_ms']):.3f} ms. The extraction statistics use four "
          'crops; repeated forward calls are timing observations rather than independent rock samples. '
          'These measurements support millisecond evaluation of a precomputed graph, but do not '
          'demonstrate a large solver acceleration or a measured memory advantage over a 3D CNN.']
    (output/'manuscript_insert.md').write_text('\n'.join(passages)+'\n',encoding='utf-8')
    print('Built paper tables, manuscript passages and vector figures.',flush=True)


if __name__=='__main__':main()
