"""Build scan-clustered results and paper figures from the sealed discovery test."""
from itertools import product
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_throat_discovery import OUT, DATA, MODELS, SEEDS, load_models, checked_protocol
from scripts.run_experiments import write_json
from src.critical_throat_model import fractional_response
from src.experiment_data import sha256

LABELS = {'guided_edge_gat': 'Guided GAT', 'guided_edge_sage': 'Guided mean GNN',
          'guided_edge_mlp': 'Guided edge MLP', 'edge_gat': 'Unguided GAT',
          'conductance': 'Conductance', 'affine_pressure': 'Affine pressure',
          'random': 'Random', 'pnm_sensitivity_oracle': 'PNM sensitivity (solver)',
          'no_response': 'No response'}


def interval(values):
    values = np.asarray(values)
    indices = np.asarray(list(product(range(len(values)), repeat=len(values))))
    return np.quantile(values[indices].mean(1), [.025, .975]).tolist()


def paired(left, right):
    values = np.asarray(left) - np.asarray(right)
    signs = np.asarray(list(product([-1, 1], repeat=len(values))))
    observed = float(values.mean())
    p = np.mean(np.abs((signs * values).mean(1)) >= abs(observed) - 1e-12)
    return {'scan_macro_difference': observed, 'cluster_bootstrap_95_interval': interval(values),
            'two_sided_exact_sign_flip_p': float(p), 'scan_differences': values.tolist(),
            'scope': 'Five source-scan clusters; descriptive bootstrap; no intervention-level independence assumed'}


def table(frame, columns, decimals=3):
    lines = ['| ' + ' | '.join(columns.values()) + ' |', '| ' + ' | '.join(['---'] * len(columns)) + ' |']
    for _, row in frame.iterrows():
        lines.append('| ' + ' | '.join(f'{row[k]:.{decimals}f}' if isinstance(row[k], (float, np.floating))
                                      else str(row[k]) for k in columns) + ' |')
    return '\n'.join(lines)


def savefig(fig, name):
    folder = OUT / 'figures'
    folder.mkdir(exist_ok=True)
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(folder / f'{name}.{ext}', dpi=220, bbox_inches='tight')
    plt.close(fig)


def main():
    torch.set_num_threads(1)
    checked_protocol()
    complete = json.loads((OUT / 'evaluation_complete.json').read_text())
    discovery = pd.read_csv(OUT / 'discovery.csv')
    prediction = pd.read_csv(OUT / 'common_pool_predictions.csv')
    ranks = pd.read_csv(OUT / 'rankings.csv')
    timing = pd.read_csv(OUT / 'timing_observations.csv')
    inventory = pd.read_json(OUT / 'inventory.json')
    random = discovery[(discovery.method == 'guided_edge_gat') & (discovery.selection_group == 'random')].copy()
    random['method'] = 'random'
    primary = pd.concat([discovery[discovery.selection_group == 'top'], random])
    primary = primary[primary.diameter_reduction == .5]
    graph_top = primary.groupby(['method', 'scan_id', 'sample_id']).actual_loss.mean()
    scan_top = graph_top.groupby(['method', 'scan_id']).mean().unstack('method')
    order = MODELS + ['affine_pressure', 'conductance', 'random', 'pnm_sensitivity_oracle']
    summary = []
    for method in order:
        values = scan_top[method].values
        lo, hi = interval(values)
        summary.append({'method': method, 'label': LABELS[method], 'mean_actual_loss_pp': 100 * values.mean(),
                        'ci_low_pp': lo * 100, 'ci_high_pp': hi * 100,
                        'mean_sensitivity_spearman': float(ranks[ranks.method == method].groupby('scan_id').spearman_sensitivity.mean().mean())
                            if method in set(ranks.method) else np.nan})
    discovery_summary = pd.DataFrame(summary)
    scan_top.to_csv(OUT / 'per_scan_discovery.csv')
    discovery_summary.to_csv(OUT / 'discovery_metrics.csv', index=False)
    comparisons = {f'guided_edge_gat_minus_{control}': paired(scan_top.guided_edge_gat, scan_top[control])
                   for control in ['affine_pressure', 'guided_edge_mlp', 'guided_edge_sage', 'random', 'edge_gat']}
    write_json(OUT / 'paired_scan_comparisons.json', comparisons)
    prediction['absolute_error'] = abs(prediction.predicted_loss - prediction.actual_loss)
    graph_error = prediction.groupby(['method', 'scan_id', 'sample_id']).absolute_error.mean()
    scan_error = graph_error.groupby(['method', 'scan_id']).mean().unstack('method')
    accuracy_rows = []
    for method in MODELS + ['affine_pressure', 'conductance', 'no_response']:
        lo, hi = interval(scan_error[method].values)
        accuracy_rows.append({'method': method, 'label': LABELS[method],
                              'common_pool_mae_pp': scan_error[method].mean() * 100,
                              'ci_low_pp': lo * 100, 'ci_high_pp': hi * 100})
    accuracy = pd.DataFrame(accuracy_rows)
    accuracy.to_csv(OUT / 'accuracy_metrics.csv', index=False)
    scan_error.to_csv(OUT / 'per_scan_accuracy.csv')
    # Seed variability is secondary; the primary deployed ensemble stays fixed.
    models = load_models()
    seed_rows = []
    for case in inventory.to_dict('records'):
        graph = torch.load(DATA / f"{case['sample_id']}.pt", map_location='cpu', weights_only=False)
        batch = Batch.from_data_list([graph])
        targets = prediction[(prediction.sample_id == case['sample_id']) & (prediction.method == 'no_response')]
        edges = targets.throat_index.to_numpy(dtype=int)
        doses = targets.diameter_reduction.map({.1: 0, .25: 1, .5: 2}).to_numpy(dtype=int)
        for method in MODELS:
            for seed, model in zip(SEEDS, models[method]):
                with torch.inference_mode():
                    predicted = fractional_response(*model(batch)).numpy()[edges, doses]
                seed_rows.append({'method': method, 'seed': seed, 'sample_id': case['sample_id'],
                                  'scan_id': case['scan_id'],
                                  'mae': float(np.abs(predicted - targets.actual_loss.to_numpy()).mean())})
    seed_frame = pd.DataFrame(seed_rows)
    seed_frame.to_csv(OUT / 'seed_accuracy.csv', index=False)
    seed_macro = seed_frame.groupby(['method', 'seed', 'scan_id']).mae.mean().groupby(['method', 'seed']).mean()
    seed_summary = seed_macro.groupby('method').agg(['mean', 'std']) * 100
    seed_summary.to_csv(OUT / 'seed_accuracy_summary_pp.csv')
    per_graph_time = timing.groupby(['method', 'scope', 'scan_id', 'sample_id']).ms.median().reset_index()
    per_scan_time = per_graph_time.groupby(['method', 'scope', 'scan_id']).ms.mean().reset_index()
    latency = per_scan_time.groupby(['method', 'scope']).ms.agg(['mean', 'min', 'max']).reset_index()
    latency.rename(columns={'mean': 'scan_macro_median_ms', 'min': 'min_scan_ms', 'max': 'max_scan_ms'}, inplace=True)
    latency.to_csv(OUT / 'timing_summary.csv', index=False)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    colors = ['#176b9a', '#6d8a9b', '#6d8a9b', '#9ca9af', '#ce882c', '#d1ac78', '#999999', '#694d87']
    offsets = np.linspace(-.15, .15, len(scan_top))
    for i, (method, color) in enumerate(zip(order, colors)):
        values = scan_top[method].values * 100
        ax.bar(i, values.mean(), width=.62, color=color, alpha=.72)
        ax.scatter(i + offsets, values, color='#222222', s=19, zorder=3)
    ax.set_xticks(range(len(order)), [LABELS[m] for m in order], rotation=23, ha='right')
    ax.set_ylabel('Actual permeability loss (%)')
    ax.set_title('Geometry-selected throats: individual 50% diameter restriction')
    ax.text(.01, .98, 'Bars: equal-weight scan means; dots: five source scans (two crops each)\nPNM sensitivity requires a flow solve; all other selections are solver-free',
            transform=ax.transAxes, va='top', fontsize=9)
    ax.set_ylim(0, max(scan_top.max()) * 100 * 1.35)
    fig.tight_layout()
    savefig(fig, 'model_selected_throat_loss')
    gat_groups = discovery[(discovery.method == 'guided_edge_gat') & (discovery.diameter_reduction == .5)]
    grouped = gat_groups.groupby(['selection_group', 'scan_id', 'sample_id']).actual_loss.mean().groupby(['selection_group', 'scan_id']).mean().unstack('selection_group')
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    for scan, values in grouped.iterrows():
        axes[0].plot(range(3), values[['low', 'random', 'top']].values * 100, marker='o', label=scan)
    axes[0].set_xticks(range(3), ['GAT low', 'Random', 'GAT top'])
    axes[0].set_ylabel('Actual permeability loss (%)')
    axes[0].set_title('Does GAT ranking identify higher-impact throats?')
    axes[0].legend(fontsize=8)
    sorted_accuracy = accuracy.sort_values('common_pool_mae_pp')
    axes[1].barh(sorted_accuracy.label, sorted_accuracy.common_pool_mae_pp, color='#477d9d')
    axes[1].invert_yaxis()
    axes[1].set_xlabel('Fractional-loss MAE (percentage points)')
    axes[1].set_title('Same intervention pool, all three doses')
    fig.tight_layout()
    savefig(fig, 'ranking_and_response_accuracy')
    show = latency[(latency.scope.isin(['five_seed_ensemble_all_throats_three_doses', 'base_solve',
                      'three_fresh_50pct_solves_base_excluded', 'base_and_all_throat_leverage_precompute',
                      'cached_all_throats_three_doses', 'all_throat_scores']))]
    time_labels = [LABELS.get(r.method, r.method) + ': ' + {
        'five_seed_ensemble_all_throats_three_doses': '5-seed ensemble', 'base_solve': 'base solve',
        'three_fresh_50pct_solves_base_excluded': '3 new solves', 'base_and_all_throat_leverage_precompute': 'precompute',
        'cached_all_throats_three_doses': 'cached update', 'all_throat_scores': 'ranking'}[r.scope] for r in show.itertuples()]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(time_labels, show.scan_macro_median_ms, color='#447a8d')
    ax.set_xscale('log')
    ax.invert_yaxis()
    ax.set_xlabel('Warm CPU time (ms, logarithmic scale); one thread')
    ax.set_title('Inference and exact PNM response have different setup costs')
    fig.tight_layout()
    savefig(fig, 'measured_response_cost')
    gat_top = float(discovery_summary.loc[discovery_summary.method == 'guided_edge_gat', 'mean_actual_loss_pp'].iloc[0])
    random_top = float(discovery_summary.loc[discovery_summary.method == 'random', 'mean_actual_loss_pp'].iloc[0])
    affine_top = float(discovery_summary.loc[discovery_summary.method == 'affine_pressure', 'mean_actual_loss_pp'].iloc[0])
    gat_mae = float(accuracy.loc[accuracy.method == 'guided_edge_gat', 'common_pool_mae_pp'].iloc[0])
    mlp_mae = float(accuracy.loc[accuracy.method == 'guided_edge_mlp', 'common_pool_mae_pp'].iloc[0])
    no_mae = float(accuracy.loc[accuracy.method == 'no_response', 'common_pool_mae_pp'].iloc[0])
    def cost(method, scope):
        return float(latency.loc[(latency.method == method) & (latency.scope == scope), 'scan_macro_median_ms'].iloc[0])
    gat_cost = cost('guided_edge_gat', 'five_seed_ensemble_all_throats_three_doses')
    base_cost = cost('pnm', 'base_solve')
    precompute_cost = cost('pnm_exact', 'base_and_all_throat_leverage_precompute')
    cached_cost = cost('pnm_exact', 'cached_all_throats_three_doses')
    verdict = ('The GAT identifies higher-impact throats than the random control on this aggregate endpoint.'
               if gat_top > random_top else 'The GAT does not improve aggregate throat discovery over the random control.')
    analytic_verdict = ('It exceeds the affine-pressure ranking on the aggregate endpoint.' if gat_top > affine_top
                        else 'The affine-pressure ranking identifies larger actual losses than GAT without a flow solve.')
    discovery_table = table(discovery_summary.fillna('n/a'), {'label': 'Selection method', 'mean_actual_loss_pp': 'Actual top-throat loss (%)',
        'ci_low_pp': 'Cluster interval low', 'ci_high_pp': 'Cluster interval high', 'mean_sensitivity_spearman': 'Sensitivity Spearman'})
    accuracy_table = table(accuracy, {'label': 'Response predictor', 'common_pool_mae_pp': 'Common-pool MAE (pp)',
        'ci_low_pp': 'Cluster interval low', 'ci_high_pp': 'Cluster interval high'})
    timing_table = table(latency, {'method': 'Method', 'scope': 'Timed operation', 'scan_macro_median_ms': 'Time (ms)'})
    primary_p = comparisons['guided_edge_gat_minus_affine_pressure']['two_sided_exact_sign_flip_p']
    manuscript = f'''# TopoFlow: solver-free throat discovery — verified manuscript text

## Suggested title

Geometry-Driven Discovery and Response Prediction of Critical Pore Throats: A Controlled Graph-Learning Study

## Abstract

Localized throat restrictions can alter pore-network permeability while leaving a node-only graph representation unchanged. We evaluate an edge-aware graph attention model that predicts global hydraulic importance and single-throat restriction responses from geometry, connectivity and boundary membership. Training uses eight crops from four development scans. A prospectively frozen evaluation uses {len(inventory)} nonoverlapping 128³-voxel crops from five previously unused Imperial College source scans, S2, S4, S6, S7 and S8. Five fixed training seeds are ensembled; matched guided mean-aggregation and edge-MLP models, an unguided GAT, conductance ranking and an affine-pressure ranking provide controls. All geometry-driven throat selections are sealed before flow-derived outcomes are calculated. Individual 50% restrictions of the guided GAT's three highest-ranked throats produce a scan-macro mean permeability loss of {gat_top:.2f}%, compared with {random_top:.2f}% for random selection and {affine_top:.2f}% for affine-pressure ranking. On a common intervention pool spanning 10%, 25% and 50% restrictions, guided GAT response MAE is {gat_mae:.2f} percentage points, compared with {mlp_mae:.2f} for the guided edge MLP and {no_mae:.2f} for no response. Thus, the GAT ranks useful intervention locations but does not beat the strongest analytical ranking or the no-response accuracy control. These results quantify the capabilities and limits of learned restriction surrogates under a specified cylindrical-throat pore-network model; they do not establish physical permeability accuracy or a unique advantage from attention.

## Evaluation procedure

The five scan IDs are source groups, not independently certified geological specimens. Two opposite-corner crops per scan are fixed before extraction; failed geometry attempts are retained in the audit. Eligible throats connect physical pores within a topology-spanning inlet–outlet component. Eligibility does not use pressure, flux or dissipation. Each learned model ranks every eligible throat with its global importance head and selects three highest and three lowest scores. A shared uniform sample supplies three random controls. The importance head differs from native GAT attention coefficients. The conductance control uses d⁴/L; the affine-pressure control ranks g(Δx)², approximating dissipation under an affine pressure field without solving flow. Neither analytical ranking uses a fitted coefficient or flow-derived test inputs.

The source is the official [Imperial College ImagesICPSC2007 archive](https://imperialcollegelondon.box.com/v/ImagesICPSC2007). All five parent arrays contain 300³ voxels. Isotropic voxel sizes for S2, S4, S6, S7 and S8 are 4.956, 8.960, 5.100, 4.803 and 4.892 μm, respectively. Array value zero denotes pore space; whole-volume pore fractions agree with the source-header porosities to their reported precision. Published full-volume permeabilities are not used as labels for the crops. Exact voxel-byte comparisons rule out duplicate crops against the locally available earlier development and external data, without establishing geological independence.

All four model architectures, twenty checkpoints, seed ensemble and evaluation settings are fixed before extracting these validation scans. The guided mean/MLP controls were added after the earlier four-scan pilot, but are predefined for this new validation group. Their training objective, optimizer, guidance weight and development-only epoch-selection rule match the guided GAT; parameter counts and selected epoch counts differ and are disclosed. The mean-aggregation network is edge-conditioned and GraphSAGE-style rather than a literal SAGEConv implementation. The edge MLP has no message passing but shares the global edge-score normalization and bounded response readout.

The GAT uses three single-head, 32-channel edge-aware layers with ELU activations. Seven node inputs encode scale-normalized log diameter and volume, normalized coordinates and inlet/outlet flags. Six throat inputs encode normalized log diameter, length, a d⁴/L proxy and three absolute direction components. A symmetric edge head maps the summed and absolute-difference endpoint embeddings plus throat features through 70→32→2 channels. Its first output is globally softmax-normalized over throats to importance s; the second gives leverage l=(1−s)·sigmoid(z). For a restriction q, r=(1−q)⁴ and the predicted fractional loss is (1−r)s/[1−(1−r)l]. This constrained readout follows the known rank-one identity of a linear resistor network and is not claimed as a new physical law. Training uses AdamW with learning rate 0.003, weight decay 0.0001, batch size two, gradient clipping at five and importance-guidance weight 0.05. Development-only epoch selection yields 15 epochs for guided GAT, 12 for guided mean GNN, 22 for guided edge MLP and seven for unguided GAT. No epoch or seed is chosen using the new scans.

Reference outcomes use fixed-pressure, incompressible linear flow with cylindrical inscribed-diameter throats and direct center-to-center lengths, including SNOW2 boundary conduits. A diameter restriction q changes its conductance by (1−q)⁴. Single-throat responses are independently recomputed by fresh sparse solves; an exact rank-one update provides a numerical cross-check and an efficient solver-based control. Individual restrictions are separate counterfactuals, not simultaneous three-throat narrowing. The primary endpoint averages the three selected throat responses, then crops within a scan, then the five scans equally. Shared-pool accuracy uses the union of all geometry-selected top, low and random throats; solver-oracle selections are excluded. Interventions, crops and training seeds are not treated as independent rock specimens.

## Discovery results

{discovery_table}

{verdict} {analytic_verdict} The paired scan-level sign-flip comparison with affine-pressure ranking gives p={primary_p:.4f}. Only five scan groups are available, so the smallest possible two-sided exact p-value is 0.0625. Bootstrap intervals describe the observed groups and do not establish broad geological generalization.

## Response results

{accuracy_table}

These errors use a different, common selection pool from the previous oracle-selected pilot. The previous 24.8% reduction and the new errors are separate experiments and must not be combined. Reported seed variability is secondary to the fixed ensemble endpoint.

## Computational cost and physical scope

Warm CPU timings include the complete neural ensemble response output and separate exact-PNM setup and reuse costs. The guided GAT ensemble takes {gat_cost:.2f} ms, compared with {base_cost:.2f} ms for a PNM base solve, {precompute_cost:.2f} ms for exact all-throat response precomputation and {cached_cost:.3f} ms for the cached all-throat update. These operations have different scopes and setup requirements, but the measurements establish no runtime advantage for the GAT on these graphs. Extraction, feature conversion, batching and disk/model loading are outside forward-only times. The cached exact update uses flow-derived sensitivity and leverage from a base solve and factorization; it is not solver-free. The complete measured timings accompany the results rather than extrapolating to larger volumes or other hardware.

The previously observed S1/S5 full-volume PNM values remain approximately one-quarter of published numerical reference values. This study neither calibrates away that discrepancy nor establishes matched direct numerical simulation. Every intervention conclusion is conditional on the stated linear cylindrical-throat model. No LBM ground truth, verified physical clogging dynamics, attention-specific superiority or universal novelty is claimed.
'''
    (OUT / 'MANUSCRIPT_INSERT.md').write_text(manuscript, encoding='utf-8')
    report = f'''# TopoFlow — sealed solver-free discovery evaluation

The new experiment addresses the review's oracle-selection problem with a separately frozen five-scan validation group. {verdict} {analytic_verdict} Guided GAT response MAE is {gat_mae:.2f} pp on the shared intervention pool, versus {no_mae:.2f} pp for no response, which has the lowest error among evaluated predictors. The measured runtime also establishes no GAT advantage over the exact PNM alternatives on these graphs. Publication readiness remains limited by the reference-physics discrepancy, small scan count and the strength of simpler controls.

## Evidence and split

- Official Imperial ImagesICPSC2007 archive; original source headers retained; archive and checkpoint SHA-256 recorded in [protocol.json](protocol.json). The [source-byte audit](source_byte_audit.json) checks phase interpretation and rules out exact duplicates against locally available prior crops.
- New scan IDs: S2, S4, S6, S7, S8. These were not in the earlier development or external experiments. This is a sequential follow-up motivated by that pilot, not a retrospectively pristine overall research program.
- {len(inventory)} accepted geometry crops, five source-scan groups, two fixed opposite-corner attempts per scan. No replacement search or outcome-based crop exclusion.
- Four frozen architectures, five seeds each, zero retraining. Training remains eight development crops from four earlier scans.
- [Selection seal](selection_seal.json) timestamps and hashes all rankings and selections before [outcome evaluation](evaluation_started.json). Selection performs zero flow solves. Geometry and frozen checkpoint identities are verified again before outcomes.
- {complete['fresh_intervention_solves']} fresh numerical intervention solves check the exact finite-change identity. Repeated solves validate numerics; they do not increase the number of independent specimens.

## Primary result: actual loss at model-selected throats

Each value is the mean loss from restricting each selected throat separately by 50%. Percentages describe physical loss within the PNM, not prediction error.

{discovery_table}

![Model-selected throat loss](figures/model_selected_throat_loss.png)

The random control is identical across methods. The PNM sensitivity ranking uses true base-flow dissipation and is explicitly solver-based; it is a first-order sensitivity control, not a guaranteed optimum for finite restrictions. The other rankings are computed from geometry before outcomes. See [per-scan values](per_scan_discovery.csv) and [paired comparisons](paired_scan_comparisons.json).

## Fair response comparison

All response predictors use the same intervention pool in each graph, covering every geometry-selected top/low/random throat at three doses. Graph means are averaged within scan and scan means equally; the number of throats selected by different methods cannot change the weighting. The affine/conductance response controls use normalized proxy importance and heuristic leverage (1−s)/2; they are ranking controls with an explicit heuristic response, not exact hydraulic solutions.

{accuracy_table}

![Ranking and common-pool accuracy](figures/ranking_and_response_accuracy.png)

[Seed response errors](seed_accuracy_summary_pp.csv) report mean/standard deviation across five single-seed scan-macro errors. They do not select the deployed seed or establish an independent sample size of 25.

## Measured costs

One CPU thread, five warmups, thirty measured calls per graph; exact all-throat precomputation uses five measured calls. Each table entry first takes the median calls in a graph, then averages the two crops within scan, then averages scans equally. The ensemble operation computes all-throat responses at all three doses. Three fresh PNM solves compute three 50% responses with the base permeability already available; their scope differs and is stated rather than equated. Exact precompute includes a base solve, factorization and all-edge leverage; cached readout excludes that setup.

{timing_table}

Mean SNOW2 extraction is {inventory.snow2_ms.mean()/1000:.3f} s per crop; feature construction is {inventory.feature_ms.mean():.3f} ms. Forward timing excludes graph batching, extraction and model/disk load. All twenty models stay resident during benchmarking. [resources.json](resources.json) reports observed process RSS and exact tensor/parameter bytes. RSS is not peak or incremental model memory.

![Response calculation costs](figures/measured_response_cost.png)

## What the review now resolves

| Concern | New evidence | Remaining limit |
| --- | --- | --- |
| Solver-selected important test throats | Every deployed ranking and top/low/random selection sealed before outcomes | Teacher guidance remains PNM-derived during training |
| Reused external tests | New five-scan group, frozen prior checkpoints and matched controls | All scans are from the same archive; geology independence not certified |
| Unequal guidance | Guided GAT, mean GNN and edge MLP all fixed before new outcomes | Architecture sizes/selected epochs differ; no attention-specific conclusion |
| No comparable runtime evidence | Warm neural, fresh-solve, exact-precompute and cached-update timings | Small CPU graphs, no extrapolated speedup |
| Physical reference discrepancy | Conclusions explicitly restricted to declared PNM | S1/S5 discrepancy remains unresolved; no DNS or real-clogging validation |

## Relation to the literature

[Zhao and Xiao (2026)](https://doi.org/10.1016/j.jhydrol.2026.135602) combine learned conductances with a PNM and analyze permeability sensitivities. [Nassiri, van Paassen and Mahabadi (2026)](https://doi.org/10.1016/j.compgeo.2026.108292) examine location- and connectivity-dependent clogging responses. These primary publisher records were checked for this follow-up. Edge features, sensitivity and clogging-location effects therefore cannot supply a stand-alone novelty claim. The narrower question tested here is blind geometry-only selection and bounded counterfactual response prediction with matched controls. This literature check does not establish a universal first.

## Submission assessment

The result supports a controlled empirical study of where learned throat-response surrogates help or fail. It does not establish a new best GAT method. A workshop or exploratory paper can foreground the representation blind spot, sealed discovery experiment, analytical controls and exact-update comparison. A full methods paper still needs a convincing benefit over those controls, stronger independent source coverage and independently validated physics. The new [manuscript insert](MANUSCRIPT_INSERT.md) states completed methods and results directly, without presenting unperformed experiments as findings.
'''
    (OUT / 'REPORT.md').write_text(report, encoding='utf-8')
    write_json(OUT / 'report_verification.json', {'evaluation_complete_sha256': sha256(OUT / 'evaluation_complete.json'),
        'graphs': len(inventory), 'scan_ids': sorted(scan_top.index.tolist()),
        'primary_endpoint': discovery_summary.drop(columns=['mean_sensitivity_spearman']).to_dict('records'),
        'common_pool_rows': len(prediction), 'primary_ensemble_fixed': True,
        'source': str(Path(__file__).relative_to(ROOT)), 'source_sha256': sha256(__file__)})
    print(discovery_summary.to_string(index=False))
    print(accuracy.to_string(index=False))
    print('Built report, manuscript insert and three figures in PNG/PDF/SVG.', flush=True)


if __name__ == '__main__':
    main()
