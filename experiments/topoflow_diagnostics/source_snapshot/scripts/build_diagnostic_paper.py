"""Consolidate completed experiments as a comparative manuscript, without new fitting."""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_discovery_review import OUT, OLD, FEATURES
from scripts.build_discovery_report import table, LABELS


def save(fig, name):
    (OUT / 'figures').mkdir(exist_ok=True)
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(OUT / 'figures' / f'{name}.{ext}', dpi=220, bbox_inches='tight')
    plt.close(fig)


def main():
    scan = pd.read_csv(OUT / 'network_diagnostics_per_scan.csv')
    associations = pd.read_csv(OUT / 'exploratory_associations.csv')
    response = pd.read_csv(OUT / 'response_strata.csv')
    distribution = pd.read_csv(OUT / 'outcome_distribution.csv')
    native = pd.read_csv(OUT / 'original_network_reference.csv')
    variants = pd.read_csv(OUT / 'crop_physics_variants.csv')
    checks = pd.read_csv(OUT / 'independent_solver_checks.csv')
    discovery = pd.read_csv(OLD / 'discovery_metrics.csv')
    cost = pd.read_csv(OLD / 'timing_summary.csv')
    low_n = int(distribution[distribution.group == 'low_under_0.1pp'].interventions.iloc[0])
    total_n = int(distribution.interventions.sum())
    def error(method, group, column='mae_pp'):
        return float(response.loc[(response.method == method) & (response.group == group), column].iloc[0])
    high_gat = error('guided_edge_gat', 'actual_high_at_least_1pp')
    high_zero = error('no_response', 'actual_high_at_least_1pp')
    top_gat = error('guided_edge_gat', 'gat_selected_top')
    top_zero = error('no_response', 'gat_selected_top')
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    names = ['Shortest-path tortuosity (lower bound)', 'Conductance-weighted axial alignment',
             'Physical-network degree CV', 'Min edge cut / eligible throats']
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, feature, label in zip(axes.flat, FEATURES, names):
        ax.axhline(0, color='grey', linewidth=.8)
        ax.scatter(scan[feature], scan.gat_minus_affine_loss_pp, color='#276f91')
        for row in scan.itertuples():
            ax.annotate(row.scan_id, (getattr(row, feature), row.gat_minus_affine_loss_pp),
                        xytext=(5, 4), textcoords='offset points', fontsize=9)
        rho = associations.loc[associations.feature == feature, 'spearman_with_gat_advantage'].iloc[0]
        ax.set_xlabel(label)
        ax.set_ylabel('GAT − affine loss (pp)')
        ax.set_title(f'Post-hoc scan association: Spearman {rho:.2f}')
    fig.suptitle('Five reused test scans: exploratory associations, no fitted selector', fontsize=12)
    fig.tight_layout()
    save(fig, 'topology_and_ranking_advantage')
    methods = ['guided_edge_gat', 'guided_edge_sage', 'guided_edge_mlp', 'edge_gat', 'affine_pressure', 'no_response']
    groups = ['actual_low_under_0.1pp', 'actual_moderate_0.1_to_1pp', 'actual_high_at_least_1pp']
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.7))
    for ax, group, title in zip(axes, groups, ['Actual loss < 0.1 pp', 'Actual loss 0.1–1 pp', 'Actual loss ≥ 1 pp']):
        part = response[response.group == group].set_index('method').loc[methods]
        ax.barh([LABELS[m] for m in methods], part.mae_pp, color=['#216b93'] + ['#8c9da5'] * 4 + ['#c08c4b'])
        ax.invert_yaxis()
        ax.set_xlabel('MAE (pp; separate panel scales)')
        ax.set_title(f'{title}\n{int(part.interventions.iloc[0])} interventions, {int(part.scan_groups_present.iloc[0])} scans')
    fig.tight_layout()
    save(fig, 'impact_stratified_response_error')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.7))
    labels = ['Circular: total length', 'Circular: series conduit', 'Shape-factor conduit']
    for ax, sample in zip(axes, ['S1', 'S5']):
        part = native[native['sample'] == sample]
        ax.bar(range(3), part.k_mD, color=['#829aa5', '#4b819b', '#b17a38'])
        reference = part.published_network_k_mD.iloc[0]
        ax.axhline(reference, color='#333333', linestyle='--', label=f'Published network: {reference:g} mD')
        ax.set_xticks(range(3), labels, rotation=20, ha='right')
        ax.set_ylabel('Permeability (mD)')
        ax.set_title(sample + ': original maximal-ball network')
        ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, 'original_network_closure_audit')
    response_rows = []
    for method in ['guided_edge_gat', 'guided_edge_sage', 'guided_edge_mlp', 'edge_gat', 'affine_pressure', 'conductance', 'no_response']:
        response_rows.append({'label': LABELS[method], 'all': error(method, 'all'),
            'low': error(method, groups[0]), 'moderate': error(method, groups[1]),
            'high': error(method, groups[2]), 'severe': error(method, 'actual_at_least_5pp')})
    response_table = table(pd.DataFrame(response_rows), {'label': 'Predictor', 'all': 'All', 'low': '<0.1 pp',
        'moderate': '0.1–1 pp', 'high': '≥1 pp', 'severe': '≥5 pp subset'})
    topology_table = table(scan, {'scan_id': 'Scan', 'gat_top_loss_pp': 'GAT loss (%)', 'affine_top_loss_pp': 'Affine loss (%)',
        'geodesic_tortuosity_lower_bound': 'Path tortuosity', 'axial_alignment_conductance_weighted': 'Axial alignment', 'degree_cv': 'Degree CV'})
    native_table = table(native, {'sample': 'Sample', 'variant': 'Closure on original network', 'k_mD': 'Computed mD',
        'published_network_k_mD': 'Published network mD', 'ratio_to_published_network_k': 'Ratio'})
    variant_summary = variants.groupby('variant').ratio_to_original_crop_pnm.agg(['mean', 'min', 'max']).reset_index()
    variant_table = table(variant_summary, {'variant': 'Crop diagnostic variant', 'mean': 'Mean ratio', 'min': 'Minimum', 'max': 'Maximum'})
    assoc_table = table(associations, {'feature': 'Geometry diagnostic', 'spearman_with_gat_advantage': 'Spearman with GAT advantage'})
    max_error = max(checks.relative_difference.max(), native.openpnm_relative_difference.max())
    high_count = response[(response.method == 'no_response') & (response.group == groups[2])].iloc[0]
    extreme = response[(response.method == 'no_response') & (response.group == 'actual_at_least_5pp')].iloc[0]
    report = f'''# TopoFlow — response to review of 4c48050

The review is supported by the new diagnostics. The GAT ranks useful throats but its response estimates are poorly calibrated. Neither an overall GAT advantage nor a validated topology-dependent switching rule has been established. Numerical solver checks now separate implementation agreement from the unresolved physical-reference issue. The findings are consolidated in [MANUSCRIPT.md](MANUSCRIPT.md).

## Scope and integrity

This is a post-hoc analysis of the same five test scans, not a new validation cohort. [analysis_plan.json](analysis_plan.json) records that the winner labels and one S2 debugging example were already known. No neural weights, calibration coefficients or switching rules are fitted. The original protocol, checkpoints and selection seal remain unchanged. The same twenty checkpoints and all original frozen predictions are retained.

## 1. Where does the learned ranking help?

{topology_table}

Path tortuosity is the shortest geometric inlet–outlet path length divided by boundary-plane separation, including the declared boundary conduits. It is a lower bound and is not bulk hydraulic tortuosity. Axial alignment weights squared axial direction cosines by d⁴/L on eligible internal throats. Degree CV describes active physical-pore connectivity. The edge-cut metric is an unweighted topological cut between inlet-adjacent and outlet-adjacent physical pores, normalized by eligible throat count; it is not a hydraulic min-cut.

{assoc_table}

The two scans favoring GAT have larger shortest-path tortuosity than the other three. The scan-level correlation with advantage is 0.60. Axial alignment and degree CV have the opposite signs to the candidate directional hypotheses recorded in the plan; cut density has the proposed negative sign. These mixed observations do not identify a reliable regime, threshold or mechanism. No fitted boundary, selector or causal claim is reported. All four candidate associations are shown, including those contradicting the hypotheses. The sample size remains five scan groups.

![Topology diagnostics](figures/topology_and_ranking_advantage.png)

The affine energy-excess and pressure-error fields in the CSV use solved pressures and are outcome diagnostics only. They are never proposed as solver-free test-time inputs. A three-layer GAT also does not establish learned long-range hydraulic reasoning merely because its importance head is globally normalized.

## 2. Does low-impact imbalance explain the response result?

There are {total_n} distinct throat-dose interventions in the common pool. Of these, {low_n} ({100*low_n/total_n:.1f}%) produce less than 0.1 pp actual loss, 117 produce 0.1–1 pp, and 78 produce at least 1 pp. The bins are fixed, interpretable post-hoc descriptions of outcomes; they do not alter selections or enter the model. The ≥5 pp subset contains {int(extreme.interventions)} interventions in {int(extreme.scan_groups_present)} scans and overlaps the ≥1 pp group.

All table entries are response MAE in percentage points, averaged within graph, then within scan, then across available scans equally. Counts differ between bins. The ≥1 pp group covers {int(high_count.graphs_present)} graphs in five scans; zero-count graphs are explicitly retained in the per-graph CSV and do not receive artificial zero errors. Every method uses identical cases in each comparison.

{response_table}

The imbalance makes the no-response control competitive, but does not account for the full GAT deficit. Guided GAT MAE is {high_gat:.2f} pp on ≥1 pp interventions versus {high_zero:.2f} for no response, with signed bias {error('guided_edge_gat', groups[2], 'bias_pp'):+.2f} pp. On the GAT's own frozen top selections, evaluated for all three doses, its MAE is {top_gat:.2f} pp versus {top_zero:.2f} for no response. Predicted loss on those selections averages {error('guided_edge_gat','gat_selected_top','mean_predicted_pp'):.2f} pp while actual loss averages {error('guided_edge_gat','gat_selected_top','mean_true_pp'):.2f} pp. Ranking information therefore does not translate into calibrated amplitudes.

Some simpler learned controls improve on no response within the ≥1 pp subset, despite losing on the overall mixture. This supports reporting both consequential-intervention and whole-pool errors. The marginal GAT improvement in the small ≥5 pp tail is a threshold-sensitivity observation, not a replacement headline or an independently confirmed benefit. No correction is fitted to these test outcomes.

![Stratified response errors](figures/impact_stratified_response_error.png)

## 3. Physical-reference audit

Our solver and an independently assembled OpenPNM StokesFlow solve agree with a maximum relative difference of {max_error:.2g} across ten SNOW crop networks and six original-network closure cases. These checks hold conductances, dimensions, viscosity and pressure conditions fixed. They make a simple linear-system assembly or Darcy-unit error an unlikely explanation for the discrepancy on the checked systems. They do not verify extracted geometry or real-rock permeability.

Four fixed crop diagnostics change one modeling assumption at a time. No factor is fitted to a reference:

{variant_table}

Changing diameter definition has a large effect; removing artificial boundary-node resistance has a smaller but variable effect. These are crop sensitivity checks, not a correction to the full S1/S5 labels. Changing the closure would also change the intervention teacher and require a new declared study.

The archive supplies the original maximal-ball networks with pore/throat radii, shape factors and three conduit sub-lengths. Circular total-length, circular series-conduit and OpenPNM Valvatne–Blunt closures are compared on the same original graph. Endpoint-associated sub-lengths are reordered whenever endpoint IDs are reordered. The shape-factor calculation uses the installed OpenPNM model; a circular-tube limiting-case test verifies its interface.

{native_table}

![Original-network closure effects](figures/original_network_closure_audit.png)

The original-network shape-factor results are closer to the published scalar network values than either circular closure, but still differ by 12.5% for S1 and 33.5% for S5. Reservoir throats are removed and pressures imposed on their adjacent physical pores, following the OpenPNM importer convention; the legacy calculation's exact reservoir treatment is not reproduced. The original networks' boundary displacement is predominantly native coordinate x. The earlier SNOW result compared array-axis-0 flow nominally with published image Kz. These are different extraction and boundary/direction conventions. Consequently this is evidence of strong closure sensitivity, not an exact error decomposition or proof that the fourfold SNOW/image discrepancy is fixed.

The full-volume SNOW baselines remain 342.49 and 1187.48 mD, versus published image Kz values 1312 and 4440 mD. No new full SNOW extraction was attempted with only 0.8 GiB available at setup; the prior jobs peaked near 2.4 GB. The supplied original networks permit the cross-check without that extraction. The physical-reference discrepancy remains unresolved and the paper consistently states PNM-conditional conclusions.

## 4. Paper consolidation

The [comparative manuscript](MANUSCRIPT.md) preserves micro-CT, SNOW2, GAT and permeability-response prediction, while leading with the representation blind spot and rigorous learned-versus-analytical comparison. It contains completed findings rather than promises of unperformed results. It includes all three authors and source citations. The historical 24.8% oracle-pilot improvement is not substituted for the new frozen test results.

[VALIDATION_PROTOCOL.md](VALIDATION_PROTOCOL.md) separates an unexecuted prospective test from the manuscript. It specifies new independent source specimens, a continuous tortuosity association and development-only calibration; it does not fit a rule to S2/S4/S6/S7/S8.

The work is stronger as an exploratory comparative or benchmarking submission than as a claim of a superior GAT method. A publication decision still depends on venue fit and reviewer assessment. No numerical acceptance score or guarantee is assigned.
'''
    (OUT / 'REPORT.md').write_text(report, encoding='utf-8')
    manuscript = f'''# Geometry-Driven Discovery and Response Prediction of Critical Pore Throats: A Controlled Graph-Learning Study

Shaunak Rane · Tanisha Somai · Hemant Maddu

## Abstract

Localized pore-throat restrictions can change permeability without changing the node features or connectivity used by a node-only graph surrogate. We investigate this representation limitation and compare edge-aware graph attention with analytical throat ranking and matched learned controls. Networks are extracted from Imperial College micro-CT images using SNOW2. Models are trained on eight crops from four development scans and evaluated with fixed five-seed ensembles on ten predetermined crops from five previously unused scans, S2, S4, S6, S7 and S8. Throat selections are sealed before reference-flow outcomes are calculated. Individually narrowing the guided GAT's three highest-ranked throats by 50% produces a scan-balanced mean permeability loss of 3.57%, versus 0.37% for random selection and 4.08% for a solver-free affine-pressure ranking. On a common intervention pool, guided GAT response MAE is 2.04 percentage points, versus 0.46 for no response. Stratification shows that this deficit persists for interventions causing at least 1 percentage point of loss: {high_gat:.2f} versus {high_zero:.2f} points. A post-hoc topology analysis finds an association between geometric path tortuosity and ranking advantage, without establishing a generalizable regime. Independent flow assembly checks agree numerically, while closure comparisons on supplied original networks demonstrate substantial modeling sensitivity. The study identifies useful learned ranking information and poor response calibration under a declared cylindrical-throat PNM; it establishes neither attention-specific superiority nor validated physical clogging accuracy.

Keywords: digital rock; graph attention; pore-network model; permeability; counterfactual restriction; analytical baseline.

## 1. Introduction

Pore networks reduce micro-CT images to geometric and topological representations on which transport can be approximated. Both maximal-ball extraction and watershed-based extraction have established precedents [1,2]. Their geometric simplifications influence hydraulic labels, so agreement with a particular pore-network calculation must be separated from agreement with voxel-resolved flow or measured permeability.

A localized restriction introduces a second distinction: accurate bulk-permeability regression does not ensure a model can represent changes in individual throats. If only pore diameter, pore volume and connectivity enter a model, partial throat narrowing leaves its input unchanged. Such a model returns exactly the same output even when the reference network flow changes. This is an information limitation independent of optimizer quality or attention weights.

TopoFlow addresses this question with an edge-aware restriction surrogate and a controlled evaluation. We ask whether geometry-driven learned importance identifies influential throats without solving a new test flow, whether a constrained readout predicts restriction magnitude, and how either task compares with analytical alternatives. The contribution is empirical: a representation check, sealed geometry-only selections, shared-pool response evaluation and an audit separating numerical solver consistency from reference-model uncertainty. An overall GAT win is not a premise of the study.

## 2. Related work

Graph attention weights neighborhood contributions during message passing [3]. A separately trained global importance head should not be equated with those neighborhood-normalized coefficients. Pore-network extraction methods already encode geometrical and connectivity information relevant to transport [1,2]. Learning edge conductance and analyzing permeability sensitivity are also established directions: Zhao and Xiao embed learned conductances in a differentiable pore-network model [4]. Location and connectivity effects in biogeochemical clogging have been studied with pore-scale network simulations [5].

Our comparison concerns geometry-only throat discovery and bounded single-throat response prediction, including simpler controls that receive equivalent inputs and supervision. The single-throat rank-one response identity is a property of the linear reference network, not a newly discovered physical law. The literature reviewed here does not establish a universal first for this experimental question.

## 3. Materials and methods

### 3.1. Images and source separation

The official Imperial College collection supplies the segmented source images and original maximal-ball networks [6]. Development uses the two lexicographically first accepted crops from each of Berea, C1, C2 and Estaillades. Estaillades comes from the separate official 1000³-voxel archive at 3.31136 μm resolution [8]. The restriction model therefore trains on eight graphs, not on the larger permeability-regression benchmark. The earlier S1/S3/S5/S9 pilot informs this follow-up but is not the new validation group.

The frozen discovery test uses S2, S4, S6, S7 and S8, with two 128³-voxel cubes taken from opposite corners of each 300³ source volume. Voxel sizes are respectively 4.956, 8.960, 5.100, 4.803 and 4.892 μm. Pore space is encoded by zero in the source arrays. Whole-volume pore fractions agree with source-header porosities to reported precision. All ten predetermined crops pass the geometry criteria; no fallback search is used. Exact voxel-byte comparisons rule out duplicate crops against 56 earlier available natural-rock crops. Named source scans are grouping units, not independently certified geological specimens.

### 3.2. Extraction and hydraulic reference

SNOW2 uses sigma 0.4, r_max 4 and boundary width [3,0,0], with one extraction division. The reference assigns cylindrical Hagen–Poiseuille conductance to each throat:

g = πd⁴/(128μL),

where d is inscribed diameter, L is direct pore-center distance and μ is 0.001 Pa·s. Inlet and outlet pressures are 1 and 0 Pa. Nonspanning components are excluded and conservation residuals checked. Darcy permeability uses the original physical image dimensions. This closure includes artificial boundary conduits and is explicitly a PNM approximation, not LBM or measured ground truth.

For a single diameter restriction q, conductance is multiplied by r=(1−q)⁴ while other properties remain fixed. If s denotes normalized baseline dissipation and l the Dirichlet edge leverage, the exact fractional permeability loss in this linear model is

loss = (1−r)s / [1−(1−r)l].

Fresh sparse solves independently check the finite-change identity for 966 distinct selected throat-dose interventions, with maximum fractional discrepancy below 4.5×10⁻⁸. Individual throat restrictions are separate counterfactuals, not simultaneous narrowing of three throats or a time-resolved clogging process.

### 3.3. Model and controls

Seven node features encode scale-normalized log equivalent diameter and volume, normalized coordinates and inlet/outlet flags. Six edge features encode normalized log inscribed diameter, direct length, a d⁴/L proxy and three absolute direction components. The diameter scale is the geometric mean over physical pores. No pressures, fluxes, teacher sensitivities or leverages enter inference.

The GAT has three single-head, 32-channel edge-aware layers with ELU activation. A symmetric edge head uses endpoint-embedding sums and absolute differences plus throat features, followed by 70→32→2 channels. The first output is globally softmax-normalized to predicted importance s; the second gives l=(1−s)·sigmoid(z). This parameterization constrains the response to be bounded and monotone. It does not guarantee that importance or leverage matches the reference values.

Matched guided controls replace the GAT encoder with an edge-conditioned mean aggregator or a node MLP. The latter omits message passing but retains symmetric edge features, global normalization and the response readout. The mean model is GraphSAGE-style, rather than a literal SAGEConv implementation. An unguided GAT separates guidance from architecture. AdamW uses learning rate 0.003, weight decay 0.0001, batch size two and gradient clipping at five. Response MSE uses three teacher-high and three random internal throats per development graph at three restriction doses; guided models add all-throat importance cross-entropy with weight 0.05. Teacher-based training selection is distinct from the sealed geometry-only test selection. Development-parent validation selects 15, 12, 22 and seven epochs for guided GAT, guided mean GNN, guided MLP and unguided GAT respectively. All twenty checkpoints are fixed before the five new scans are extracted. Test ranking averages importance across the five seeds; responses are computed for each seed and then averaged, preserving the nonlinear readout.

The conductance ranking uses d⁴/L. Affine-pressure ranking uses g(Δx)², corresponding to an assumed affine pressure gradient and requiring no flow solve. Their response controls use normalized proxy importance with heuristic leverage (1−s)/2; these are not exact hydraulic solutions. A solver-based dissipation ranking and an exact response update are labeled separately. No-response prediction is retained as an essential accuracy control.

### 3.4. Frozen selection and evaluation

Eligible throats join physical pores in a topology-spanning component; eligibility uses no hydraulic outcome. Each method selects its own three highest and three lowest importance scores, with stable index tie-breaking, plus three shared uniform-random throats. Rankings and selected indices are saved and hashed before reference outcomes are calculated. The primary discovery endpoint averages the actual loss from each top-selected throat at 50% restriction. Crops are averaged within scan and the five scans equally.

Response accuracy uses the same union of all geometry-selected top, low and random throats at 10%, 25% and 50% restriction. This yields {total_n} unique interventions. Oracle-selected throats are excluded from that shared pool. Interventions, doses, crops and training seeds are not treated as independent rock samples. Exact paired scan sign-flip comparisons have a minimum attainable two-sided p-value of 0.0625 for five groups. Bootstrap intervals describe these observed groups and do not establish general geological coverage.

### 3.5. Post-hoc diagnostic scope

After observing the five-scan results, we examine four candidate geometry associations: shortest-path tortuosity, conductance-weighted axial alignment, degree CV and normalized unweighted edge-cut size. The tortuosity is a geometric lower bound, not a flow-weighted bulk property. All four associations are reported without fitting a selector. Response outcomes are described with fixed bins below 0.1 pp, 0.1–1 pp and at least 1 pp; a 5 pp subset checks sensitivity to the definition of consequential loss. These are post-hoc descriptions, not a second untouched test.

## 4. Results

### 4.1. Representation and throat discovery

Partial throat restrictions leave the original node-only model's inputs unchanged; its predicted response is therefore exactly zero. The edge-aware representation removes that input invariance. On the frozen five-scan discovery test, the guided GAT selects more consequential throats than uniform random selection, but the affine-pressure control gives a larger aggregate loss:

{table(discovery.fillna('n/a'), {'label':'Selection method','mean_actual_loss_pp':'Actual top-throat loss (%)','ci_low_pp':'Descriptive interval low','ci_high_pp':'Descriptive interval high'})}

![Actual loss from model-selected throat restrictions](../topoflow_discovery/figures/model_selected_throat_loss.png)

GAT exceeds affine ranking on S2 and S4 but loses on S6, S7 and S8. Their paired comparison gives p=0.625; guided GAT versus guided MLP gives p=0.5625. No statistically established architecture winner is inferred. Native GAT attention coefficients are not the global scores used for selection.

### 4.2. Response prediction and calibration

The common pool is dominated by weak interventions: {low_n}/{total_n} ({100*low_n/total_n:.1f}%) cause less than 0.1 pp loss. Overall guided-GAT MAE is 2.04 pp, versus 0.46 pp for no response. Outcome-stratified errors show that imbalance is not the only problem:

{response_table}

![Response errors separated by intervention impact](figures/impact_stratified_response_error.png)

The ≥1 pp stratum contains 78 interventions across nine graphs and all five scans; the ≥5 pp subset contains {int(extreme.interventions)} interventions in {int(extreme.scan_groups_present)} scans. Aggregation uses available graph means within scan and available scans equally, with zero-count groups retained in the audit. For guided GAT, the ≥1 pp bias is {error('guided_edge_gat',groups[2],'bias_pp'):+.2f} pp. On its own frozen top selections across three doses, predicted loss averages {error('guided_edge_gat','gat_selected_top','mean_predicted_pp'):.2f} pp versus actual {error('guided_edge_gat','gat_selected_top','mean_true_pp'):.2f} pp; MAE is {top_gat:.2f} pp. Thus, informative ordering coexists with substantial amplitude overestimation. No calibration correction is fitted to the test set.

### 4.3. Exploratory geometry associations

{topology_table}

The two scans favoring GAT have greater geometric shortest-path tortuosity than the other three. Across five scan means the association with GAT-minus-affine loss is Spearman 0.60. Corresponding coefficients for axial alignment, degree CV and cut density are 0.30, −0.50 and −0.30. The alignment and degree signs contradict the initial directional hypotheses recorded for this post-hoc analysis. These observations do not establish a regime threshold or explain the winners causally. Three message-passing layers and global score normalization alone do not demonstrate learned long-range flow reasoning.

![Post-hoc geometry associations across five reused scans](figures/topology_and_ranking_advantage.png)

### 4.4. Numerical consistency and physical uncertainty

Ten existing SNOW crop networks and six original-network closure cases are independently assembled in OpenPNM StokesFlow with the same conductances and boundary values. The largest relative permeability difference is {max_error:.2g}. This checks the numerical implementation, not the physical validity of the conductance closure.

The supplied original S1/S5 maximal-ball graphs permit a separate closure comparison. Pore/throat radii, shape factors and three conduit sub-lengths are taken directly from the source files; endpoint lengths remain aligned when connections are sorted. The shape-factor closure is evaluated using OpenPNM's Valvatne–Blunt implementation [7], without fitting a multiplier:

{native_table}

The shape-factor closure is closer to the published scalar network values, but differs by 12.5% and 33.5%. Reservoir throats are removed and Dirichlet values placed on adjacent physical pores, while the legacy exact convention remains unverified. Native original-network boundaries are predominantly along coordinate x; the earlier SNOW comparison maps array-axis-0 to source z. Extraction, boundary and direction differences prevent interpreting this table as an exact reconciliation of the SNOW/image reference gap.

![Closure dependence on original supplied networks](figures/original_network_closure_audit.png)

The previously committed full-volume SNOW predictions remain 342.49 and 1187.48 mD, compared with published image Kz values 1312 and 4440 mD. Crop diagnostics further show mean permeability ratios of 3.64 when replacing inscribed by equivalent throat diameter, 0.93 when replacing direct by total length, and 1.18 when contracting boundary conduits. These changes illustrate modeling sensitivity and are not adopted as corrected labels or independent physical validation.

### 4.5. Computational cost

The frozen benchmark measures one CPU thread, five warmups and thirty calls per graph; exact precomputation uses five measured calls. Median graph times are averaged within scan and across scans. The guided five-seed GAT ensemble takes 25.69 ms for all-throat, three-dose responses; a single seed takes 5.31 ms. A PNM base solve takes 2.60 ms, exact all-throat precomputation 13.67 ms and its cached response readout 0.058 ms. Three fresh 50% solves take 7.81 ms with base permeability available. Operation scopes differ, and the exact update needs flow-derived quantities, but these measurements establish no GAT speed advantage on the tested networks. SNOW2 extraction averages 6.76 s per crop and is excluded from forward-only timing.

## 5. Discussion

The results separate three tasks often conflated in permeability-surrogate studies: representing a geometry perturbation, locating influential throats, and predicting its amplitude. Adding throat features resolves an input blind spot, but does not establish useful response calibration. The GAT's ordering is informative relative to random selection while a simple affine-pressure ranking remains competitive. A constrained response formula supplies physical bounds, yet learned importance and leverage still require accurate estimation.

The outcome distribution explains why no response is a strong overall baseline. Stratification also reveals that the guided GAT underperforms within consequential interventions, whereas some simpler learned controls improve in that subset. Selectively reporting the favorable severe tail or the earlier oracle-selected pilot would obscure these distinctions. The same transparent groups and competing controls are retained throughout.

Numerical agreement between two assemblies is valuable but insufficient physical validation. Changing closure on a fixed source graph produces large permeability changes, and the full SNOW/image discrepancy is not resolved. Our restriction targets remain conditional on cylindrical inscribed-diameter throats and the specified boundaries. Real precipitation, biofilms, geometry evolution, pore-filling and multiphase flow are outside the experiment.

The study has eight training graphs, five new validation source groups and a later post-hoc diagnostic analysis of those groups. Source IDs do not certify independent geology. Diagnostic associations, bootstrap intervals and occasional per-scan wins cannot establish population-level attention superiority, a universal topology threshold or a deployment selector. The supported contribution is the controlled comparison and explicit demonstration of these limits.

## 6. Conclusion

An edge-aware representation enables a response to throat geometry changes that a node-only surrogate cannot observe. In the frozen five-scan evaluation, guided GAT identifies more consequential throats than random selection, but fails to beat the affine-pressure ranking in aggregate and overestimates response amplitudes. Independent numerical checks agree while closure-dependent physical uncertainty persists. Separating ranking, calibration and reference-model error produces a more defensible assessment of graph-learning permeability surrogates than a single accuracy or inference-time headline.

## Reproducibility

Source archives, checksums, fixed checkpoints, selection seals, per-intervention outcomes, error strata, timing calls, model configurations and source snapshots are linked from the [frozen discovery study](../topoflow_discovery/README.md) and [diagnostic evidence](README.md). Scripts reproduce all tables and figures. Raw images and derived networks remain outside the Git release; the source archive is downloaded from its official collection. The new diagnostics fit no parameters, change no checkpoints and do not overwrite the original frozen study.

## References

1. H. Dong and M. J. Blunt, “Pore-network extraction from micro-computerized-tomography images,” Physical Review E, 80, 036307, 2009. [DOI](https://doi.org/10.1103/PhysRevE.80.036307).
2. J. T. Gostick, “Versatile and efficient pore network extraction method using marker-based watershed segmentation,” Physical Review E, 96, 023307, 2017. [DOI](https://doi.org/10.1103/PhysRevE.96.023307).
3. P. Veličković et al., “Graph Attention Networks,” ICLR, 2018. [Author manuscript](https://arxiv.org/abs/1710.10903).
4. Q. Zhao and H. Xiao, “An end-to-end differentiable, graph neural network-embedded pore network model for permeability prediction,” Journal of Hydrology, 675, 135602, 2026. [DOI](https://doi.org/10.1016/j.jhydrol.2026.135602).
5. S. Nassiri, L. van Paassen and N. Mahabadi, “Evolution of porous media and porosity–permeability relations during biogeochemical clogging processes: A pore-scale simulation study,” Computers and Geotechnics, 198, 108292, 2026. [DOI](https://doi.org/10.1016/j.compgeo.2026.108292).
6. Imperial College London, Pore-scale Modelling Group, official ImagesICPSC2007 collection and source headers. [Archive](https://imperialcollegelondon.box.com/v/ImagesICPSC2007).
7. OpenPNM, Valvatne–Blunt hydraulic-conductance implementation; installed version 3.6.0. [Official implementation documentation](https://openpnm.org/_modules/openpnm/models/physics/hydraulic_conductance/_funcs.html#valvatne_blunt).
8. Imperial College London, Estaillades 1000³ micro-CT image, 3.31136 μm voxel size. [Official source archive](https://imperialcollegelondon.box.com/shared/static/2n4lb95amz7fstk19q826uzu922eb4b1.zip).
'''
    (OUT / 'MANUSCRIPT.md').write_text(manuscript, encoding='utf-8')
    print('Built diagnostic report, comparative manuscript and three figures.', flush=True)


if __name__ == '__main__':
    main()
