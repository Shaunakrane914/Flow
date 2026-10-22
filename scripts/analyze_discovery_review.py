"""Post-hoc consolidation of commit 4c48050; no retraining or winner selection."""
from datetime import datetime, timezone
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import zipfile

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.experiment_data import sha256
from src.research_diagnostics import (network_diagnostics, loss_group, remove_boundary_nodes,
    solve_given_conductances, openpnm_crosscheck, parse_native_network, native_conductances)
from scripts.run_throat_discovery import checked_protocol
from scripts.run_experiments import write_json, write_csv

OLD = ROOT / 'experiments/topoflow_discovery'
OUT = ROOT / 'experiments/topoflow_diagnostics'
DATA = ROOT / 'data/throat_discovery'
FEATURES = ['geodesic_tortuosity_lower_bound', 'axial_alignment_conductance_weighted',
            'degree_cv', 'min_cut_per_eligible_throat']


def strata_tables(predictions, selection):
    frame = predictions.copy()
    frame['actual_group'] = frame.actual_loss.map(loss_group)
    reference = selection[selection.method == 'guided_edge_gat']
    masks = {'all': np.ones(len(frame), dtype=bool)}
    for group in ['low_under_0.1pp', 'moderate_0.1_to_1pp', 'high_at_least_1pp']:
        masks['actual_' + group] = frame.actual_group == group
    masks['actual_at_least_5pp'] = frame.actual_loss >= .05
    for group in ['top', 'low', 'random']:
        keys = set(zip(reference[reference.selection_group == group].sample_id,
                       reference[reference.selection_group == group].throat_index))
        masks['gat_selected_' + group] = np.asarray([(sid, edge) in keys for sid, edge in zip(frame.sample_id, frame.throat_index)])
    graph_rows = []
    for group, mask in masks.items():
        for (method, scan, sample), part in frame.groupby(['method', 'scan_id', 'sample_id']):
            selected = part.loc[mask[part.index]]
            error = selected.predicted_loss - selected.actual_loss
            graph_rows.append({'group': group, 'method': method, 'scan_id': scan, 'sample_id': sample,
                'n_interventions': len(selected), 'n_throats': selected.throat_index.nunique(),
                'mae_pp': float(error.abs().mean() * 100) if len(error) else None,
                'bias_pp': float(error.mean() * 100) if len(error) else None,
                'mse_pp_squared': float((error ** 2).mean() * 10000) if len(error) else None,
                'mean_true_pp': float(selected.actual_loss.mean() * 100) if len(error) else None,
                'mean_predicted_pp': float(selected.predicted_loss.mean() * 100) if len(error) else None})
    graphs = pd.DataFrame(graph_rows)
    summaries = []
    for (group, method), part in graphs.groupby(['group', 'method']):
        present = part[part.n_interventions > 0]
        by_scan = present.groupby('scan_id')[['mae_pp', 'bias_pp', 'mse_pp_squared', 'mean_true_pp', 'mean_predicted_pp']].mean()
        summaries.append({'group': group, 'method': method, 'scan_groups_present': len(by_scan),
            'graphs_present': len(present), 'interventions': int(part.n_interventions.sum()),
            'mae_pp': float(by_scan.mae_pp.mean()) if len(by_scan) else None,
            'bias_pp': float(by_scan.bias_pp.mean()) if len(by_scan) else None,
            'rmse_pp': float(np.sqrt(by_scan.mse_pp_squared.mean())) if len(by_scan) else None,
            'mean_true_pp': float(by_scan.mean_true_pp.mean()) if len(by_scan) else None,
            'mean_predicted_pp': float(by_scan.mean_predicted_pp.mean()) if len(by_scan) else None})
    return graphs, pd.DataFrame(summaries)


def run(resume=False):
    checked_protocol()
    OUT.mkdir(parents=True, exist_ok=resume)
    if (OUT / 'completion.json').exists():
        raise ValueError('Completed analyses are not overwritten')
    seal = json.loads((OLD / 'selection_seal.json').read_text())
    for path, digest in seal['files'].items():
        if sha256(OLD / path) != digest:
            raise ValueError('Prior sealed evidence changed')
    plan = {
        'recorded_utc': datetime.now(timezone.utc).isoformat(), 'status': 'posthoc_analysis_of_existing_test_set',
        'prior_results': '4c48050; all five scan-level winner labels were known before this plan',
        'debug_peek': 'S2 crop 00 geometry/flow diagnostics and identical-g OpenPNM crosscheck exercised before this plan; no fitting',
        'new_independent_validation': False, 'training_or_calibration_fitted': False,
        'primary_association_features': FEATURES,
        'candidate_hypothesis': 'GAT minus affine top-throat loss increases with path tortuosity/degree heterogeneity and decreases with axial alignment/min-cut density',
        'association_scope': 'Four fixed candidate associations across five scan means; Spearman coefficients only, no fitted selector or causal claims',
        'outcome_strata_fractional_loss': {'low': '[0,0.001)', 'moderate': '[0.001,0.01)', 'high': '[0.01,1]', 'severe_subset': '[0.05,1]'},
        'selection_strata': 'Top, low and random groups from frozen guided-GAT selections; all methods evaluated on the same groups',
        'aggregation': 'Mean inside graph, mean available graphs in scan, equal-weight available scans; missing groups explicitly counted',
        'physics_audit': ['identical-g independent OpenPNM assembly on ten existing SNOW crop networks',
                          'fixed one-factor diameter/length/boundary alternatives on those crop networks',
                          'original archive S1/S5 maximal-ball networks with fixed circular and shape-factor closures'],
        'no_fit_to_published_reference': True,
        'full_snow_reextraction': 'Not scheduled: only 0.8 GiB available at setup, earlier full SNOW jobs peaked at 2.4 GB; use committed full-volume baselines and supplied original networks',
        'source_hashes': {p: sha256(ROOT / p) for p in ['src/research_diagnostics.py', 'scripts/analyze_discovery_review.py']},
        'previous_protocol_sha256': sha256(OLD / 'protocol.json'), 'previous_selection_seal_sha256': sha256(OLD / 'selection_seal.json')}
    if resume:
        previous = json.loads((OUT / 'analysis_plan.json').read_text())
        previous['resume_source_hashes'] = plan['source_hashes']
        previous['resume_reason'] = 'OpenPNM 3.6.0 Valvatne-Blunt expects an Nx3 conduit-length array; replaced dictionary-valued namespace with that array. No scientific settings or fitted models changed.'
        plan = previous
    write_json(OUT / 'analysis_plan.json', plan)
    prediction = pd.read_csv(OLD / 'common_pool_predictions.csv')
    selection = pd.read_csv(OLD / 'frozen_selections.csv')
    graphs, summary = strata_tables(prediction, selection)
    graphs.to_csv(OUT / 'response_per_graph_strata.csv', index=False)
    summary.to_csv(OUT / 'response_strata.csv', index=False)
    unique = prediction[prediction.method == 'no_response'].copy()
    unique['group'] = unique.actual_loss.map(loss_group)
    unique.groupby('group').agg(interventions=('actual_loss', 'size'), scans=('scan_id', 'nunique'),
                               mean_loss=('actual_loss', 'mean')).to_csv(OUT / 'outcome_distribution.csv')
    inventory = json.loads((OLD / 'inventory.json').read_text())
    diagnostic, crosschecks, alternatives = [], [], []
    for case in inventory:
        with np.load(DATA / f"{case['sample_id']}.npz") as package:
            net = {key: package[key] for key in package.files}
        prefix = {'sample_id': case['sample_id'], 'scan_id': case['scan_id']}
        diagnostics = network_diagnostics(net, case['shape_zyx'], case['voxel_size_m'])
        diagnostic.append({**prefix, **diagnostics})
        diameter = net['throat.inscribed_diameter'].astype(float)
        length = net['throat.direct_length'].astype(float)
        base_g = np.pi * diameter ** 4 / (128 * .001 * length)
        base, _ = solve_given_conductances(net, base_g, case['shape_zyx'], case['voxel_size_m'])
        independent = openpnm_crosscheck(net, base_g, case['shape_zyx'], case['voxel_size_m'])
        relative_error = abs(independent / base - 1)
        if relative_error > 1e-8:
            raise ValueError('Independent assembly mismatch')
        crosschecks.append({**prefix, 'pnm_k_m2': base, 'openpnm_k_m2': independent, 'relative_difference': relative_error})
        contracted = remove_boundary_nodes(net)
        variants = [('inscribed_direct', net, base_g),
            ('inscribed_total_length', net, np.pi * diameter ** 4 / (128 * .001 * net['throat.total_length'])),
            ('equivalent_diameter_direct', net, np.pi * net['throat.equivalent_diameter'].astype(float) ** 4 / (128 * .001 * length)),
            ('inscribed_direct_reservoir_contracted', contracted,
             np.pi * contracted['throat.inscribed_diameter'].astype(float) ** 4 / (128 * .001 * contracted['throat.direct_length']))]
        for name, candidate, g in variants:
            value, _ = solve_given_conductances(candidate, g, case['shape_zyx'], case['voxel_size_m'])
            alternatives.append({**prefix, 'variant': name, 'k_m2': value, 'ratio_to_original_crop_pnm': value / base,
                                 'scope': 'Closure/boundary sensitivity on crop, not full-volume validation'})
        print('Diagnosed', case['sample_id'], flush=True)
    pd.DataFrame(diagnostic).to_csv(OUT / 'network_diagnostics_per_graph.csv', index=False)
    write_csv(OUT / 'independent_solver_checks.csv', crosschecks)
    write_csv(OUT / 'crop_physics_variants.csv', alternatives)
    scan = pd.DataFrame(diagnostic).groupby('scan_id').mean(numeric_only=True)
    prior = pd.read_csv(OLD / 'per_scan_discovery.csv').set_index('scan_id')
    scan['gat_minus_affine_loss_pp'] = (prior.guided_edge_gat - prior.affine_pressure) * 100
    scan['gat_top_loss_pp'] = prior.guided_edge_gat * 100
    scan['affine_top_loss_pp'] = prior.affine_pressure * 100
    scan.to_csv(OUT / 'network_diagnostics_per_scan.csv')
    association = []
    for feature in FEATURES:
        rho = spearmanr(scan[feature], scan.gat_minus_affine_loss_pp).statistic
        association.append({'feature': feature, 'scan_groups': len(scan), 'spearman_with_gat_advantage': float(rho),
                            'scope': 'Post-hoc descriptive association, no independent validation or learned selector'})
    write_csv(OUT / 'exploratory_associations.csv', association)
    references = json.loads((ROOT / 'experiments/topoflow_followup/external_data/source_references.json').read_text())
    native_rows, native_meta = [], []
    with zipfile.ZipFile(ROOT / 'data/public/ImagesICPSC2007.zip') as archive:
        for name in ['S1', 'S5']:
            member = f'ImagesICPSC2007/cnm_2007/{name}_cnm2009.tar.gz'
            blob = archive.read(member)
            with tarfile.open(fileobj=io.BytesIO(blob), mode='r:gz') as package:
                files = {key: package.extractfile(next(m for m in package.getmembers() if m.name.endswith(f'{name}_{key}.dat'))).read()
                         for key in ['node1', 'node2', 'link1', 'link2']}
            net, size, meta = parse_native_network(files)
            reference = next(r for r in references if r['parent'] == name)
            voxel = reference['voxel_size_m']
            np.testing.assert_allclose(size, np.asarray(reference['source_shape_zyx']) * voxel, rtol=1e-6)
            delta = net['pore.coords'][net['pore.xmax']].mean(0) - net['pore.coords'][net['pore.xmin']].mean(0)
            axis = int(np.argmax(abs(delta)))
            native_meta.append({'sample': name, **meta, 'domain_size_m': size.tolist(),
                'boundary_group_mean_displacement_m': delta.tolist(), 'dominant_native_coordinate_axis': axis,
                'archive_member_sha256': hashlib.sha256(blob).hexdigest(),
                'source_file_sha256': {k: hashlib.sha256(v).hexdigest() for k, v in files.items()},
                'boundary_convention': 'Remove reservoir throats, apply fixed pressures to adjacent physical pores; legacy exact convention unverified'})
            for variant, g in native_conductances(net).items():
                value, diag = solve_given_conductances(net, g, reference['source_shape_zyx'], voxel)
                independent = openpnm_crosscheck(net, g, reference['source_shape_zyx'], voxel)
                if abs(value / independent - 1) > 1e-8:
                    raise ValueError('Original network independent solve failed')
                native_rows.append({'sample': name, 'network': 'original_maximal_ball', 'variant': variant,
                    'k_mD': value / 9.869233e-16, 'published_network_k_mD': reference['published_network_k_mD'],
                    'ratio_to_published_network_k': value / 9.869233e-16 / reference['published_network_k_mD'],
                    'published_image_kx_mD': reference['published_full_volume_directional_k_mD']['x'],
                    'published_image_ky_mD': reference['published_full_volume_directional_k_mD']['y'],
                    'published_image_kz_mD': reference['published_full_volume_directional_k_mD']['z'],
                    'openpnm_relative_difference': abs(value / independent - 1),
                    'mass_balance': diag['relative_mass_imbalance']})
            print('Checked original maximal-ball reference network', name, flush=True)
    write_csv(OUT / 'original_network_reference.csv', native_rows)
    write_json(OUT / 'original_network_provenance.json', native_meta)
    write_json(OUT / 'completion.json', {'completed_utc': datetime.now(timezone.utc).isoformat(),
        'graphs': len(inventory), 'scan_groups': len(scan), 'independent_crop_assembly_checks': len(crosschecks),
        'original_network_closure_checks': len(native_rows), 'no_model_weights_modified': True,
        'original_selection_seal_sha256': sha256(OLD / 'selection_seal.json')})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(resume=args.resume)
