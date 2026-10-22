"""Independently reaggregate fixed predictions and verify diagnostic provenance."""
from collections import defaultdict
import csv
import importlib.metadata
import json
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.experiment_data import sha256
from scripts.run_throat_discovery import checked_protocol
from scripts.run_experiments import write_json

OUT = ROOT / 'experiments/topoflow_diagnostics'
OLD = ROOT / 'experiments/topoflow_discovery'


def read(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def applies(row, group, selections):
    loss = float(row['actual_loss'])
    if group == 'all':
        return True
    if group == 'actual_low_under_0.1pp':
        return loss < .001
    if group == 'actual_moderate_0.1_to_1pp':
        return .001 <= loss < .01
    if group == 'actual_high_at_least_1pp':
        return loss >= .01
    if group == 'actual_at_least_5pp':
        return loss >= .05
    return (row['sample_id'], row['throat_index']) in selections[group.removeprefix('gat_selected_')]


def main():
    original_protocol = checked_protocol()
    assert sha256(ROOT / 'data/public/ImagesICPSC2007.zip') == original_protocol['archive_sha256']
    plan = json.loads((OUT / 'analysis_plan.json').read_text())
    assert plan['status'] == 'posthoc_analysis_of_existing_test_set'
    assert not plan['new_independent_validation'] and not plan['training_or_calibration_fitted']
    for name, digest in plan.get('resume_source_hashes', plan['source_hashes']).items():
        assert sha256(ROOT / name) == digest, name
    assert sha256(OLD / 'protocol.json') == plan['previous_protocol_sha256']
    assert sha256(OLD / 'selection_seal.json') == plan['previous_selection_seal_sha256']
    seal = json.loads((OLD / 'selection_seal.json').read_text())
    for name, digest in seal['files'].items():
        assert sha256(OLD / name) == digest, name
    predictions = read(OLD / 'common_pool_predictions.csv')
    selections = defaultdict(set)
    for row in read(OLD / 'frozen_selections.csv'):
        if row['method'] == 'guided_edge_gat':
            selections[row['selection_group']].add((row['sample_id'], row['throat_index']))
    metrics = read(OUT / 'response_strata.csv')
    for metric in metrics:
        graphs = defaultdict(list)
        selected_rows = [r for r in predictions if r['method'] == metric['method'] and applies(r, metric['group'], selections)]
        for row in selected_rows:
            graphs[row['scan_id'], row['sample_id']].append(float(row['predicted_loss']) - float(row['actual_loss']))
        scans = defaultdict(list)
        for (scan, sample), errors in graphs.items():
            scans[scan].append([np.abs(errors).mean() * 100, np.mean(errors) * 100,
                                np.mean(np.asarray(errors) ** 2) * 10000])
        aggregate = np.mean([np.mean(values, axis=0) for values in scans.values()], axis=0)
        np.testing.assert_allclose(aggregate, [float(metric['mae_pp']), float(metric['bias_pp']), float(metric['rmse_pp']) ** 2], rtol=1e-10, atol=1e-10)
        assert len(selected_rows) == int(metric['interventions'])
        assert len(scans) == int(metric['scan_groups_present'])
        assert len(graphs) == int(metric['graphs_present'])
    checks = read(OUT / 'independent_solver_checks.csv')
    native = read(OUT / 'original_network_reference.csv')
    assert len(checks) == 10 and len(native) == 6
    max_relative = max([float(r['relative_difference']) for r in checks] + [float(r['openpnm_relative_difference']) for r in native])
    assert max_relative < 1e-8
    for row in native:
        assert float(row['mass_balance']) < 1e-7
        np.testing.assert_allclose(float(row['k_mD']) / float(row['published_network_k_mD']), float(row['ratio_to_published_network_k']), rtol=1e-12)
    test = json.loads((OUT / 'test_run.json').read_text())
    output = (OUT / 'test_output.txt').read_text(encoding='utf-8')
    assert test['exit_code'] == 0 and re.search(r'\nOK\s*\n', output)
    test_count = int(re.search(r'Ran (\d+) tests', output).group(1))
    write_json(OUT / 'verification.json', {'status': 'passed', 'tests_passed': test_count,
        'test_output_sha256': sha256(OUT / 'test_output.txt'),
        'previous_protocol_and_selection_seal_unchanged': True, 'frozen_checkpoint_hashes_unchanged': True,
        'posthoc_scope_disclosed': True, 'new_independent_test_scans': 0, 'trained_or_calibrated_models': 0,
        'response_stratum_rows_independently_recomputed': len(metrics),
        'identical_conductance_assembly_checks': len(checks) + len(native),
        'maximum_independent_assembly_relative_difference': max_relative,
        'original_network_comparisons_are_not_matched_voxel_flow_validation': True,
        'unresolved': ['SNOW/image full-reference mismatch', 'independent validation of topology association', 'GAT response calibration'],
        'package_versions': {p: importlib.metadata.version(p) for p in ['numpy', 'scipy', 'openpnm', 'networkx', 'pandas', 'torch', 'matplotlib']}})
    print(f'Verified {len(metrics)} response-stratum rows, sixteen assembly comparisons and unchanged frozen study.', flush=True)


if __name__ == '__main__':
    main()
