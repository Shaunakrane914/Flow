"""Verify saved evidence, temporal seals and headline metrics without model fitting."""
from collections import defaultdict
from datetime import datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_throat_discovery import OUT, checked_protocol, read_csv, read_source
from scripts.run_experiments import write_json
from src.experiment_data import sha256


def macro(rows, value):
    graphs = defaultdict(list)
    for row in rows:
        graphs[row['scan_id'], row['sample_id']].append(value(row))
    scans = defaultdict(list)
    for (scan, sample), values in graphs.items():
        scans[scan].append(float(np.mean(values)))
    return float(np.mean([np.mean(values) for values in scans.values()]))


def main():
    protocol = checked_protocol()
    seal = json.loads((OUT / 'selection_seal.json').read_text())
    start = json.loads((OUT / 'evaluation_started.json').read_text())
    complete = json.loads((OUT / 'evaluation_complete.json').read_text())
    assert sha256(OUT / 'protocol.json') == seal['protocol_sha256']
    assert sha256(OUT / 'selection_seal.json') == start['selection_seal_sha256'] == complete['selection_seal_sha256']
    for path, digest in seal['files'].items():
        assert sha256(OUT / path) == digest, path
    times = [protocol['locked_utc'], seal['sealed_utc'], start['started_utc'], complete['completed_utc']]
    assert [datetime.fromisoformat(t) for t in times] == sorted(datetime.fromisoformat(t) for t in times)
    assert not set(protocol['validation_scan_ids']) & set(protocol['prior_external_scan_ids'])
    outcomes = read_csv(OUT / 'outcomes.csv')
    predictions = read_csv(OUT / 'common_pool_predictions.csv')
    selected = read_csv(OUT / 'frozen_selections.csv')
    discovery = read_csv(OUT / 'discovery.csv')
    outcomes_by_key = {(r['sample_id'], r['throat_index'], r['diameter_reduction']): float(r['actual_loss']) for r in outcomes}
    assert len(outcomes_by_key) == len(outcomes) == complete['fresh_intervention_solves']
    identity_error = max(abs(float(r['actual_loss']) - float(r['exact_update_loss'])) for r in outcomes)
    assert identity_error < 1e-6
    sets = defaultdict(set)
    for row in predictions:
        key = row['sample_id'], row['throat_index'], row['diameter_reduction']
        assert float(row['actual_loss']) == outcomes_by_key[key]
        sets[row['sample_id'], row['method']].add(key[1:])
    for sample in {r['sample_id'] for r in predictions}:
        pools = [v for (sid, method), v in sets.items() if sid == sample]
        assert all(v == pools[0] for v in pools)
        scores = read_csv(OUT / 'scores' / f'{sample}.csv')
        lookup = {(r['method'], r['throat_index']): r for r in scores}
        for row in [r for r in predictions if r['sample_id'] == sample and r['method'] != 'no_response']:
            assert float(row['predicted_loss']) == float(lookup[row['method'], row['throat_index']][f"prediction_{row['diameter_reduction']}"])
        selections = [r for r in selected if r['sample_id'] == sample]
        random = [{r['throat_index'] for r in selections if r['method'] == m and r['selection_group'] == 'random'}
                  for m in {r['method'] for r in selections}]
        assert all(v == random[0] for v in random) and len(random[0]) == 3
    recomputed = {}
    for metric in read_csv(OUT / 'discovery_metrics.csv'):
        method = metric['method']
        if method == 'random':
            rows = [r for r in discovery if r['method'] == 'guided_edge_gat' and r['selection_group'] == 'random']
        else:
            rows = [r for r in discovery if r['method'] == method and r['selection_group'] == 'top']
        value = macro([r for r in rows if float(r['diameter_reduction']) == .5], lambda r: float(r['actual_loss'])) * 100
        assert abs(value - float(metric['mean_actual_loss_pp'])) < 1e-10
        recomputed[method] = value
    for metric in read_csv(OUT / 'accuracy_metrics.csv'):
        rows = [r for r in predictions if r['method'] == metric['method']]
        value = macro(rows, lambda r: abs(float(r['predicted_loss']) - float(r['actual_loss']))) * 100
        assert abs(value - float(metric['common_pool_mae_pp'])) < 1e-10
    # Verify phase interpretation and exact source-crop identities separately
    # from model performance; this audit cannot influence the frozen selections.
    prior_hashes = set()
    prior_paths = list((ROOT / 'data/locked_external').glob('*.npy'))
    manifest = ROOT / 'data/benchmark_dataset/manifest.json'
    if manifest.exists():
        old = json.loads(manifest.read_text())['samples']
        prior_paths.extend(ROOT / 'data/benchmark_dataset/chunks' / f"{r['id']}.npy"
                           for r in old if r['formation'] != 'Synthetic')
    for path in prior_paths:
        prior_hashes.add(hashlib.sha256(np.load(path).astype(np.uint8).tobytes()).hexdigest())
    audit, new_hashes = [], set()
    attempts = read_csv(OUT / 'crop_attempts.csv')
    with zipfile.ZipFile(ROOT / 'data/public/ImagesICPSC2007.zip') as package:
        for name in protocol['validation_scan_ids']:
            header, volume, voxel = read_source(package, name)
            actual_phi = float((volume == 0).mean())
            header_phi = float(re.search(r'Porosity\(%\)\s*([\d.]+)', header).group(1)) / 100
            assert abs(actual_phi - header_phi) < .001
            crops = []
            for row in [r for r in attempts if r['scan_id'] == name]:
                origin = tuple(map(int, row['origin_zyx'].split(',')))
                image = volume[tuple(slice(v, v + 128) for v in origin)] == 0
                digest = hashlib.sha256(image.astype(np.uint8).tobytes()).hexdigest()
                assert digest not in prior_hashes and digest not in new_hashes
                new_hashes.add(digest)
                assert abs(float(image.mean()) - float(row['porosity'])) < 1e-12
                crops.append({'sample_id': row['sample_id'], 'pore_uint8_sha256': digest})
            audit.append({'scan_id': name, 'phase_rule': '0 is pore', 'voxel_size_m': voxel,
                          'whole_volume_porosity': actual_phi, 'header_porosity': header_phi,
                          'source_raw_uint8_sha256': hashlib.sha256(volume.tobytes()).hexdigest(), 'crops': crops})
    write_json(OUT / 'source_byte_audit.json', {'prior_crops_checked': len(prior_hashes),
               'new_crops_checked': len(new_hashes), 'exact_crop_duplicates': 0, 'sources': audit,
               'scope': 'Exact voxel duplicates ruled out for locally available prior crops; this does not certify geological independence'})
    test_run = json.loads((OUT / 'test_run.json').read_text())
    test_output = (OUT / 'test_output.txt').read_text(encoding='utf-8')
    assert test_run['exit_code'] == 0 and re.search(r'\nOK\s*\n', test_output)
    test_count = int(re.search(r'Ran (\d+) tests', test_output).group(1))
    write_json(OUT / 'verification.json', {'status': 'passed', 'tests_passed': test_count,
        'test_output_sha256': sha256(OUT / 'test_output.txt'),
        'protocol_sources_checkpoints_and_selection_hashes_match': True, 'timestamps_ordered': times,
        'selection_before_flow_evaluation': True, 'new_scan_ids_disjoint_from_prior_external': True,
        'graphs': complete['graphs'], 'scan_groups': complete['scans'], 'fresh_intervention_solves': len(outcomes),
        'prediction_rows': len(predictions), 'identical_intervention_pool_for_every_predictor': True,
        'predictions_match_sealed_geometry_only_outputs': True,
        'fresh_vs_exact_update_max_fractional_error': identity_error,
        'independently_recomputed_discovery_pp': recomputed,
        'headline_tables_recomputed_with_plain_csv_and_scan_macro_loops': True,
        'package_versions': {p: importlib.metadata.version(p) for p in ['numpy', 'scipy', 'torch', 'torch-geometric', 'porespy', 'pandas', 'matplotlib']},
        'limitations': ['five source-scan groups, not independently certified geological specimens',
                        'no new physically validated reference', 'no GAT accuracy or runtime superiority']})
    print('Verified seals, common-pool fairness, frozen predictions, fresh solves and every headline metric.', flush=True)


if __name__ == '__main__':
    main()
