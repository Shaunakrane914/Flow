"""Prospective, geometry-only critical-throat discovery on five unused scan IDs.

Run freeze, select and evaluate in separate commands, in that order. Selection
files are sealed before importing the outcome solver. Existing trained models
are never retrained or selected using these five scans.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import zipfile

import numpy as np
import torch
from threadpoolctl import threadpool_limits
from torch_geometric.data import Batch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.critical_throat_model import CriticalThroatRegressor, fractional_response
from src.experiment_data import sha256
from src.throat_discovery import geometry_graph, ranked_selection, analytic_scores, response_array
from scripts.prepare_locked_external import read_source
from scripts.run_experiments import write_csv, write_json

OUT = ROOT / 'experiments/topoflow_discovery'
DATA = ROOT / 'data/throat_discovery'
PILOT = ROOT / 'experiments/topoflow_followup/throat_aware_pilot/training'
MODELS = ['guided_edge_gat', 'guided_edge_sage', 'guided_edge_mlp', 'edge_gat']
SEEDS = [11, 22, 33, 44, 55]
DOSES = [.1, .25, .5]
SOURCES = ['scripts/run_throat_discovery.py', 'src/throat_discovery.py',
           'src/critical_throat_model.py', 'src/research_pnm.py',
           'scripts/prepare_locked_external.py']


def now():
    return datetime.now(timezone.utc).isoformat()


def read_csv(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def freeze():
    OUT.mkdir(parents=True, exist_ok=False)
    DATA.mkdir(parents=True, exist_ok=False)
    archive = ROOT / 'data/public/ImagesICPSC2007.zip'
    if hashlib.sha1(archive.read_bytes()).hexdigest() != 'd07c1d3403a19d4b39492d9671805cfd3b65c2f5':
        raise ValueError('Official archive checksum differs')
    previous = json.loads((PILOT / 'protocol.json').read_text())
    checkpoints = {}
    for model in MODELS:
        for seed in SEEDS:
            path = PILOT / 'checkpoints' / f'{model}_seed{seed}.pth'
            checkpoints[str(path.relative_to(ROOT))] = sha256(path)
    protocol = {
        'status': 'frozen_before_new_scan_extraction', 'locked_utc': now(),
        'validation_scan_ids': ['S2', 'S4', 'S6', 'S7', 'S8'],
        'prior_external_scan_ids': ['S1', 'S3', 'S5', 'S9'],
        'sampling': 'Two fixed nonoverlapping 128-cube opposite corners (000 and maximum origin); no fallback crops',
        'independence': 'Five named source scans, not independently certified geological specimens; crops and interventions cluster within scan',
        'primary_model': 'guided_edge_gat', 'models': MODELS, 'seeds': SEEDS,
        'deployment': 'Arithmetic mean of five fixed seed predictions; no best-seed selection',
        'training': 'Reuse frozen pilot checkpoints trained on eight crops from four development scans; no new fitting',
        'matched_training': {k: previous[k] for k in ['learning_rate', 'weight_decay', 'epochs_max', 'patience', 'guidance_weight', 'epoch_selection']},
        'locked_epochs': json.loads((PILOT / 'locked_epochs.json').read_text()),
        'historical_test_reuse': 'Guided mean/MLP were post-hoc on the old four scans, but all four models are fixed before this new five-scan validation',
        'eligible_throats': 'Both endpoints physical and in a topology-spanning inlet-outlet component; no pressure or flow criteria',
        'selection': 'Each model ranks its own global importance head; three highest and three lowest, stable edge-index ties; three shared uniform random edges per graph',
        'random_seed': 20261023, 'restrictions': DOSES,
        'controls': ['conductance d^4/L', 'affine-pressure energy g*(delta x)^2', 'shared random', 'solver-based sensitivity oracle'],
        'primary_endpoint': 'Scan-macro mean actual fractional loss at 50% narrowing of three individually restricted model-selected top throats',
        'primary_comparisons': ['guided_edge_gat vs affine_pressure', 'guided_edge_gat vs guided_edge_mlp'],
        'response_accuracy': 'Each method evaluated on the same union of all geometry-selected top/low/random throats at all three doses; oracle selections excluded from this shared pool',
        'secondary': ['top versus low and random', 'all-eligible sensitivity Spearman', 'top-three sensitivity overlap', 'seed variation', 'warm CPU latency'],
        'statistics': 'Average crops inside each scan, then average five scans equally; exact paired scan sign-flip tests (minimum two-sided p=0.0625); cluster bootstrap descriptive intervals',
        'timing': {'threads': 1, 'warmups': 5, 'repeats': 30, 'exact_precompute_repeats': 5,
                   'include': 'All-throat neural forward + response readout + ensemble aggregation; separate SNOW/features, PNM base, three fresh solves, cached exact readout and exact precompute',
                   'exclude': 'Disk/model load, PyG batching and extraction from forward-only timing; reported separately'},
        'physics_scope': 'Cylindrical inscribed-diameter, direct-length Hagen-Poiseuille PNM only; unresolved S1/S5 physical-reference discrepancy is not repaired here',
        'source_url': 'https://imperialcollegelondon.box.com/v/ImagesICPSC2007',
        'archive_sha256': sha256(archive), 'checkpoints': checkpoints,
        'source_hashes': {p: sha256(ROOT / p) for p in SOURCES}}
    write_json(OUT / 'protocol.json', protocol)
    print('Protocol locked before new scans were extracted or evaluated.', flush=True)


def checked_protocol():
    protocol = json.loads((OUT / 'protocol.json').read_text())
    for path, digest in {**protocol['source_hashes'], **protocol['checkpoints']}.items():
        if sha256(ROOT / path) != digest:
            raise ValueError(f'Frozen source or checkpoint changed: {path}')
    return protocol


def load_models():
    models = {}
    for name in MODELS:
        models[name] = []
        for seed in SEEDS:
            ck = torch.load(PILOT / 'checkpoints' / f'{name}_seed{seed}.pth', map_location='cpu', weights_only=False)
            model = CriticalThroatRegressor(name.removeprefix('guided_'))
            model.load_state_dict(ck['state_dict'])
            models[name].append(model.eval())
    return models


@torch.inference_mode()
def ensemble(models, batch):
    predictions = [model(batch) for model in models]
    importance = torch.stack([p[0] for p in predictions]).mean(0).numpy()
    # Average responses rather than applying nonlinear readout to average heads.
    response = torch.stack([fractional_response(*p) for p in predictions]).mean(0).numpy()
    return importance, response


def select():
    protocol = checked_protocol()
    if (OUT / 'selection_seal.json').exists() or (OUT / 'crop_attempts.csv').exists():
        raise ValueError('Refusing to overwrite selection attempts')
    import porespy as ps
    models = load_models()
    attempts, inventory, selections = [], [], []
    archive = ROOT / 'data/public/ImagesICPSC2007.zip'
    if sha256(archive) != protocol['archive_sha256']:
        raise ValueError('Source archive changed')
    (OUT / 'scores').mkdir()
    rng = np.random.default_rng(protocol['random_seed'])
    with zipfile.ZipFile(archive) as package:
        for name in protocol['validation_scan_ids']:
            header, volume, voxel = read_source(package, name)
            (OUT / f'{name}_source_header.txt').write_text(header, encoding='utf-8')
            for index, origin in enumerate([(0, 0, 0), tuple(d - 128 for d in volume.shape)]):
                sid = f'discovery_{name}_crop_{index:02d}'
                image = volume[tuple(slice(v, v + 128) for v in origin)] == 0
                row = {'sample_id': sid, 'scan_id': name, 'origin_zyx': ','.join(map(str, origin)),
                       'source_shape_zyx': ','.join(map(str, volume.shape)), 'voxel_size_m': voxel,
                       'porosity': float(image.mean()), 'status': 'pending', 'reason': ''}
                try:
                    started = time.perf_counter_ns()
                    net = ps.networks.snow2(image, voxel_size=voxel, boundary_width=[3, 0, 0],
                                           parallel_kw={'divs': 1}, sigma=.4, r_max=4).network
                    snow_ms = (time.perf_counter_ns() - started) / 1e6
                    started = time.perf_counter_ns()
                    graph, eligible = geometry_graph(net, image.shape, voxel)
                    if len(eligible) < 9:
                        raise ValueError('Fewer than nine geometry-eligible throats')
                    feature_ms = (time.perf_counter_ns() - started) / 1e6
                except ValueError as exc:
                    row.update(status='excluded_geometry', reason=str(exc))
                    attempts.append(row)
                    write_csv(OUT / 'crop_attempts.csv', attempts)
                    print(sid, row['status'], row['reason'], flush=True)
                    continue
                path = DATA / f'{sid}.npz'
                np.savez_compressed(path, **{key: np.asarray(value) for key, value in net.items()})
                torch.save(graph, DATA / f'{sid}.pt')
                batch = Batch.from_data_list([graph])
                shared_random = rng.choice(eligible, 3, replace=False)
                score_rows = []
                scores = analytic_scores(net)
                for method in MODELS + list(scores):
                    if method in models:
                        importance, response = ensemble(models[method], batch)
                    else:
                        importance = scores[method]
                        response = response_array(importance, (1 - importance) * .5)
                    for edge in range(len(importance)):
                        score_rows.append({'sample_id': sid, 'scan_id': name, 'method': method,
                                           'throat_index': edge, 'eligible': int(edge in set(eligible)),
                                           'importance': float(importance[edge]),
                                           **{f'prediction_{d}': float(response[edge, j]) for j, d in enumerate(DOSES)}})
                    groups = ranked_selection(importance, eligible)
                    groups['random'] = shared_random
                    for group, edges in groups.items():
                        for rank, edge in enumerate(edges, 1):
                            selections.append({'sample_id': sid, 'scan_id': name, 'method': method,
                                               'selection_group': group, 'rank': rank, 'throat_index': int(edge)})
                write_csv(OUT / 'scores' / f'{sid}.csv', score_rows)
                inventory.append({'sample_id': sid, 'scan_id': name, 'shape_zyx': list(image.shape),
                                  'voxel_size_m': voxel, 'eligible_throats': len(eligible),
                                  'nodes': graph.num_nodes, 'throats': graph.throat_index.shape[1],
                                  'snow2_ms': snow_ms, 'feature_ms': feature_ms,
                                  'network_sha256': sha256(path), 'graph_sha256': sha256(DATA / f'{sid}.pt')})
                row.update(status='accepted_geometry')
                attempts.append(row)
                write_json(OUT / 'inventory.json', inventory)
                write_csv(OUT / 'crop_attempts.csv', attempts)
                write_csv(OUT / 'frozen_selections.csv', selections)
                print(sid, 'selected without flow labels;', len(eligible), 'eligible throats', flush=True)
            del volume
    files = [OUT / 'inventory.json', OUT / 'crop_attempts.csv', OUT / 'frozen_selections.csv', *sorted((OUT / 'scores').glob('*.csv'))]
    write_json(OUT / 'selection_seal.json', {'sealed_utc': now(), 'protocol_sha256': sha256(OUT / 'protocol.json'),
               'outcome_solver_calls_before_seal': 0, 'files': {str(p.relative_to(OUT)): sha256(p) for p in files},
               'graphs': len(inventory), 'scan_ids': sorted({r['scan_id'] for r in inventory})})
    print('All selections sealed. Outcome evaluation can now begin.', flush=True)


def benchmark(function, warmups=5, repeats=30):
    for _ in range(warmups):
        function()
    values = []
    for repeat in range(repeats):
        started = time.perf_counter_ns()
        function()
        values.append((time.perf_counter_ns() - started) / 1e6)
    return values


def evaluate():
    protocol = checked_protocol()
    seal = json.loads((OUT / 'selection_seal.json').read_text())
    if sha256(OUT / 'protocol.json') != seal['protocol_sha256']:
        raise ValueError('Protocol changed after selection')
    for path, digest in seal['files'].items():
        if sha256(OUT / path) != digest:
            raise ValueError(f'Sealed selection changed: {path}')
    if (OUT / 'evaluation_started.json').exists():
        raise ValueError('Refusing to overwrite an outcome evaluation')
    write_json(OUT / 'evaluation_started.json', {'started_utc': now(), 'selection_seal_sha256': sha256(OUT / 'selection_seal.json')})
    # Only after the seal has been verified may flow-derived targets be obtained.
    from src.research_pnm import solve_permeability
    from src.throat_discovery import exact_response
    from scipy.stats import spearmanr
    import psutil
    models = load_models()
    inventory = json.loads((OUT / 'inventory.json').read_text())
    selected = read_csv(OUT / 'frozen_selections.csv')
    outcomes, predictions, discovery, rankings, timing, resources, checks = [], [], [], [], [], [], []
    for case in inventory:
        sid, scan = case['sample_id'], case['scan_id']
        if sha256(DATA / f'{sid}.npz') != case['network_sha256'] or sha256(DATA / f'{sid}.pt') != case['graph_sha256']:
            raise ValueError('Selected geometry changed')
        with np.load(DATA / f'{sid}.npz') as package:
            net = {key: package[key] for key in package.files}
        graph = torch.load(DATA / f'{sid}.pt', map_location='cpu', weights_only=False)
        batch = Batch.from_data_list([graph])
        _, eligible = geometry_graph(net, case['shape_zyx'], case['voxel_size_m'])
        base, sensitivity, leverage, truth, diag = exact_response(net, case['shape_zyx'], case['voxel_size_m'])
        score_rows = read_csv(OUT / 'scores' / f'{sid}.csv')
        this_selection = [r for r in selected if r['sample_id'] == sid]
        common_edges = sorted({int(r['throat_index']) for r in this_selection})
        oracle_edges = ranked_selection(sensitivity, eligible)['top']
        all_edges = sorted(set(common_edges) | set(oracle_edges.tolist()))
        max_error = 0.
        for edge in all_edges:
            for j, dose in enumerate(DOSES):
                changed = np.asarray(net['throat.inscribed_diameter'], dtype=float).copy()
                changed[edge] *= 1 - dose
                value, _ = solve_permeability(net['pore.coords'], net['throat.conns'], changed,
                    net['throat.direct_length'], net['pore.xmin'], net['pore.xmax'], case['shape_zyx'], case['voxel_size_m'])
                actual = 1 - value / base
                error = abs(actual - truth[edge, j])
                max_error = max(max_error, error)
                if error > 1e-6 or actual < -1e-7 or actual > 1 + 1e-7:
                    raise ValueError('Fresh solve and exact update disagree')
                outcomes.append({'sample_id': sid, 'scan_id': scan, 'throat_index': edge,
                                 'diameter_reduction': dose, 'actual_loss': actual,
                                 'exact_update_loss': float(truth[edge, j]), 'common_pool': int(edge in common_edges)})
        actual_lookup = {(r['throat_index'], r['diameter_reduction']): r['actual_loss'] for r in outcomes if r['sample_id'] == sid}
        for method in MODELS + ['conductance', 'affine_pressure']:
            rows = [r for r in score_rows if r['method'] == method]
            importance = np.asarray([float(r['importance']) for r in rows])
            predicted = np.asarray([[float(r[f'prediction_{d}']) for d in DOSES] for r in rows])
            rho = spearmanr(importance[eligible], sensitivity[eligible]).statistic
            top = ranked_selection(importance, eligible)['top']
            rankings.append({'sample_id': sid, 'scan_id': scan, 'method': method,
                             'spearman_sensitivity': float(rho) if np.isfinite(rho) else None,
                             'top3_sensitivity_overlap': len(set(top) & set(oracle_edges)) / 3})
            for edge in common_edges:
                for j, dose in enumerate(DOSES):
                    predictions.append({'sample_id': sid, 'scan_id': scan, 'method': method,
                                        'throat_index': edge, 'diameter_reduction': dose,
                                        'actual_loss': actual_lookup[edge, dose], 'predicted_loss': float(predicted[edge, j])})
            for row in [r for r in this_selection if r['method'] == method]:
                edge = int(row['throat_index'])
                for j, dose in enumerate(DOSES):
                    discovery.append({**row, 'diameter_reduction': dose, 'actual_loss': actual_lookup[edge, dose],
                                      'predicted_loss': float(predicted[edge, j])})
        for edge in common_edges:
            for dose in DOSES:
                predictions.append({'sample_id': sid, 'scan_id': scan, 'method': 'no_response',
                                    'throat_index': edge, 'diameter_reduction': dose,
                                    'actual_loss': actual_lookup[edge, dose], 'predicted_loss': 0.})
        for rank, edge in enumerate(oracle_edges, 1):
            for j, dose in enumerate(DOSES):
                discovery.append({'sample_id': sid, 'scan_id': scan, 'method': 'pnm_sensitivity_oracle',
                                  'selection_group': 'top', 'rank': rank, 'throat_index': int(edge),
                                  'diameter_reduction': dose, 'actual_loss': actual_lookup[int(edge), dose],
                                  'predicted_loss': float(truth[edge, j])})
        def add_timing(method, scope, function, repeats=30):
            values = benchmark(function, repeats=repeats)
            timing.extend({'sample_id': sid, 'scan_id': scan, 'method': method, 'scope': scope,
                           'repeat': i, 'ms': value} for i, value in enumerate(values))
        for method in MODELS:
            add_timing(method, 'five_seed_ensemble_all_throats_three_doses', lambda: ensemble(models[method], batch))
            add_timing(method, 'single_seed_all_throats_three_doses', lambda: ensemble(models[method][:1], batch))
        args = (net['pore.coords'], net['throat.conns'], net['throat.inscribed_diameter'],
                net['throat.direct_length'], net['pore.xmin'], net['pore.xmax'], case['shape_zyx'], case['voxel_size_m'])
        add_timing('pnm', 'base_solve', lambda: solve_permeability(*args))
        chosen = [int(r['throat_index']) for r in this_selection
                  if r['method'] == 'guided_edge_gat' and r['selection_group'] == 'top']
        def fresh_three():
            values = []
            for edge in chosen:
                changed = np.asarray(args[2], dtype=float).copy()
                changed[edge] *= .5
                values.append(solve_permeability(args[0], args[1], changed, *args[3:])[0])
            return values
        add_timing('pnm', 'three_fresh_50pct_solves_base_excluded', fresh_three)
        add_timing('pnm_exact', 'base_and_all_throat_leverage_precompute',
                   lambda: exact_response(net, case['shape_zyx'], case['voxel_size_m']), repeats=5)
        add_timing('pnm_exact', 'cached_all_throats_three_doses', lambda: response_array(sensitivity, leverage))
        add_timing('affine_pressure', 'all_throat_scores', lambda: analytic_scores(net))
        resources.append({'sample_id': sid, 'scan_id': scan,
            'process_rss_after_benchmarks_bytes': psutil.Process().memory_info().rss,
            'graph_tensor_bytes': sum(v.numel() * v.element_size() for _, v in graph if torch.is_tensor(v)),
            'parameters_bytes_per_seed': {m: sum(p.numel() * p.element_size() for p in models[m][0].parameters()) for m in MODELS},
            'rss_scope': 'Observed resident set after calls, includes libraries/models/data; not peak or incremental model memory'})
        checks.append({'sample_id': sid, 'base_k_m2': base, 'fresh_intervention_solves': len(all_edges) * len(DOSES),
                       'common_pool_edges': len(common_edges), 'finite_update_max_error': max_error,
                       'relative_mass_imbalance': diag['relative_mass_imbalance']})
        for name, rows in [('outcomes', outcomes), ('common_pool_predictions', predictions), ('discovery', discovery),
                           ('rankings', rankings), ('timing_observations', timing)]:
            write_csv(OUT / f'{name}.csv', rows)
        write_json(OUT / 'resources.json', resources)
        write_json(OUT / 'physics_checks.json', checks)
        print(sid, 'outcomes verified;', len(common_edges), 'common-pool throats;',
              'max identity error', f'{max_error:.2g}', flush=True)
    write_json(OUT / 'evaluation_complete.json', {'completed_utc': now(), 'graphs': len(inventory),
        'scans': len({r['scan_id'] for r in inventory}), 'fresh_intervention_solves': sum(r['fresh_intervention_solves'] for r in checks),
        'selection_seal_sha256': sha256(OUT / 'selection_seal.json'),
        'environment': {'python': sys.version, 'platform': platform.platform(), 'torch': torch.__version__,
                        'cpu_threads': torch.get_num_threads(), 'cuda_used': False}})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['freeze', 'select', 'evaluate'])
    args = parser.parse_args()
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        {'freeze': freeze, 'select': select, 'evaluate': evaluate}[args.stage]()
