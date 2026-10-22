"""Nested parent-held-out evaluation; preserves historical data and outputs.

Every natural parent is tested once. Hyperparameters and training duration are
selected by inner leave-one-parent-out validation on the other natural scans.
Synthetic data never enter the primary benchmark or inflate its sample count.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.experiment_data import load_manifest, sha256
from src.experiment_metrics import regression_metrics
from src.research_evaluation import (TrainTransform, ScaleAwareRegressor, scale_features,
                                     parent_weights, parent_macro_mse)
from scripts.run_experiments import set_seed, write_csv, write_json

NEURAL = {'gat_standard': ('gat', False), 'gat_physics': ('gat', True),
          'sage_physics': ('sage', True), 'uniform_physics': ('uniform_gat', True),
          'pore_set_physics': ('pore_set', True)}
BASELINES = {'constant': [None], 'porosity_kc': [None], 'scaled_kc': [None],
             'physics_ridge': [0.1, 1., 10., 100.], 'descriptors': [1, 2, 4]}
CONFIGS = [dict(learning_rate=0.001, dropout=0., weight_decay=0.001),
           dict(learning_rate=0.003, dropout=0.1, weight_decay=0.001)]


def predict(model, transformed, indices, transform):
    model.eval()
    result = []
    with torch.inference_mode():
        for batch in DataLoader([transformed[i] for i in indices], batch_size=8, shuffle=False):
            value = model(batch).double() * transform.target_std + transform.target_mean + batch.offset
            result.extend(value.tolist())
    return np.asarray(result)


def fit_neural(name, records, graphs, train, validation, seed, config, epochs, patience,
               fixed_epochs=None, geometry=None):
    set_seed(seed)
    architecture, physics = NEURAL[name]
    transform = TrainTransform.fit(records, graphs, train, physics)
    # Transform only needed samples: no test labels enter the optimization or selection.
    needed = sorted(set(train) | set(validation))
    transformed = {i: transform.apply(records[i], graphs[i]) for i in needed}
    model = ScaleAwareRegressor(architecture, physics=physics, dropout=config['dropout'],
                                geometry=geometry)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'],
                                  weight_decay=config['weight_decay'])
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader([transformed[i] for i in train], batch_size=8, shuffle=True, generator=generator)
    weights = dict(zip(train, parent_weights(records, train)))
    for i in train:
        transformed[i].weight = torch.tensor([weights[i]], dtype=torch.float32)
    best, best_epoch, best_state, stale = float('inf'), 0, None, 0
    history = []
    for epoch in range(1, (fixed_epochs or epochs) + 1):
        model.train()
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = torch.mean(batch.weight * (model(batch) - batch.y.reshape(-1)) ** 2)
            if not torch.isfinite(loss): raise ValueError('Nonfinite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
        if fixed_epochs is not None:
            continue
        values = predict(model, transformed, validation, transform)
        score = parent_macro_mse(records, validation, values)
        history.append({'epoch': epoch, 'validation_parent_macro_mse': score})
        if score < best:
            best, best_epoch, stale = score, epoch, 0
            best_state = deepcopy(model.state_dict())
        else:
            stale += 1
        if stale >= patience: break
    if fixed_epochs is None:
        model.load_state_dict(best_state)
    else:
        best_epoch = fixed_epochs
    return model, transform, {'selected_epoch': best_epoch, 'validation_mse': best,
                               'history': history}


def baseline(name, parameter, records, graphs, train, test, seed=2026):
    y = np.asarray([r['true_log10_k_m2'] for r in records])
    weights = parent_weights(records, train)
    phi = np.asarray([r['porosity'] for r in records])
    if name == 'constant': return np.full(len(test), np.average(y[train], weights=weights))
    scaled = [scale_features(r, g) for r, g in zip(records, graphs)]
    offsets = np.asarray([f[2] for f in scaled])
    if name in ('porosity_kc', 'scaled_kc'):
        base = np.log10(phi ** 3 / (1 - phi) ** 2)
        if name == 'scaled_kc': base += offsets
        coefficient = np.average((y - base)[train], weights=weights)
        return base[test] + coefficient
    if name == 'physics_ridge':
        x = np.stack([f[1] for f in scaled])
        model = make_pipeline(StandardScaler(), Ridge(alpha=parameter))
        model.fit(x[train], (y - offsets)[train], ridge__sample_weight=weights)
        return model.predict(x[test]) + offsets[test]
    x = np.asarray([r['descriptors'] for r in records])
    model = ExtraTreesRegressor(n_estimators=200, min_samples_leaf=parameter,
                                random_state=seed, n_jobs=1)
    model.fit(x[train], y[train], sample_weight=weights)
    return model.predict(x[test])


def summarize(predictions, output, seeds):
    models = sorted({r['model'] for r in predictions})
    summaries, per_parent, ensemble = [], [], []
    for name in models:
        rows = [r for r in predictions if r['model'] == name]
        parents = sorted({r['parent_id'] for r in rows})
        seed_metrics = []
        for seed in seeds:
            selected = [r for r in rows if r['seed'] == seed]
            pooled = regression_metrics([r['true_log10_k_m2'] for r in selected],
                                        [r['predicted_log10_k_m2'] for r in selected])
            macro_mse = np.mean([np.mean([(r['predicted_log10_k_m2'] - r['true_log10_k_m2']) ** 2
                                         for r in selected if r['parent_id'] == p]) for p in parents])
            seed_metrics.append(dict(pooled, parent_macro_rmse=float(np.sqrt(macro_mse))))
        summary = {'model': name, 'natural_crops': len(rows) // len(seeds),
                   'natural_parents': len(parents), 'seeds': len(seeds)}
        for metric in ['rmse_log10_m2', 'mae_log10_m2', 'r2_log10_m2', 'r2_linear',
                       'factor_of_two_accuracy', 'parent_macro_rmse']:
            values = [m[metric] for m in seed_metrics if m[metric] is not None]
            summary[metric] = float(np.mean(values)) if values else None
            summary[metric + '_seed_std'] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.
        summary['geometric_mean_error_factor'] = float(10 ** summary['mae_log10_m2'])
        summaries.append(summary)
        for parent in parents:
            selected = [r for r in rows if r['parent_id'] == parent]
            ms = [regression_metrics([r['true_log10_k_m2'] for r in selected if r['seed'] == s],
                                     [r['predicted_log10_k_m2'] for r in selected if r['seed'] == s]) for s in seeds]
            per_parent.append({'model': name, 'parent_id': parent,
                               'n_crops': len(selected) // len(seeds),
                               **{key: float(np.mean([m[key] for m in ms if m[key] is not None]))
                                  if any(m[key] is not None for m in ms) else None
                                  for key in ['rmse_log10_m2', 'mae_log10_m2', 'r2_log10_m2',
                                              'factor_of_two_accuracy']}})
        for sid in sorted({r['sample_id'] for r in rows}):
            subset = [r for r in rows if r['sample_id'] == sid]
            ensemble.append({'model': name, 'sample_id': sid, 'parent_id': subset[0]['parent_id'],
                             'true_log10_k_m2': subset[0]['true_log10_k_m2'],
                             'predicted_log10_k_m2': float(np.mean([r['predicted_log10_k_m2'] for r in subset])),
                             'prediction_seed_std': float(np.std([r['predicted_log10_k_m2'] for r in subset], ddof=1))})
    write_csv(output / 'metrics.csv', summaries)
    write_csv(output / 'per_parent_metrics.csv', per_parent)
    write_csv(output / 'ensemble_predictions.csv', ensemble)
    # Exact paired sign-flip test on parent effects, not crops or seed repeats.
    effects = []
    for other in models:
        if other == 'gat_physics': continue
        parent_effects = []
        for parent in sorted({r['parent_id'] for r in ensemble}):
            def loss(model):
                selected = [r for r in predictions if r['model'] == model and r['parent_id'] == parent]
                return np.mean([(r['predicted_log10_k_m2'] - r['true_log10_k_m2']) ** 2 for r in selected])
            parent_effects.append(float(loss(other) - loss('gat_physics')))
        observed = abs(np.mean(parent_effects))
        signs = np.asarray(list(itertools.product([-1, 1], repeat=len(parent_effects))))
        probability = float(np.mean(np.abs(np.mean(signs * parent_effects, axis=1)) >= observed - 1e-14))
        effects.append({'comparator': other, 'natural_parents': len(parent_effects),
                        'parent_mse_differences': parent_effects,
                        'mean_mse_difference_positive_favors_gat_physics': float(np.mean(parent_effects)),
                        'exact_two_sided_sign_flip_p': probability})
    write_json(output / 'paired_parent_comparisons.json', {
        'comparisons': effects, 'limitation': 'Four scans; minimum possible two-sided exact p is 0.125. '
              'Cross-validation training folds overlap; effects are exploratory, not population-level evidence.'})
    make_report(output, summaries, per_parent)


def make_report(output, summaries, per_parent):
    sorted_metrics = sorted(summaries, key=lambda r: r['parent_macro_rmse'])
    lines = ['# TopoFlow natural rock evaluation', '',
             'The primary endpoint is permeability error on a held-out natural scan. '
             'Synthetic media are excluded from all primary training, validation and testing. '
             'All four available natural scans rotate through the outer test fold; inner parent validation '
             'selects settings and epoch counts without the outer test labels. Five seeds measure '
             'optimization variation, not five independent datasets.', '',
             'These existing scans have already been examined during development. This is an exploratory '
             'nested cross-validation study, not a newly acquired untouched external validation set.', '',
             '| Model | Parent macro RMSE log10 K | Crop RMSE log10 K | MAE log10 K | R2 log | R2 linear | Within factor 2 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for r in sorted_metrics:
        lines.append(f"| {r['model']} | {r['parent_macro_rmse']:.4f} +/- {r['parent_macro_rmse_seed_std']:.4f} "
                     f"| {r['rmse_log10_m2']:.4f} | {r['mae_log10_m2']:.4f} | {r['r2_log10_m2']:.4f} "
                     f"| {r['r2_linear']:.4f} | {100*r['factor_of_two_accuracy']:.1f}% |")
    lines += ['', 'The +/- values are sample standard deviations over training seeds. '
              'Parent macro RMSE is the square root of the mean of four parent MSE values; '
              'crop RMSE weights individual crops equally. R2 is calculated in both explicitly named spaces. '
              'Targets are cylindrical-throat PNM approximations in square metres, not LBM or measured permeability.', '',
              '## Per scan results', '', '| Model | Held-out scan | Crops | RMSE log10 K | Within factor 2 |',
              '|---|---|---:|---:|---:|']
    for r in per_parent:
        lines.append(f"| {r['model']} | {r['parent_id']} | {r['n_crops']} | {r['rmse_log10_m2']:.4f} "
                     f"| {100*r['factor_of_two_accuracy']:.1f}% |")
    lines += ['', '## Interpretation and paper use', '',
              'gat_standard retains the original GAT and mean readout with train-only feature and target '
              'standardization. The physics variants predict a dimensionless residual plus twice the log '
              'geometric mean pore diameter. They also receive the same dimensionless graph descriptors; '
              'GAT, uniform attention, GraphSAGE and the edge-free pore-set control have matched descriptor heads. '
              'The physics variant changes both dimensional scaling and graph inputs, so it is a combined '
              'ablation rather than evidence isolating either change.', '',
              'selected_gat chooses between the two GAT formulations using inner validation only. '
              'selected_all chooses among all model families by the same rule. Their outer metrics '
              'evaluate selection strategies rather than retrospectively choosing the best test score.', '',
              'The four-scan sample cannot establish a universal topology threshold or statistically '
              'significant architectural superiority. A threshold outside the observed CV range is untested, '
              'not refuted. A PNM conservation residual validates the linear solve numerically; it does not '
              'validate the pore geometry or cylindrical closure against voxel-resolved flow.', '',
              'See protocol.json, inner_selection.csv, predictions.csv, metrics.csv, per_parent_metrics.csv, '
              'paired_parent_comparisons.json and checkpoints/ for the complete audit trail. '
              'Runtime measurements are reported separately; no CNN resource advantage is inferred from parameter count.']
    (output / 'research_report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default='data/benchmark_dataset/manifest.json')
    parser.add_argument('--output', default='experiments/topoflow_research_v2')
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--seeds', type=int, nargs='+', default=[11, 22, 33, 44, 55])
    args = parser.parse_args()
    if min(args.epochs, args.patience) < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error('Positive training limits and unique seeds are required')
    torch.set_num_threads(1)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'checkpoints').mkdir()
    _, records, graphs = load_manifest(args.manifest)
    natural = [i for i, r in enumerate(records) if r['formation'] != 'Synthetic']
    parents = sorted({records[i]['parent_id'] for i in natural})
    if len(parents) < 4: raise ValueError('Nested study requires at least four natural parents')
    protocol = {'status': 'running', 'created_utc': datetime.now(timezone.utc).isoformat(),
                'args': vars(args), 'natural_parents': parents, 'natural_crops': len(natural),
                'synthetic_samples_excluded': len(records) - len(natural),
                'outer': 'leave-one-natural-parent-out', 'inner': 'leave-one-remaining-parent-out',
                'primary_score': 'parent-balanced mean squared log10 permeability error',
                'neural_configs': CONFIGS, 'baseline_configs': BASELINES, 'selection_seed': 2026,
                'final_epoch_rule': 'rounded median of best inner epochs of selected configuration',
                'target': 'log10(k / 1 m^2); cylindrical-throat PNM approximation',
                'test_reuse': 'Existing data previously inspected; exploratory model-development study',
                'manifest_sha256': sha256(args.manifest), 'python': platform.python_version(),
                'torch': torch.__version__, 'device': 'cpu', 'threads': 1,
                'source_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in
                                  [Path(__file__), ROOT/'src/research_evaluation.py', ROOT/'src/experiment_models.py',
                                   ROOT/'src/model.py', ROOT/'src/experiment_data.py', ROOT/'src/experiment_metrics.py']}}
    write_json(output / 'protocol.json', protocol)
    write_json(output / 'data_inventory.json', [records[i] for i in natural])
    selections, predictions = [], []
    started = time.perf_counter()
    for outer in parents:
        train_outer = [i for i in natural if records[i]['parent_id'] != outer]
        test = [i for i in natural if records[i]['parent_id'] == outer]
        remaining = [p for p in parents if p != outer]
        choices = {}
        for name in list(BASELINES) + list(NEURAL):
            candidates = CONFIGS if name in NEURAL else BASELINES[name]
            candidate_results = []
            for ci, config in enumerate(candidates):
                scores, epochs = [], []
                for validation_parent in remaining:
                    train = [i for i in train_outer if records[i]['parent_id'] != validation_parent]
                    validation = [i for i in train_outer if records[i]['parent_id'] == validation_parent]
                    if name in NEURAL:
                        model, transform, info = fit_neural(name, records, graphs, train, validation,
                                                            2026, config, args.epochs, args.patience)
                        transformed = {i: transform.apply(records[i], graphs[i]) for i in validation}
                        values = predict(model, transformed, validation, transform)
                        epochs.append(info['selected_epoch'])
                    else:
                        values = baseline(name, config, records, graphs, train, validation)
                    score = parent_macro_mse(records, validation, values)
                    scores.append(score)
                    selections.append({'outer_test_parent': outer, 'model': name, 'candidate': ci,
                                       'inner_validation_parent': validation_parent,
                                       'inner_train_parents': '|'.join(p for p in remaining if p != validation_parent),
                                       'validation_macro_mse': score,
                                       'best_epoch': epochs[-1] if epochs else None})
                candidate_results.append((float(np.mean(scores)), ci,
                                          max(1, int(np.rint(np.median(epochs))))) if epochs else
                                         (float(np.mean(scores)), ci, None))
            score, ci, epochs = min(candidate_results, key=lambda x: (x[0], x[1]))
            choices[name] = {'inner_macro_mse': score, 'config': candidates[ci], 'epochs': epochs}
            print(f'{outer}: {name} selected inner MSE={score:.4f}, epochs={epochs}', flush=True)
            write_csv(output / 'inner_selection.csv', selections)
            for seed in args.seeds:
                if name in NEURAL:
                    model, transform, _ = fit_neural(name, records, graphs, train_outer, [], seed,
                                                     candidates[ci], args.epochs, args.patience,
                                                     fixed_epochs=epochs)
                    transformed = {i: transform.apply(records[i], graphs[i]) for i in test}
                    values = predict(model, transformed, test, transform)
                    checkpoint = output/'checkpoints'/f'{outer}_{name}_seed{seed}.pth'
                    torch.save({'state_dict': model.state_dict(), 'transform': transform,
                                'model': name, 'seed': seed, 'outer_test_parent': outer,
                                'train_parents': remaining, 'config': candidates[ci], 'epochs': epochs}, checkpoint)
                else:
                    values = baseline(name, candidates[ci], records, graphs, train_outer, test, seed)
                for i, value in zip(test, values):
                    if not np.isfinite(value): raise ValueError('Nonfinite prediction')
                    predictions.append({'sample_id': records[i]['id'], 'parent_id': outer,
                                        'model': name, 'seed': seed, 'selection': name,
                                        'true_log10_k_m2': records[i]['true_log10_k_m2'],
                                        'predicted_log10_k_m2': float(value)})
            write_csv(output/'predictions.csv', predictions)
        for selector, available in [('selected_gat', ['gat_standard', 'gat_physics']),
                                    ('selected_all', list(choices))]:
            selected_name = min(available, key=lambda n: choices[n]['inner_macro_mse'])
            selected_rows = [r for r in predictions if r['parent_id'] == outer and r['model'] == selected_name]
            predictions.extend([dict(r, model=selector, selection=selected_name) for r in selected_rows])
        write_json(output/f'selection_{outer}.json', choices)
        write_csv(output/'predictions.csv', predictions)
        print(f'Finished held-out parent {outer}; elapsed {time.perf_counter()-started:.1f}s', flush=True)
    summarize(predictions, output, args.seeds)
    protocol.update(status='complete', elapsed_seconds=time.perf_counter() - started)
    write_json(output/'protocol.json', protocol)
    print(f'Completed: {output}', flush=True)


if __name__ == '__main__': main()
