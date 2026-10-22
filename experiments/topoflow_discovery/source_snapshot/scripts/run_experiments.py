"""Experiments 1 and 2: validation-selected models and paired pore-CV analysis."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import platform
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import sklearn
from sklearn.ensemble import ExtraTreesRegressor
import torch
import torch_geometric
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.experiment_data import load_manifest, sha256
from src.experiment_metrics import paired_loss, regression_metrics, paired_parent_interval
from src.experiment_models import make_model


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=str), encoding='utf-8')


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, graphs, batch_size, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in DataLoader(graphs, batch_size=batch_size, shuffle=False):
            batch = batch.to(device)
            values = model(batch.x, batch.edge_index, batch.batch).reshape(-1)
            if not torch.isfinite(values).all():
                raise ValueError('Model emitted nonfinite predictions')
            predictions.extend(values.cpu().double().tolist())
    return np.asarray(predictions)


def train_model(name, seed, graphs, indices, args, output):
    set_seed(seed)
    model = make_model(name, args.dropout).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader([graphs[i] for i in indices['train']], batch_size=args.batch_size,
                              shuffle=True, generator=generator)
    validation = [graphs[i] for i in indices['validation']]
    true_validation = np.asarray([g.y.item() for g in validation])
    criterion = torch.nn.MSELoss()
    best, best_epoch, stale, history = float('inf'), None, 0, []
    checkpoint = output / f'{name}_seed{seed}.pth'
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.
        for batch in train_loader:
            batch = batch.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            loss = paired_loss(model(batch.x, batch.edge_index, batch.batch), batch.y, criterion)
            loss.backward()
            optimizer.step()
            total += loss.item() * batch.num_graphs
        predicted_validation = evaluate(model, validation, args.batch_size, args.device)
        validation_mse = float(np.mean((predicted_validation - true_validation) ** 2))
        history.append({'epoch': epoch, 'train_mse_log10_m2': total / len(indices['train']),
                        'validation_mse_log10_m2': validation_mse})
        if validation_mse < best:
            best, best_epoch, stale = validation_mse, epoch, 0
            torch.save({'model': name, 'seed': seed, 'state_dict': model.state_dict(),
                        'selected_epoch': epoch, 'validation_mse_log10_m2': best,
                        'config': vars(args)}, checkpoint)
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(f'{name} seed={seed} epoch={epoch} validation MSE={validation_mse:.6g}', flush=True)
        if stale >= args.patience:
            break
    write_csv(output / f'{name}_seed{seed}_history.csv', history)
    model.load_state_dict(torch.load(checkpoint, map_location=args.device, weights_only=False)['state_dict'])
    return model, {'model': name, 'seed': seed, 'selected_epoch': best_epoch,
                   'validation_mse_log10_m2': best,
                   'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
                   'checkpoint': checkpoint.name, 'checkpoint_sha256': sha256(checkpoint)}


def analyze_heterogeneity(predictions, records, output, args):
    train_cv = [r['pore_radius_cv'] for r in records if r['split'] == 'train']
    cuts = np.quantile(train_cv, [1 / 3, 2 / 3])
    table = {}
    for row in predictions:
        if row['split'] == 'test':
            table.setdefault(row['sample_id'], {}).setdefault(row['model'], {})[row['seed']] = row
    differences, groups = [], {}
    for sample_id, models in table.items():
        for comparator in ('sage', 'uniform_gat', 'pore_set', 'descriptors', 'kc', 'constant'):
            if 'gat' not in models or comparator not in models:
                continue
            common_seeds = sorted(set(models['gat']) & set(models[comparator]))
            gat = [models['gat'][seed] for seed in common_seeds]
            other = [models[comparator][seed] for seed in common_seeds]
            if not gat:
                continue
            exemplar = gat[0]
            cv = exemplar['pore_radius_cv']
            cv_bin = ('low', 'medium', 'high')[int(np.searchsorted(cuts, cv, side='right'))]
            effect = np.mean([o['squared_log_error'] - g['squared_log_error'] for g, o in zip(gat, other)])
            row = {'sample_id': sample_id, 'parent_id': exemplar['parent_id'],
                   'formation': exemplar['formation'], 'pore_radius_cv': cv, 'cv_bin': cv_bin,
                   'comparator': comparator, 'seeds': len(common_seeds),
                   'gat_mean_squared_log_error': float(np.mean([g['squared_log_error'] for g in gat])),
                   'comparator_mean_squared_log_error': float(np.mean([o['squared_log_error'] for o in other])),
                   'squared_log_error_difference': float(effect)}
            differences.append(row)
            for scope, value in [('all', 'all'), ('cv_bin', cv_bin), ('formation', row['formation'])]:
                groups.setdefault((comparator, scope, value), []).append(row)
    write_csv(output / 'heterogeneity_pairs.csv', differences)
    summaries = [{'comparator': comparator, 'scope': scope, 'value': value,
                  **paired_parent_interval(rows, args.bootstrap_repeats)}
                 for (comparator, scope, value), rows in groups.items()]
    write_json(output / 'heterogeneity.json', {
        'purpose': args.purpose, 'cv_definition': 'population std(pore radius) / mean(pore radius)',
        'bin_boundaries_from_train_only': cuts.tolist(),
        'threshold_estimated': False,
        'inference_limits': 'Exploratory association; bins do not remove formation confounding. '
                            'Intervals resample parents after averaging paired seed errors per sample. '
                            'Few parents cannot support a population-level discovery.',
        'paired_comparisons': summaries})
    if differences:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        selected = [r for r in differences if r['comparator'] == 'sage']
        if selected:
            fig, ax = plt.subplots(figsize=(8, 5))
            for formation in sorted({r['formation'] for r in selected}):
                rows = [r for r in selected if r['formation'] == formation]
                ax.scatter([r['pore_radius_cv'] for r in rows],
                           [r['squared_log_error_difference'] for r in rows], label=formation, alpha=.7)
            ax.axhline(0, color='black', linewidth=.8)
            ax.set(xlabel='Per-sample pore-radius CV', ylabel='GraphSAGE squared log error minus GAT error',
                   title=f'{args.purpose.upper()}: paired held-out errors (positive favors GAT)')
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(output / 'heterogeneity.png', dpi=180)
            plt.close(fig)


def run(args):
    manifest, records, graphs = load_manifest(args.manifest, args.protocol)
    if manifest['purpose'] != args.purpose:
        raise ValueError('--purpose must match the manifest; pilot data cannot silently become research data')
    output = Path(args.output)
    if output.exists():
        raise ValueError(f'Output already exists: {output}; choose a new run directory')
    output.mkdir(parents=True)
    indices = {split: [i for i, r in enumerate(records) if r['split'] == split]
               for split in ('train', 'validation', 'test')}
    y = np.asarray([r['true_log10_k_m2'] for r in records])
    descriptors = np.asarray([r['descriptors'] for r in records])
    phi = np.asarray([r['porosity'] for r in records])
    kc_base = np.log10(phi ** 3 / (1 - phi) ** 2)
    # Fit C in log-space, the same objective used by the GNN; never use validation/test labels.
    log_c = float(np.mean((y - kc_base)[indices['train']]))
    constant = float(y[indices['train']].mean())
    source_files = [Path(__file__), ROOT / 'src/model.py', ROOT / 'src/experiment_models.py',
                    ROOT / 'src/experiment_metrics.py', ROOT / 'src/experiment_data.py']
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    args.purpose = manifest['purpose']
    metadata = {'status': 'running', 'purpose': args.purpose, 'config': vars(args),
                'created_utc': datetime.now(timezone.utc).isoformat(), 'git_commit': commit,
                'python': platform.python_version(), 'torch': torch.__version__,
                'torch_geometric': torch_geometric.__version__, 'sklearn': sklearn.__version__,
                'numpy': np.__version__, 'platform': platform.platform(),
                'manifest_sha256': sha256(args.manifest),
                'source_sha256': {p.name: sha256(p) for p in source_files},
                'split_counts': {s: {'samples': len(ix), 'parents': len({records[i]['parent_id'] for i in ix})}
                                 for s, ix in indices.items()},
                'kc_log10_coefficient_m2': log_c, 'constant_train_log10_k_m2': constant,
                'target': 'log10(k / 1 m^2), with original additive epsilon removed',
                'descriptors': ['porosity', 'mean_log_diameter', 'std_log_diameter', 'mean_log_volume',
                                'std_log_volume', 'log_nodes', 'log_directed_edges', 'mean_degree', 'std_degree', 'radius_cv'],
                'seed_protocol': 'Same seeds, widths, optimizer, batch order and selection rule. '
                                 'Architectures have different parameter counts; no claim of identical capacity.',
                'limitations': ['Manifest provenance is an author declaration, not independently certified.',
                                'Geometry and label validation remain prerequisites for physical claims.']}
    write_json(output / 'run.json', metadata)
    write_json(output / 'manifest_resolved.json', dict(manifest, samples=records))
    predictions, metrics, model_info = [], [], []

    def record_predictions(name, seed, values):
        for i, predicted in enumerate(values):
            r = records[i]
            with np.errstate(over='ignore'):
                predicted_k = float(np.power(10., predicted))
            if not np.isfinite(predicted_k) or predicted_k <= 0:
                raise ValueError(f'{name}: nonfinite/nonpositive physical prediction')
            predictions.append({'sample_id': r['id'], 'parent_id': r['parent_id'], 'formation': r['formation'],
                                'split': r['split'], 'label_source': r['label_source'], 'model': name, 'seed': seed,
                                'true_log10_k_m2': y[i], 'predicted_log10_k_m2': float(predicted),
                                'true_k_m2': r['true_k_m2'], 'predicted_k_m2': predicted_k,
                                'squared_log_error': float((predicted - y[i]) ** 2),
                                'pore_radius_cv': r['pore_radius_cv']})
        for split, ix in indices.items():
            for formation in ['__pooled__'] + sorted({records[i]['formation'] for i in ix}):
                subset = ix if formation == '__pooled__' else [i for i in ix if records[i]['formation'] == formation]
                metrics.append({'model': name, 'seed': seed, 'split': split, 'formation': formation,
                                **regression_metrics(y[subset], values[subset])})
        # Incremental export retains evidence if a later model fails.
        write_csv(output / 'predictions.csv', predictions)
        write_csv(output / 'metrics.csv', metrics)

    for seed in args.seeds:
        tree = ExtraTreesRegressor(n_estimators=200, min_samples_leaf=2, random_state=seed, n_jobs=1)
        tree.fit(descriptors[indices['train']], y[indices['train']])
        record_predictions('constant', seed, np.full(len(y), constant))
        record_predictions('kc', seed, kc_base + log_c)
        record_predictions('descriptors', seed, tree.predict(descriptors))
        # Store baseline state without adding a new serialization dependency.
        import pickle
        with (output / f'descriptors_seed{seed}.pkl').open('wb') as stream:
            pickle.dump(tree, stream)
        for name in args.models:
            model, info = train_model(name, seed, graphs, indices, args, output)
            model_info.append(info)
            # Test inference happens only after validation-based checkpoint selection.
            record_predictions(name, seed, evaluate(model, graphs, args.batch_size, args.device))
            write_json(output / 'models.json', model_info)
    analyze_heterogeneity(predictions, records, output, args)
    # Equal-weight category summary and descriptive seed variation; seeds are not independent specimens.
    summary = []
    for name in sorted({m['model'] for m in metrics}):
        rows = [m for m in metrics if m['model'] == name and m['split'] == 'test' and m['formation'] != '__pooled__']
        seed_macro = [np.mean([r['rmse_log10_m2'] for r in rows if r['seed'] == seed]) for seed in args.seeds]
        summary.append({'model': name, 'test_macro_rmse_log10_m2_mean': float(np.mean(seed_macro)),
                        'test_macro_rmse_log10_m2_seed_std': float(np.std(seed_macro, ddof=1)) if len(seed_macro) > 1 else None,
                        'seeds': len(seed_macro)})
    write_json(output / 'summary.json', {'purpose': args.purpose, 'models': summary,
                                       'note': 'Seed standard deviation is not a parent-scan confidence interval.'})
    metadata['status'] = 'complete'
    write_json(output / 'run.json', metadata)
    print(f'Completed {args.purpose} run: {output}. No universal CV threshold has been estimated.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--purpose', choices=['research', 'pilot'], default='research')
    parser.add_argument('--protocol', choices=['grouped', 'formation-holdout'], default='grouped')
    parser.add_argument('--models', nargs='+', choices=['gat', 'sage', 'uniform_gat', 'pore_set'],
                        default=['gat', 'sage', 'uniform_gat', 'pore_set'])
    parser.add_argument('--seeds', type=int, nargs='+', default=[11, 22, 33, 44, 55])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--learning-rate', type=float, default=.001)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--dropout', type=float, default=.1)
    parser.add_argument('--bootstrap-repeats', type=int, default=2000)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.batch_size, args.bootstrap_repeats, args.threads) < 1:
        parser.error('Epochs, patience, batch size, bootstrap repeats and threads must be positive')
    if len(set(args.models)) != len(args.models) or len(set(args.seeds)) != len(args.seeds):
        parser.error('Models and seeds must not contain duplicates')
    if not 0 <= args.dropout < 1 or not 0 < args.learning_rate or args.weight_decay < 0:
        parser.error('Invalid optimizer/dropout settings')
    torch.set_num_threads(args.threads)
    try:
        run(args)
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(2, f'Experiment stopped: {exc}\n')


if __name__ == '__main__':
    main()
