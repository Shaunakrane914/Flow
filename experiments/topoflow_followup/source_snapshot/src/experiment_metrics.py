"""Paired regression metrics in explicitly defined spaces."""
import numpy as np
import torch


def paired_loss(prediction, target, criterion):
    prediction, target = prediction.reshape(-1), target.reshape(-1)
    if prediction.shape != target.shape or prediction.numel() == 0:
        raise ValueError('Prediction and target must have the same nonzero number of samples')
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError('Nonfinite prediction or target')
    return criterion(prediction, target)


def regression_metrics(true_log, predicted_log):
    true_log, predicted_log = np.asarray(true_log, dtype=float), np.asarray(predicted_log, dtype=float)
    if true_log.ndim != 1 or true_log.shape != predicted_log.shape or not true_log.size:
        raise ValueError('Metrics require equal nonempty one-dimensional arrays')
    if not np.isfinite(true_log).all() or not np.isfinite(predicted_log).all():
        raise ValueError('Metrics received nonfinite values')
    error = predicted_log - true_log
    mse = float(np.mean(error ** 2))
    denominator = float(np.sum((true_log - true_log.mean()) ** 2))
    with np.errstate(over='ignore', invalid='ignore'):
        true_k, predicted_k = 10. ** true_log, 10. ** predicted_log
        physical_error = predicted_k - true_k
        physical_mse = float(np.mean(physical_error ** 2))
        physical_var = float(np.sum((true_k - true_k.mean()) ** 2))
        values = dict(n=len(error), mse_log10_m2=mse, rmse_log10_m2=float(np.sqrt(mse)),
                      mae_log10_m2=float(np.mean(np.abs(error))),
                      r2_log10_m2=1 - float(np.sum(error ** 2)) / denominator if denominator > 0 else None,
                      mae_m2=float(np.mean(np.abs(physical_error))),
                      rmse_m2=float(np.sqrt(physical_mse)),
                      r2_linear=1 - float(np.sum(physical_error ** 2)) / physical_var if physical_var > 0 else None,
                      mean_absolute_relative_error=float(np.mean(np.abs(physical_error) / true_k)),
                      factor_of_two_accuracy=float(np.mean(np.abs(error) <= np.log10(2))))
    return {key: value if value is None or np.isfinite(value) else None for key, value in values.items()}


def paired_parent_interval(rows, bootstrap_repeats=2000, seed=42):
    """Parent-balanced mean paired effect; seeds have already been averaged per sample."""
    grouped = {}
    for row in rows:
        grouped.setdefault(row['parent_id'], []).append(row['squared_log_error_difference'])
    effects = np.asarray([np.mean(v) for v in grouped.values()])
    result = {'samples': len(rows), 'parents': len(effects),
              'parent_balanced_effect': float(effects.mean()) if len(effects) else None,
              'ci95': None, 'interpretation': 'positive means GAT has lower squared log error'}
    if len(effects) >= 2:
        rng = np.random.default_rng(seed)
        means = np.asarray([rng.choice(effects, len(effects), replace=True).mean()
                            for _ in range(bootstrap_repeats)])
        result['ci95'] = np.quantile(means, [.025, .975]).tolist()
    else:
        result['limitation'] = 'Fewer than two independent parents; no uncertainty interval estimated'
    return result
