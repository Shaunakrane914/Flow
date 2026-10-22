"""Train-only normalization and physically scale-equivariant permeability models."""
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch_geometric.data import Data

from src.experiment_models import make_model


def scale_features(record, graph):
    """Separate a physical length scale from dimensionless graph geometry.

    log10(k)=2*log10(d_g)+f(d/d_g,V/d_g^3,phi,connectivity).
    d_g is the geometric mean equivalent pore diameter, computed from inputs.
    """
    x = graph.x.double().numpy()
    log_d = float(x[:, 0].mean())
    dimensionless = x - np.array([log_d, 3 * log_d])
    desc = np.asarray(record['descriptors'], dtype=float)
    phi = float(record['porosity'])
    geometry = np.array([np.log10(phi), np.log10(1 - phi),
                         desc[2], desc[3] - 3 * log_d, desc[4],
                         *desc[5:]], dtype=float)
    return dimensionless, geometry, 2 * log_d


@dataclass
class TrainTransform:
    physics: bool
    node_mean: np.ndarray
    node_std: np.ndarray
    geometry_mean: np.ndarray
    geometry_std: np.ndarray
    target_mean: float
    target_std: float

    @classmethod
    def fit(cls, records, graphs, train_indices, physics):
        if not len(train_indices):
            raise ValueError('Training indices must not be empty')
        features = [scale_features(records[i], graphs[i]) for i in train_indices]
        nodes = np.concatenate([f[0] if physics else graphs[i].x.numpy()
                                for i, f in zip(train_indices, features)])
        geometry = np.stack([f[1] for f in features])
        target = np.asarray([records[i]['true_log10_k_m2'] - (f[2] if physics else 0.)
                             for i, f in zip(train_indices, features)])
        return cls(physics, nodes.mean(0), np.maximum(nodes.std(0), 1e-5),
                   geometry.mean(0), np.maximum(geometry.std(0), 1e-5),
                   float(target.mean()), max(float(target.std()), 1e-3))

    def apply(self, record, graph):
        nodes, geometry, offset = scale_features(record, graph)
        if not self.physics:
            nodes, offset = graph.x.numpy(), 0.
        return Data(x=torch.tensor((nodes - self.node_mean) / self.node_std, dtype=torch.float32),
                    edge_index=graph.edge_index.clone(),
                    geometry=torch.tensor((geometry - self.geometry_mean) / self.geometry_std,
                                          dtype=torch.float32).unsqueeze(0),
                    y=torch.tensor([(record['true_log10_k_m2'] - offset - self.target_mean)
                                    / self.target_std], dtype=torch.float32),
                    offset=torch.tensor([offset], dtype=torch.float64))


class ScaleAwareRegressor(nn.Module):
    """Existing GAT/SAGE/pore-set encoder with a matched graph-geometry residual head."""
    def __init__(self, architecture, geometry_features=10, physics=True, dropout=0.1,
                 geometry=None):
        super().__init__()
        self.physics = physics
        self.use_geometry = physics if geometry is None else geometry
        self.encoder = make_model(architecture, dropout)
        if self.use_geometry:
            self.geometry_head = nn.Sequential(nn.Linear(geometry_features, 16), nn.ELU(),
                                               nn.Dropout(dropout), nn.Linear(16, 1))

    def forward(self, batch):
        value = self.encoder(batch.x, batch.edge_index, batch.batch).reshape(-1)
        if self.use_geometry:
            value = value + self.geometry_head(batch.geometry).reshape(-1)
        return value


def parent_weights(records, indices):
    counts = {}
    for i in indices:
        p = records[i]['parent_id']
        counts[p] = counts.get(p, 0) + 1
    return np.asarray([len(indices) / (len(counts) * counts[records[i]['parent_id']])
                       for i in indices])


def parent_macro_mse(records, indices, prediction):
    groups = {}
    for i, p in zip(indices, prediction):
        groups.setdefault(records[i]['parent_id'], []).append(
            (float(p) - records[i]['true_log10_k_m2']) ** 2)
    if not groups:
        raise ValueError('Evaluation indices must not be empty')
    return float(np.mean([np.mean(values) for values in groups.values()]))
