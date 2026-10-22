"""Matched comparators for the TopoFlow research experiments."""
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GATConv, SAGEConv, global_mean_pool
from torch_geometric.utils import softmax

from src.model import TopoFlowGNN


class UniformGATConv(GATConv):
    """GAT projections and heads with uniform incoming weights, including self loops."""

    def edge_update(self, alpha_j, alpha_i, edge_attr, index, ptr, size_i):
        weights = softmax(torch.zeros_like(alpha_j), index, ptr, size_i)
        return F.dropout(weights, p=self.dropout, training=self.training)


class GraphSAGERegressor(nn.Module):
    """Same widths, activation, dropout and readout as TopoFlow; SAGE mean messages."""

    def __init__(self, dropout=0.1):
        super().__init__()
        self.convs = nn.ModuleList([SAGEConv(2, 128), SAGEConv(128, 64), SAGEConv(64, 64)])
        self.fc1, self.fc2 = nn.Linear(64, 32), nn.Linear(32, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = self.dropout(F.elu(conv(x, edge_index)))
        x = global_mean_pool(x, batch)
        return self.fc2(self.dropout(F.relu(self.fc1(x))))


class PoreSetRegressor(GraphSAGERegressor):
    """Feature-only control with the same layer widths and graph-level mean readout."""

    def __init__(self, dropout=0.1):
        super().__init__(dropout)
        self.convs = nn.ModuleList([nn.Linear(2, 128), nn.Linear(128, 64), nn.Linear(64, 64)])

    def forward(self, x, edge_index, batch):
        for layer in self.convs:
            x = self.dropout(F.elu(layer(x)))
        return self.fc2(self.dropout(F.relu(self.fc1(global_mean_pool(x, batch)))))


def make_model(name, dropout=0.1):
    if name == 'gat':
        return TopoFlowGNN(dropout=dropout)
    if name == 'sage':
        return GraphSAGERegressor(dropout=dropout)
    if name == 'pore_set':
        return PoreSetRegressor(dropout=dropout)
    if name == 'uniform_gat':
        model = TopoFlowGNN(dropout=dropout)
        model.gat1 = UniformGATConv(2, 64, heads=2, dropout=dropout)
        model.gat2 = UniformGATConv(128, 64, heads=1, dropout=dropout)
        model.gat3 = UniformGATConv(64, 64, heads=1, dropout=dropout)
        for conv in (model.gat1, model.gat2, model.gat3):
            conv.att_src.requires_grad_(False)
            conv.att_dst.requires_grad_(False)
        return model
    raise ValueError(f'Unknown model: {name}')
