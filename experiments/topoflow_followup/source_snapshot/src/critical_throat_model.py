"""Geometry-only edge models with a bounded, single-throat response readout.

Hydraulic importance is a separate learned head. It must not be confused with
the neighborhood-normalized attention coefficients inside a GAT layer.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import softmax,scatter


class EdgeMean(nn.Module):
    def __init__(self,inputs,outputs):
        super().__init__()
        self.message=nn.Linear(inputs+6,outputs)
        self.root=nn.Linear(inputs,outputs)

    def forward(self,x,edges,edge_attr):
        a,b=edges
        messages=self.message(torch.cat([x[a],edge_attr],dim=1))
        return self.root(x)+scatter(messages,b,dim=0,dim_size=x.shape[0],reduce='mean')


class CriticalThroatRegressor(nn.Module):
    def __init__(self,architecture='edge_gat'):
        super().__init__();self.architecture=architecture
        if architecture=='edge_mlp':self.layers=nn.ModuleList([nn.Linear(7,32),nn.Linear(32,32),nn.Linear(32,32)])
        elif architecture=='edge_sage':self.layers=nn.ModuleList([EdgeMean(7,32),EdgeMean(32,32),EdgeMean(32,32)])
        else:self.layers=nn.ModuleList([GATConv(7,32,edge_dim=6),GATConv(32,32,edge_dim=6),GATConv(32,32,edge_dim=6)])
        self.edge_head=nn.Sequential(nn.Linear(70,32),nn.ELU(),nn.Linear(32,2))

    def forward(self,batch):
        x=batch.x
        for layer in self.layers:
            x=F.elu(layer(x) if self.architecture=='edge_mlp' else layer(x,batch.edge_index,batch.edge_attr))
        a,b=batch.throat_index
        logits=self.edge_head(torch.cat([x[a]+x[b],torch.abs(x[a]-x[b]),batch.throat_attr],dim=1))
        group=batch.batch[a]
        importance=softmax(logits[:,0],group)
        leverage=(1-importance)*torch.sigmoid(logits[:,1])
        return importance,leverage


def fractional_response(importance,leverage,restrictions=(.1,.25,.5)):
    ratios=(1-torch.as_tensor(restrictions,dtype=importance.dtype,device=importance.device))**4
    change=ratios[None,:]-1
    return -change*importance[:,None]/(1+change*leverage[:,None])
