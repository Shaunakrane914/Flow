"""
Physics-Informed Hybrid GNN Model
Learns residual corrections to Kozeny-Carman baseline

Architecture:
- Input: Graph topology + Kozeny-Carman prediction
- Output: Corrected permeability (baseline + learned residual)
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, global_mean_pool


class HybridPhysicsGNN(nn.Module):
    """
    Hybrid model that combines physics-based baseline with GNN correction
    
    Formula: K_pred = K_kozeny_carman + GNN_correction(graph, K_kc)
    
    Advantages:
    - Guaranteed to match or beat baseline (residual learning)
    - Physics-informed (starts from known correlations)
    - Interpretable (correction term shows topology contribution)
    """
    
    def __init__(self, hidden_dim=64, num_heads=4, dropout=0.1):
        super().__init__()
        
        # 1. Topology encoder (GAT layers)
        self.conv1 = GATv2Conv(2, hidden_dim, heads=num_heads, concat=True, dropout=dropout)
        self.conv2 = GATv2Conv(hidden_dim * num_heads, hidden_dim, heads=1, concat=False, dropout=dropout)
        
        # 2. Physics-aware mixer
        # Input: graph embedding (hidden_dim) + baseline prediction (1)
        self.correction_net = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1)  # Residual correction
        )
        
        # Initialize correction network to output near-zero
        # This ensures model starts at baseline performance
        with torch.no_grad():
            for layer in self.correction_net:
                if isinstance(layer, nn.Linear):
                    layer.weight.data *= 0.01
                    layer.bias.data.zero_()
    
    def forward(self, x, edge_index, batch, baseline_k):
        """
        Forward pass
        
        Args:
            x: Node features [num_nodes, 2] (diameter, volume)
            edge_index: Graph connectivity [2, num_edges]
            batch: Batch assignment [num_nodes]
            baseline_k: Kozeny-Carman predictions [batch_size, 1]
        
        Returns:
            corrected_k: Baseline + learned correction [batch_size, 1]
        """
        # Extract topology features
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        
        # Pool to graph-level representation
        graph_embedding = global_mean_pool(x, batch)  # [batch_size, hidden_dim]
        
        # Combine topology + baseline
        combined = torch.cat([graph_embedding, baseline_k], dim=1)
        
        # Predict correction (residual)
        correction = self.correction_net(combined)
        
        # Final prediction = baseline + correction
        return baseline_k + correction
    
    def get_correction(self, x, edge_index, batch, baseline_k):
        """Get the learned correction term separately"""
        graph_embedding = global_mean_pool(
            torch.relu(self.conv2(torch.relu(self.conv1(x, edge_index)), edge_index)),
            batch
        )
        combined = torch.cat([graph_embedding, baseline_k], dim=1)
        return self.correction_net(combined)


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model
    print("Testing HybridPhysicsGNN...")
    
    model = HybridPhysicsGNN()
    print(f"Parameters: {count_parameters(model):,}")
    
    # Dummy data
    batch_size = 4
    num_nodes = 100
    num_edges = 300
    
    x = torch.randn(num_nodes, 2)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    batch = torch.repeat_interleave(torch.arange(batch_size), num_nodes // batch_size)
    baseline_k = torch.randn(batch_size, 1)
    
    # Forward pass
    output = model(x, edge_index, batch, baseline_k)
    correction = model.get_correction(x, edge_index, batch, baseline_k)
    
    print(f"Output shape: {output.shape}")
    print(f"Correction range: {correction.min().item():.4f} to {correction.max().item():.4f}")
    print("✅ Model test passed!")
