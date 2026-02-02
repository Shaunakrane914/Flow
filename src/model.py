"""
Topo-Flow GNN Architecture
Phase 3.1: Graph Attention Network for Permeability Prediction

Architecture:
- 3 GAT layers with multi-head attention
- Global mean pooling
- MLP prediction head
- Output: log-scaled permeability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool


class TopoFlowGNN(nn.Module):
    """
    Graph Attention Network for predicting permeability from pore network graphs
    
    Architecture:
        Input: Node features [pore.diameter, pore.volume] (log-scaled)
        Layer 1: GAT with 2 attention heads (2 → 64*2 = 128)
        Layer 2: GAT with 1 attention head (128 → 64)
        Layer 3: GAT with 1 attention head (64 → 64)
        Pooling: Global mean pooling (graph-level embedding)
        MLP Head: 64 → 32 → 1 (permeability prediction)
    """
    
    def __init__(self, dropout=0.1):
        """
        Initialize the GNN model
        
        Parameters:
        -----------
        dropout : float
            Dropout probability for regularization (default: 0.1)
        """
        super(TopoFlowGNN, self).__init__()
        
        # Graph Attention Layers
        self.gat1 = GATConv(
            in_channels=2,      # Input features: [diameter, volume]
            out_channels=64,    # Per-head output
            heads=2,            # 2 attention heads
            dropout=dropout
        )
        # After gat1: 64 * 2 = 128 dimensions (heads concatenated)
        
        self.gat2 = GATConv(
            in_channels=128,    # From previous layer (64 * 2 heads)
            out_channels=64,
            heads=1,            # Single attention head
            dropout=dropout
        )
        
        self.gat3 = GATConv(
            in_channels=64,
            out_channels=64,
            heads=1,
            dropout=dropout
        )
        
        # MLP Prediction Head
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 1)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, edge_index, batch):
        """
        Forward pass through the network
        
        Parameters:
        -----------
        x : torch.Tensor
            Node features [num_nodes, 2] (log-scaled diameter and volume)
        edge_index : torch.Tensor
            Graph connectivity [2, num_edges]
        batch : torch.Tensor
            Batch assignment vector [num_nodes]
            Maps each node to its graph in the batch
        
        Returns:
        --------
        torch.Tensor
            Predicted permeability [batch_size, 1] (log-scaled)
        """
        
        # Layer 1: GAT with 2 heads
        x = self.gat1(x, edge_index)
        x = F.elu(x)  # ELU activation (standard for GATs)
        x = self.dropout(x)
        
        # Layer 2: GAT with 1 head
        x = self.gat2(x, edge_index)
        x = F.elu(x)
        x = self.dropout(x)
        
        # Layer 3: GAT with 1 head
        x = self.gat3(x, edge_index)
        x = F.elu(x)
        x = self.dropout(x)
        
        # Global Pooling: Aggregate node features into graph-level representation
        # batch vector indicates which graph each node belongs to
        x = global_mean_pool(x, batch)  # [batch_size, 64]
        
        # MLP Prediction Head
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)  # [batch_size, 1]
        
        return x


def count_parameters(model):
    """Count trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    """
    Test the model architecture
    """
    print("="*60)
    print("🧠 TOPO-FLOW GNN ARCHITECTURE TEST")
    print("="*60)
    
    # Create model
    model = TopoFlowGNN(dropout=0.1)
    
    print(f"\n📊 Model Architecture:")
    print(model)
    
    print(f"\n🔢 Model Statistics:")
    print(f"  Total Parameters: {count_parameters(model):,}")
    
    # Test with dummy data
    print(f"\n🧪 Testing with dummy batch...")
    
    # Simulate a batch of 2 graphs
    num_nodes_graph1 = 50
    num_nodes_graph2 = 70
    total_nodes = num_nodes_graph1 + num_nodes_graph2
    
    # Random node features (log-scaled diameter and volume)
    x = torch.randn(total_nodes, 2)
    
    # Random edges
    edge_index = torch.randint(0, total_nodes, (2, 200))
    
    # Batch vector (first 50 nodes → graph 0, next 70 → graph 1)
    batch = torch.cat([
        torch.zeros(num_nodes_graph1, dtype=torch.long),
        torch.ones(num_nodes_graph2, dtype=torch.long)
    ])
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(x, edge_index, batch)
    
    print(f"  Input: {total_nodes} nodes, 2 graphs")
    print(f"  Output shape: {output.shape}")
    print(f"  Sample predictions (log10 K):")
    print(f"    Graph 1: {output[0].item():.4f}")
    print(f"    Graph 2: {output[1].item():.4f}")
    
    print(f"\n✅ Model architecture is ready for training!")
    print("="*60)
