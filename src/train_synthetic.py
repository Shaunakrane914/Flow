"""
Train GNN on Synthetic Data
Control experiment to prove GNN works when topology matters
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
import glob
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import TopoFlowGNN, count_parameters


def load_synthetic_graphs(graphs_dir="data/graphs_synthetic"):
    """Load synthetic graph data"""
    graph_files = sorted(glob.glob(os.path.join(graphs_dir, "*.pt")))
    
    if not graph_files:
        print(f"❌ No graphs found in {graphs_dir}/")
        print("Run: python src/extract_synthetic.py first")
        return [], []
    
    # Load all graphs
    graphs = []
    for f in graph_files:
        try:
            graph = torch.load(f, weights_only=False)
            graphs.append(graph)
        except Exception as e:
            print(f"⚠️  Failed to load {f}: {e}")
    
    # 80/20 split
    split_idx = int(len(graphs) * 0.8)
    train_graphs = graphs[:split_idx]
    test_graphs = graphs[split_idx:]
    
    print(f"📊 Dataset Split (Synthetic):")
    print(f"  Total: {len(graphs)}")
    print(f"  Training (80%): {len(train_graphs)}")
    print(f"  Testing (20%): {len(test_graphs)}")
    
    return train_graphs, test_graphs


def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(out, batch.y)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * batch.num_graphs
    
    return total_loss / len(loader.dataset)


def test_epoch(model, loader, criterion, device):
    """Test for one epoch"""
    model.eval()
    total_loss = 0
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y)
            
            total_loss += loss.item() * batch.num_graphs
            
            predictions.extend(out.cpu().numpy())
            targets.extend(batch.y.cpu().numpy())
    
    # Calculate R²
    predictions = torch.tensor(predictions)
    targets = torch.tensor(targets)
    
    ss_res = torch.sum((targets - predictions) ** 2)
    ss_tot = torch.sum((targets - targets.mean()) ** 2)
    r2_score = 1 - (ss_res / ss_tot)
    
    return total_loss / len(loader.dataset), r2_score.item()


def main():
    print("="*60)
    print("🧪 SYNTHETIC DATA TRAINING - CONTROL EXPERIMENT")
    print("="*60)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🔧 Device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # Load data
    print(f"\n📁 Loading Synthetic Graphs...")
    train_graphs, test_graphs = load_synthetic_graphs()
    
    if not train_graphs:
        return
    
    # Create DataLoaders
    batch_size = 16
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)
    
    print(f"\n📦 DataLoaders:")
    print(f"  Batch Size: {batch_size}")
    print(f"  Train Batches: {len(train_loader)}")
    print(f"  Test Batches: {len(test_loader)}")
    
    # Initialize model
    print(f"\n🧠 Initializing Model...")
    model = TopoFlowGNN().to(device)
    print(f"  Parameters: {count_parameters(model):,}")
    
    # Setup training
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    criterion = nn.MSELoss()
    
    print(f"\n⚙️  Training Setup:")
    print(f"  Optimizer: Adam (lr=0.001, weight_decay=5e-4)")
    print(f"  Loss: MSE")
    print(f"  Epochs: 50")
    
    # Training loop
    print(f"\n{'='*60}")
    print("🏋️  TRAINING START (SYNTHETIC DATA)")
    print("="*60)
    
    best_test_loss = float('inf')
    best_epoch = 0
    
    for epoch in range(1, 51):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, r2 = test_epoch(model, test_loader, criterion, device)
        
        # Save best model
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_epoch = epoch
            torch.save(model.state_dict(), 'models/best_model_synthetic.pth')
        
        # Print progress
        if epoch == 1 or epoch % 5 == 0:
            print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | "
                  f"Test Loss: {test_loss:.4f} | R²: {r2:.4f} | "
                  f"Best: {best_test_loss:.4f} (Epoch {best_epoch})")
    
    # Final summary
    print(f"\n{'='*60}")
    print("✅ SYNTHETIC TRAINING COMPLETE")
    print("="*60)
    print(f"Best Epoch: {best_epoch}")
    print(f"Best Test Loss (MSE): {best_test_loss:.4f}")
    print(f"Model saved to: models/best_model_synthetic.pth")
    
    print(f"\n{'='*60}")
    print("📊 NEXT STEP: Compare vs Baseline")
    print("="*60)
    print("Run: python src/baseline_synthetic.py")
    print("Expected: GNN should BEAT Kozeny-Carman on synthetic data")
    print("="*60)


if __name__ == "__main__":
    main()
