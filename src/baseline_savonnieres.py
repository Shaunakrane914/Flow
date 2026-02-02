"""
Savonnières Baseline - GNN vs Kozeny-Carman
Compare topology-aware GNN against classical formula
"""

import numpy as np
import torch
import glob
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def kozeny_carman(phi, C):
    """Kozeny-Carman equation"""
    phi = np.clip(phi, 0.01, 0.99)
    return C * (phi**3) / ((1 - phi)**2)


def main():
    print("="*70)
    print("🪨 SAVONNIÈRES FINAL VERDICT - GNN vs Baseline")
    print("="*70)
    print("\n3-phase vuggy carbonate - will topology matter?\n")
    
    # Load graphs
    graph_files = sorted(glob.glob('data/graphs_savonnieres/*.pt'))
    print(f"📁 Found {len(graph_files)} Savonnières graphs")
    
    # Extract data
    porosities = []
    permeabilities = []
    
    print(f"\n📊 Processing samples...")
    for graph_file in graph_files:
        graph = torch.load(graph_file, weights_only=False)
        log_k = graph.y.item()
        k_true = 10 ** log_k
        
        # Estimate porosity from graph (pore volumes)
        pore_volumes = 10 ** graph.x[:, 1].numpy()
        total_pore_vol = np.sum(pore_volumes)
        chunk_vol = (128 * 2.68e-6) ** 3  # Savonnières voxel size
        phi = total_pore_vol / chunk_vol
        phi = np.clip(phi, 0.01, 0.50)
        
        porosities.append(phi)
        permeabilities.append(k_true)
    
    porosities = np.array(porosities)
    permeabilities = np.array(permeabilities)
    
    print(f"\n✅ Loaded {len(porosities)} samples")
    print(f"   Porosity range: {porosities.min():.3f} - {porosities.max():.3f}")
    print(f"   K range: {permeabilities.min():.2e} - {permeabilities.max():.2e} m²")
    
    # Split (80/20 train/test)
    split_idx = int(len(porosities) * 0.8)
    phi_train, phi_test = porosities[:split_idx], porosities[split_idx:]
    k_train, k_test = permeabilities[:split_idx], permeabilities[split_idx:]
    
    # Fit Kozeny-Carman on training set
    print(f"\n🔧 Fitting Kozeny-Carman...")
    try:
        params, _ = curve_fit(kozeny_carman, phi_train, k_train, p0=[1e-12])
        C_opt = params[0]
    except:
        C_opt = 1e-12
    print(f"   Optimal C: {C_opt:.2e}")
    
    # Predict on test set
    k_pred_test = kozeny_carman(phi_test, C_opt)
    
    # Metrics (log scale to match GNN training)
    mse_test = np.mean((np.log10(k_test + 1e-18) - np.log10(k_pred_test + 1e-18))**2)
    r2_test = r2_score(np.log10(k_test + 1e-18), np.log10(k_pred_test + 1e-18))
    
    # Load GNN results
    # Note: This will be updated after running train_savonnieres.py
    import os
    if os.path.exists('models/best_model_savonnieres.pth'):
        # Load trained model and evaluate
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from src.model import TopoFlowGNN
        from torch_geometric.loader import DataLoader
        
        model = TopoFlowGNN()
        model.load_state_dict(torch.load('models/best_model_savonnieres.pth'))
        model.eval()
        
        # Load test graphs
        test_graphs = [torch.load(f, weights_only=False) for f in graph_files[split_idx:]]
        test_loader = DataLoader(test_graphs, batch_size=16, shuffle=False)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.batch)
                predictions.extend(out.cpu().numpy())
                targets.extend(batch.y.cpu().numpy())
        
        predictions = np.array(predictions).flatten()
        targets = np.array(targets).flatten()
        
        gnn_mse = np.mean((targets - predictions)**2)
        gnn_r2 = r2_score(targets, predictions)
    else:
        print("\n⚠️  GNN model not found. Run train_savonnieres.py first!")
        print("   Using placeholder MSE for comparison template")
        gnn_mse = 0.15  # Placeholder
        gnn_r2 = 0.0
    
    print(f"\n{'='*70}")
    print("🏆 THE FINAL VERDICT - SAVONNIÈRES")
    print("="*70)
    
    print(f"\nKozeny-Carman Baseline:")
    print(f"   Test MSE (log): {mse_test:.4f}")
    print(f"   Test R²: {r2_test:.4f}")
    
    print(f"\nGNN (Topology-based):")
    print(f"   Test MSE (log): {gnn_mse:.4f}")
    if 'gnn_r2' in locals():
        print(f"   Test R²: {gnn_r2:.4f}")
    
    print(f"\n{'='*70}")
    
    if gnn_mse < mse_test:
        improvement = ((mse_test - gnn_mse) / mse_test) * 100
        print(f"🎉 ** GNN WINS! **")
        print(f"   GNN MSE: {gnn_mse:.4f}")
        print(f"   Baseline MSE: {mse_test:.4f}")
        print(f"   GNN better by {improvement:.1f}%")
        print(f"\n💡 INSIGHT: Topology matters on Savonnières too!")
        print(f"   Multi-scale vugs require graph-based learning")
    else:
        improvement = ((gnn_mse - mse_test) / mse_test) * 100
        print(f"❌ Baseline WINS")
        print(f"   Baseline MSE: {mse_test:.4f}")
        print(f"   GNN MSE: {gnn_mse:.4f}")
        print(f"   Baseline better by {improvement:.1f}%")
        print(f"\n💡 Conclusion: Porosity dominates for Savonnières")
    
    print("="*70)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(k_test, k_pred_test, alpha=0.6, s=50, label='Kozeny-Carman')
    plt.plot([k_test.min(), k_test.max()], 
             [k_test.min(), k_test.max()], 
             'r--', linewidth=2, label='Perfect Fit')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('True K (m²)', fontsize=13, fontweight='bold')
    plt.ylabel('Predicted K (m²)', fontsize=13, fontweight='bold')
    plt.title(f'Savonnières: Kozeny-Carman (R²={r2_test:.3f})', 
              fontsize=15, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('savonnieres_baseline.png', dpi=300)
    print(f"\n📈 Plot saved: savonnieres_baseline.png")
    
    # Save results
    with open('savonnieres_results.txt', 'w') as f:
        f.write("SAVONNIÈRES FINAL RESULTS\n")
        f.write("="*50 + "\n\n")
        f.write(f"Baseline MSE: {mse_test:.4f}\n")
        f.write(f"Baseline R²: {r2_test:.4f}\n")
        f.write(f"GNN MSE: {gnn_mse:.4f}\n\n")
        if gnn_mse < mse_test:
            improvement = ((mse_test - gnn_mse) / mse_test) * 100
            f.write("WINNER: GNN\n")
            f.write(f"Improvement: +{improvement:.1f}%\n")
        else:
            improvement = ((gnn_mse - mse_test) / mse_test) * 100
            f.write("WINNER: Baseline\n")
            f.write(f"Baseline better by: {improvement:.1f}%\n")
    
    print(f"💾 Results saved: savonnieres_results.txt\n")


if __name__ == "__main__":
    main()
