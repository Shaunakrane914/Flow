"""
Porosity Baseline for Synthetic Data
This will show if Kozeny-Carman works on topology-driven data
"""

import numpy as np
import torch
import glob
import os
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def kozeny_carman(phi, C):
    """Kozeny-Carman equation"""
    phi = np.clip(phi, 0.01, 0.99)
    return C * (phi**3) / ((1 - phi)**2)


def main():
    print("="*70)
    print("📊 SYNTHETIC BASELINE - Kozeny-Carman vs GNN")
    print("="*70)
    
    # Load graphs
    graph_files = sorted(glob.glob('data/graphs_synthetic/*.pt'))
    print(f"\n📁 Found {len(graph_files)} synthetic graphs")
    
    # Extract labels and calculate porosity
    porosities = []
    permeabilities = []
    
    print(f"\n📊 Processing samples...")
    for graph_file in graph_files:
        # Load graph for true K
        graph = torch.load(graph_file, weights_only=False)
        log_k = graph.y.item()
        k_true = 10 ** log_k
        
        # Load corresponding raw chunk for porosity
        source_id = os.path.basename(graph_file).replace('.pt', '')
        chunk_file = f"data/synthetic_raw/{source_id}.npy"
        
        if os.path.exists(chunk_file):
            chunk = np.load(chunk_file)
            phi = np.sum(chunk) / chunk.size
            
            porosities.append(phi)
            permeabilities.append(k_true)
    
    porosities = np.array(porosities)
    permeabilities = np.array(permeabilities)
    
    print(f"\n✅ Loaded {len(porosities)} samples")
    print(f"   Porosity range: {porosities.min():.3f} - {porosities.max():.3f}")
    print(f"   K range: {permeabilities.min():.2e} - {permeabilities.max():.2e} m²")
    
    # 80/20 split
    split_idx = int(len(porosities) * 0.8)
    phi_train, phi_test = porosities[:split_idx], porosities[split_idx:]
    k_train, k_test = permeabilities[:split_idx], permeabilities[split_idx:]
    
    print(f"\n📊 Data split:")
    print(f"   Training: {len(phi_train)} samples")
    print(f"   Testing: {len(phi_test)} samples")
    
    # Fit Kozeny-Carman
    print(f"\n🔧 Fitting Kozeny-Carman equation...")
    params, _ = curve_fit(kozeny_carman, phi_train, k_train, p0=[1e-12])
    C_opt = params[0]
    print(f"   Optimal C: {C_opt:.2e}")
    
    # Predict
    k_pred_train = kozeny_carman(phi_train, C_opt)
    k_pred_test = kozeny_carman(phi_test, C_opt)
    
    # Calculate metrics (log scale like GNN)
    mse_train = np.mean((np.log10(k_train + 1e-18) - np.log10(k_pred_train + 1e-18))**2)
    mse_test = np.mean((np.log10(k_test + 1e-18) - np.log10(k_pred_test + 1e-18))**2)
    
    r2_train = r2_score(np.log10(k_train + 1e-18), np.log10(k_pred_train + 1e-18))
    r2_test = r2_score(np.log10(k_test + 1e-18), np.log10(k_pred_test + 1e-18))
    
    # Results
    print(f"\n{'='*70}")
    print("📊 SYNTHETIC DATA RESULTS")
    print("="*70)
    
    print(f"\nKozeny-Carman Baseline:")
    print(f"   Training MSE (log): {mse_train:.4f}")
    print(f"   Testing MSE (log): {mse_test:.4f}")
    print(f"   Testing R²: {r2_test:.4f}")
    
    print(f"\nGNN (from training):")
    print(f"   Testing MSE (log): 1.2666")
    
    print(f"\n{'='*70}")
    print("🎯 COMPARISON")
    print("="*70)
    
    if mse_test < 1.2666:
        improvement = ((1.2666 - mse_test) / 1.2666) * 100
        print(f"❌ Kozeny-Carman WINS")
        print(f"   Baseline MSE: {mse_test:.4f}")
        print(f"   GNN MSE: 1.2666")
        print(f"   Baseline better by {improvement:.1f}%")
        print(f"\n💡 Interpretation: Even on synthetic data with variable")
        print(f"   topology, porosity dominates permeability prediction.")
    else:
        improvement = ((mse_test - 1.2666) / mse_test) * 100
        print(f"✅ GNN WINS!")
        print(f"   GNN MSE: 1.2666")
        print(f"   Baseline MSE: {mse_test:.4f}")
        print(f"   GNN better by {improvement:.1f}%")
        print(f"\n💡 Success! GNN learned topology patterns that")
        print(f"   Kozeny-Carman cannot capture!")
    
    print("="*70)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(k_test, k_pred_test, alpha=0.6, label='Kozeny-Carman')
    plt.plot([k_test.min(), k_test.max()], 
             [k_test.min(), k_test.max()], 
             'r--', label='Perfect Prediction')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('True K (m²)', fontsize=12)
    plt.ylabel('Predicted K (m²)', fontsize=12)
    plt.title(f'Synthetic Data: Kozeny-Carman Baseline (R²={r2_test:.3f})', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('synthetic_baseline_comparison.png', dpi=300)
    print(f"\n📈 Plot saved to: synthetic_baseline_comparison.png")


if __name__ == "__main__":
    main()
