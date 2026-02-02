"""
Porosity Baseline for ILS (Indiana Limestone)
Compare Kozeny-Carman vs GNN on well-connected rock
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
    print("🪨 ILS BASELINE - Kozeny-Carman vs GNN (FINAL TEST)")
    print("="*70)
    print("\nThis is the critical comparison for well-connected rocks")
    
    # Load graphs
    graph_files = sorted(glob.glob('data/graphs_ils/*.pt'))
    print(f"\n📁 Found {len(graph_files)} ILS graphs")
    
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
        # Note: ILS chunks are in HDF5, we'll estimate from graph
        # Use number of pore nodes as proxy for porosity
        num_pores = graph.num_nodes
        
        # Estimate porosity from pore volume in graph
        # This is approximate but should correlate with true porosity
        pore_volumes = 10 ** graph.x[:, 1].numpy()  # Un-log the volume feature
        total_pore_vol = np.sum(pore_volumes)
        chunk_vol = (128 * 1e-6) ** 3  # 128³ voxels
        phi = total_pore_vol / chunk_vol
        phi = np.clip(phi, 0.01, 0.50)  # Reasonable bounds
        
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
    try:
        params, _ = curve_fit(kozeny_carman, phi_train, k_train, p0=[1e-12])
        C_opt = params[0]
        print(f"   Optimal C: {C_opt:.2e}")
    except:
        C_opt = 1e-12
        print(f"   Using default C: {C_opt:.2e}")
    
    # Predict
    k_pred_train = kozeny_carman(phi_train, C_opt)
    k_pred_test = kozeny_carman(phi_test, C_opt)
    
    # Calculate metrics (log scale like GNN)
    mse_train = np.mean((np.log10(k_train + 1e-18) - np.log10(k_pred_train + 1e-18))**2)
    mse_test = np.mean((np.log10(k_test + 1e-18) - np.log10(k_pred_test + 1e-18))**2)
    
    r2_train = r2_score(np.log10(k_train + 1e-18), np.log10(k_pred_train + 1e-18))
    r2_test = r2_score(np.log10(k_test + 1e-18), np.log10(k_pred_test + 1e-18))
    
    # Get GNN MSE from train_ils.py results
    print("\nNote: Run python src/train_ils.py first to get GNN performance")
    
    # Results
    print(f"\n{'='*70}")
    print("📊 ILS RESULTS - THE VERDICT")
    print("="*70)
    
    print(f"\nKozeny-Carman Baseline:")
    print(f"   Training MSE (log): {mse_train:.4f}")
    print(f"   Testing MSE (log): {mse_test:.4f}")
    print(f"   Testing R²: {r2_test:.4f}")
    
    print(f"\nGNN:")
    print(f"   Run train_ils.py to get GNN MSE")
    
    print(f"\n{'='*70}")
    print("🎯 CRITICAL COMPARISON")
    print("="*70)
    print(f"Baseline MSE: {mse_test:.4f}")
    print(f"Check train_ils.py output for GNN MSE")
    print(f"\nIf GNN < {mse_test:.4f}: ✅ TOPOLOGY MATTERS!")
    print(f"If GNN > {mse_test:.4f}: ❌ Porosity dominates (as expected)")
    print("="*70)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(k_test, k_pred_test, alpha=0.6, label='Kozeny-Carman', s=50)
    plt.plot([k_test.min(), k_test.max()], 
             [k_test.min(), k_test.max()], 
             'r--', label='Perfect Prediction', linewidth=2)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('True K (m²)', fontsize=13, fontweight='bold')
    plt.ylabel('Predicted K (m²)', fontsize=13, fontweight='bold')
    plt.title(f'ILS: Kozeny-Carman Baseline (R²={r2_test:.3f})', fontsize=15, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ils_baseline_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\n📈 Plot saved to: ils_baseline_comparison.png")
    
    # Save results
    with open('ils_baseline_results.txt', 'w') as f:
        f.write("ILS BASELINE RESULTS\n")
        f.write("="*50 + "\n\n")
        f.write(f"Samples: {len(porosities)}\n")
        f.write(f"Training: {len(phi_train)}\n")
        f.write(f"Testing: {len(phi_test)}\n\n")
        f.write(f"Kozeny-Carman C: {C_opt:.2e}\n\n")
        f.write(f"Training MSE: {mse_train:.4f}\n")
        f.write(f"Testing MSE: {mse_test:.4f}\n")
        f.write(f"Testing R²: {r2_test:.4f}\n")
    
    print(f"💾 Results saved to: ils_baseline_results.txt\n")


if __name__ == "__main__":
    main()
