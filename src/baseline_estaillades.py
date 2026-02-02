"""
Estaillades Baseline - The Final Verdict
Kozeny-Carman vs GNN on vuggy carbonate
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
    print("🏰 ESTAILLADES FINAL VERDICT - GNN vs Baseline")
    print("="*70)
    print("\nThe ultimate test on vuggy carbonate\n")
    
    # Load graphs
    graph_files = sorted(glob.glob('data/graphs_estaillades/*.pt'))
    print(f"📁 Found {len(graph_files)} Estaillades graphs")
    
    # Extract data
    porosities = []
    permeabilities = []
    
    print(f"\n📊 Processing samples...")
    for graph_file in graph_files:
        graph = torch.load(graph_file, weights_only=False)
        log_k = graph.y.item()
        k_true = 10 ** log_k
        
        # Estimate porosity from graph
        pore_volumes = 10 ** graph.x[:, 1].numpy()
        total_pore_vol = np.sum(pore_volumes)
        chunk_vol = (128 * 3.31e-6) ** 3
        phi = total_pore_vol / chunk_vol
        phi = np.clip(phi, 0.01, 0.50)
        
        porosities.append(phi)
        permeabilities.append(k_true)
    
    porosities = np.array(porosities)
    permeabilities = np.array(permeabilities)
    
    print(f"\n✅ Loaded {len(porosities)} samples")
    print(f"   Porosity range: {porosities.min():.3f} - {porosities.max():.3f}")
    print(f"   K range: {permeabilities.min():.2e} - {permeabilities.max():.2e} m²")
    
    # Split
    split_idx = int(len(porosities) * 0.8)
    phi_train, phi_test = porosities[:split_idx], porosities[split_idx:]
    k_train, k_test = permeabilities[:split_idx], permeabilities[split_idx:]
    
    # Fit
    print(f"\n🔧 Fitting Kozeny-Carman...")
    try:
        params, _ = curve_fit(kozeny_carman, phi_train, k_train, p0=[1e-12])
        C_opt = params[0]
    except:
        C_opt = 1e-12
    print(f"   Optimal C: {C_opt:.2e}")
    
    # Predict
    k_pred_test = kozeny_carman(phi_test, C_opt)
    
    # Metrics (log scale)
    mse_test = np.mean((np.log10(k_test + 1e-18) - np.log10(k_pred_test + 1e-18))**2)
    r2_test = r2_score(np.log10(k_test + 1e-18), np.log10(k_pred_test + 1e-18))
    
    # GNN result
    gnn_mse = 0.0802  # From training
    
    print(f"\n{'='*70}")
    print("🏆 THE FINAL VERDICT - ESTAILLADES")
    print("="*70)
    
    print(f"\nKozeny-Carman Baseline:")
    print(f"   Test MSE (log): {mse_test:.4f}")
    print(f"   Test R²: {r2_test:.4f}")
    
    print(f"\nGNN (Topology-based):")
    print(f"   Test MSE (log): {gnn_mse:.4f}")
    
    print(f"\n{'='*70}")
    
    if gnn_mse < mse_test:
        improvement = ((mse_test - gnn_mse) / mse_test) * 100
        print(f"🎉 ** GNN WINS! **")
        print(f"   GNN MSE: {gnn_mse:.4f}")
        print(f"   Baseline MSE: {mse_test:.4f}")
        print(f"   GNN better by {improvement:.1f}%")
        print(f"\n💡 BREAKTHROUGH: Topology DOES matter on vuggy carbonates!")
        print(f"   Complex pore connectivity overwhelms simple porosity!")
    else:
        factor = mse_test / gnn_mse if gnn_mse != 0 else 0
        print(f"❌ Baseline WINS")
        print(f"   Baseline MSE: {mse_test:.4f}")
        print(f"   GNN MSE: {gnn_mse:.4f}")
        if factor > 1:
            print(f"   Baseline better by {(1-1/factor)*100:.1f}%")
        print(f"\n💡 Conclusion: Even on vuggy carbonates, porosity dominates")
    
    print("="*70)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(k_test, k_pred_test, alpha=0.6, s=50)
    plt.plot([k_test.min(), k_test.max()], 
             [k_test.min(), k_test.max()], 
             'r--', linewidth=2)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('True K (m²)', fontsize=13, fontweight='bold')
    plt.ylabel('Predicted K (m²)', fontsize=13, fontweight='bold')
    plt.title(f'Estaillades: Kozeny-Carman (R²={r2_test:.3f})', 
              fontsize=15, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('estaillades_baseline.png', dpi=300)
    print(f"\n📈 Plot saved: estaillades_baseline.png")
    
    # Save results
    with open('estaillades_results.txt', 'w') as f:
        f.write("ESTAILLADES FINAL RESULTS\n")
        f.write("="*50 + "\n\n")
        f.write(f"Baseline MSE: {mse_test:.4f}\n")
        f.write(f"Baseline R²: {r2_test:.4f}\n")
        f.write(f"GNN MSE: {gnn_mse:.4f}\n\n")
        if gnn_mse < mse_test:
            f.write("WINNER: GNN\n")
        else:
            f.write("WINNER: Baseline\n")
    
    print(f"💾 Results saved: estaillades_results.txt\n")


if __name__ == "__main__":
    main()
