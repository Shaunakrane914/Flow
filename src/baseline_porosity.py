"""
Porosity Baseline using Kozeny-Carman Equation
Simplest possible baseline: Predicts permeability from porosity alone

For publication comparison:
- GNN learns from pore network topology
- This baseline uses only bulk porosity
- Shows whether topology matters
"""

import numpy as np
import torch
import glob
import os
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

def kozeny_carman(phi, C):
    """
    Kozeny-Carman equation: K = C * phi^3 / (1-phi)^2
    
    Parameters:
    -----------
    phi : float or array
        Porosity (0 to 1)
    C : float
        Kozeny-Carman constant (to be fitted)
    
    Returns:
    --------
    K : Permeability (m²)
    """
    # Prevent division by zero
    phi = np.clip(phi, 0.01, 0.99)
    return C * (phi**3) / ((1 - phi)**2)


def load_data():
    """Load graph labels and calculate porosities from raw chunks"""
    print("="*70)
    print("POROSITY BASELINE (Kozeny-Carman)")
    print("="*70)
    
    # Load all graph files
    graph_files = sorted(glob.glob('data/graphs/*.pt'))
    
    if not graph_files:
        print("❌ No graph files found in data/graphs/")
        return None, None
    
    print(f"\n📁 Found {len(graph_files)} graph files")
    
    true_k = []
    porosities = []
    
    print(f"\n📊 Processing chunks...")
    
    for i, graph_file in enumerate(graph_files):
        # Load graph to get true permeability
        graph = torch.load(graph_file, weights_only=False)
        
        # Extract true K (stored as log10, need to convert back)
        log_k = graph.y.item()
        k_true = 10 ** log_k
        
        # Find corresponding chunk file
        graph_name = os.path.basename(graph_file).replace('.pt', '')
        
        # Try both data/raw and data/processed
        chunk_file = None
        for data_dir in ['data/raw', 'data/processed']:
            potential_file = os.path.join(data_dir, f"{graph_name}.npy")
            if os.path.exists(potential_file):
                chunk_file = potential_file
                break
        
        if chunk_file is None:
            print(f"  ⚠️  Chunk not found for {graph_name}")
            continue
        
        # Load chunk and calculate porosity
        chunk = np.load(chunk_file)
        porosity = np.sum(chunk) / chunk.size
        
        true_k.append(k_true)
        porosities.append(porosity)
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(graph_files)} chunks...")
    
    true_k = np.array(true_k)
    porosities = np.array(porosities)
    
    print(f"\n✅ Loaded {len(true_k)} samples")
    print(f"   Porosity range: {porosities.min():.3f} - {porosities.max():.3f}")
    print(f"   K range: {true_k.min():.2e} - {true_k.max():.2e} m²")
    
    return porosities, true_k


def fit_kozeny_carman(porosities, true_k):
    """Fit Kozeny-Carman constant to data"""
    print(f"\n🔧 Fitting Kozeny-Carman equation...")
    
    # Fit C parameter
    popt, pcov = curve_fit(kozeny_carman, porosities, true_k, p0=[1e-10])
    C_opt = popt[0]
    
    print(f"   Optimal C: {C_opt:.2e}")
    
    # Predict using fitted equation
    k_pred = kozeny_carman(porosities, C_opt)
    
    return k_pred, C_opt


def evaluate(true_k, pred_k, name="Model"):
    """Calculate metrics"""
    # MSE on log scale (same as GNN training)
    log_true = np.log10(true_k + 1e-18)
    log_pred = np.log10(pred_k + 1e-18)
    
    mse_log = mean_squared_error(log_true, log_pred)
    r2 = r2_score(log_true, log_pred)
    
    # MSE on linear scale
    mse_linear = mean_squared_error(true_k, pred_k)
    
    return {
        'mse_log': mse_log,
        'mse_linear': mse_linear,
        'r2': r2
    }


def plot_comparison(true_k, pred_k, output_file='porosity_comparison.png'):
    """Create log-log scatter plot"""
    print(f"\n📈 Creating comparison plot...")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Log-log scatter
    ax.scatter(true_k, pred_k, alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
    
    # Perfect prediction line
    min_k = min(true_k.min(), pred_k.min())
    max_k = max(true_k.max(), pred_k.max())
    ax.plot([min_k, max_k], [min_k, max_k], 'r--', linewidth=2, label='Perfect Prediction')
    
    # Log scale
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Labels
    ax.set_xlabel('True Permeability (m²)', fontsize=14)
    ax.set_ylabel('Kozeny-Carman Prediction (m²)', fontsize=14)
    ax.set_title('Porosity Baseline: Kozeny-Carman vs Ground Truth', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved to {output_file}")


def main():
    # Load data
    porosities, true_k = load_data()
    
    if porosities is None:
        return
    
    # Same 80/20 split as GNN
    n_train = int(0.8 * len(porosities))
    
    train_phi = porosities[:n_train]
    train_k = true_k[:n_train]
    
    test_phi = porosities[n_train:]
    test_k = true_k[n_train:]
    
    print(f"\n📊 Data split:")
    print(f"   Training: {len(train_phi)} samples")
    print(f"   Testing: {len(test_phi)} samples")
    
    # Fit on training data
    train_pred, C_opt = fit_kozeny_carman(train_phi, train_k)
    
    # Predict on test data
    test_pred = kozeny_carman(test_phi, C_opt)
    
    # Evaluate
    train_metrics = evaluate(train_k, train_pred, "Train")
    test_metrics = evaluate(test_k, test_pred, "Test")
    
    # Print results
    print(f"\n{'='*70}")
    print("📊 RESULTS")
    print("="*70)
    
    print(f"\nTraining Set:")
    print(f"   MSE (log scale): {train_metrics['mse_log']:.4f}")
    print(f"   R² Score: {train_metrics['r2']:.4f}")
    
    print(f"\nTest Set:")
    print(f"   MSE (log scale): {test_metrics['mse_log']:.4f}")
    print(f"   R² Score: {test_metrics['r2']:.4f}")
    
    # Comparison with GNN
    print(f"\n{'='*70}")
    print("🔬 COMPARISON")
    print("="*70)
    print(f"Porosity Baseline MSE:  {test_metrics['mse_log']:.4f}")
    print(f"GNN (Graph) MSE:        0.2763 (from training)")
    
    if test_metrics['mse_log'] > 0.2763:
        improvement = ((test_metrics['mse_log'] - 0.2763) / test_metrics['mse_log']) * 100
        print(f"\n✅ GNN is BETTER by {improvement:.1f}%")
        print("   → Graph topology provides significant value!")
        print("   → Permeability depends on pore connectivity, not just bulk porosity")
    elif test_metrics['mse_log'] < 0.2763:
        decline = ((0.2763 - test_metrics['mse_log']) / 0.2763) * 100
        print(f"\n⚠️  Porosity baseline is better by {decline:.1f}%")
        print("   → GNN may be overfitting or needs tuning")
    else:
        print(f"\n➖ Models perform equally")
    
    print("="*70)
    
    # Plot
    plot_comparison(test_k, test_pred)
    
    # Save results to file
    with open('porosity_baseline_results.txt', 'w') as f:
        f.write("POROSITY BASELINE RESULTS\n")
        f.write("="*50 + "\n\n")
        f.write(f"Kozeny-Carman constant: {C_opt:.4e}\n")
        f.write(f"Test MSE (log): {test_metrics['mse_log']:.4f}\n")
        f.write(f"Test R²: {test_metrics['r2']:.4f}\n")
        f.write(f"\nComparison:\n")
        f.write(f"  Porosity Baseline: {test_metrics['mse_log']:.4f}\n")
        f.write(f"  GNN: 0.2763\n")
    
    print(f"\n💾 Results saved to porosity_baseline_results.txt")


if __name__ == "__main__":
    main()
