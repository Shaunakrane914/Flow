"""
Synthetic Data Generator for Control Experiment

Creates blob-based synthetic rocks where TOPOLOGY matters (not just porosity)
This will prove the GNN works when pore connectivity is important
"""

import numpy as np
import porespy as ps
import os
from tqdm import tqdm

def generate_synthetic_data(n_samples=200, output_dir="data/synthetic_raw"):
    """
    Generate synthetic sandstone-like samples using porespy blobs
    
    Key: Varying blobiness creates different topologies at same porosity
    This forces models to learn connectivity, not just bulk properties
    """
    print("="*70)
    print("🧪 SYNTHETIC DATA GENERATOR - CONTROL EXPERIMENT")
    print("="*70)
    print(f"\nGoal: Generate {n_samples} synthetic rocks where topology matters")
    print("Strategy: Vary porosity AND blobiness independently")
    print("Expected: GNN will beat Kozeny-Carman on these samples")
    
    os.makedirs(output_dir, exist_ok=True)
    
    generated = 0
    attempts = 0
    max_attempts = n_samples * 3  # Allow some retries
    
    print(f"\n{'='*70}")
    print("🚀 GENERATING SAMPLES")
    print("="*70)
    
    with tqdm(total=n_samples, desc="Generating", unit="rock") as pbar:
        while generated < n_samples and attempts < max_attempts:
            attempts += 1
            
            # Randomize parameters to create diverse topology
            shape = (128, 128, 128)  # Same size as MEC chunks
            porosity_target = np.random.uniform(0.15, 0.35)
            blobiness = np.random.uniform(0.8, 2.0)
            
            try:
                # Generate blob structure
                im = ps.generators.blobs(
                    shape=shape,
                    porosity=porosity_target,
                    blobiness=blobiness
                )
                
                # Quick quality check
                actual_porosity = np.sum(im) / im.size
                
                # Accept if porosity is reasonable (avoid extremes)
                if 0.1 < actual_porosity < 0.4:
                    filename = f"synthetic_{generated:03d}.npy"
                    filepath = os.path.join(output_dir, filename)
                    np.save(filepath, im)
                    
                    generated += 1
                    pbar.update(1)
                    pbar.set_postfix({
                        'Porosity': f'{actual_porosity:.3f}',
                        'Blob': f'{blobiness:.2f}'
                    })
                
            except Exception as e:
                continue
    
    print(f"\n{'='*70}")
    print("📊 GENERATION SUMMARY")
    print("="*70)
    print(f"Successful: {generated} samples")
    print(f"Attempts: {attempts}")
    print(f"Success rate: {generated/attempts*100:.1f}%")
    print(f"\n💾 Saved to: {output_dir}/")
    
    # Statistics
    print(f"\n{'='*70}")
    print("🔬 DATASET CHARACTERISTICS")
    print("="*70)
    print("These synthetic rocks have:")
    print("  ✓ Variable porosity (0.15-0.35)")
    print("  ✓ Variable topology (blobiness 0.8-2.0)")
    print("  ✓ Complex pore connectivity")
    print("  ✓ Topology-dependent permeability")
    print("\n⚡ Expected: GNN > Kozeny-Carman on this dataset")
    print("="*70)
    
    return generated


if __name__ == "__main__":
    n_generated = generate_synthetic_data(n_samples=200)
    
    if n_generated >= 150:
        print(f"\n✅ SUCCESS: Generated {n_generated} samples")
        print("\nNext steps:")
        print("1. Run: python src/extract_nuclear.py (point to data/synthetic_raw)")
        print("2. Run: python src/train.py")
        print("3. Run: python src/baseline_porosity.py")
        print("4. Compare: GNN vs Baseline on topology-driven data")
    else:
        print(f"\n⚠️  WARNING: Only generated {n_generated} samples")
        print("May need to adjust parameters")
