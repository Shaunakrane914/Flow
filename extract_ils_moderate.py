"""
Resource-Limited ILS Chunk Extractor
Uses moderate CPU/RAM with debugging to find correct porosity range
"""

import numpy as np
import os
import h5py
import time

def extract_ils_moderate():
    """Extract with resource limits and debugging"""
    
    print("="*70)
    print("📦 RESOURCE-LIMITED ILS EXTRACTION (Medium CPU/RAM)")
    print("="*70)
    
    mat_file = "data/raw/ILS_seg_hr.mat"
    output_dir = "data/ils_chunks"
    
    os.makedirs(output_dir, exist_ok=True)
    
    CHUNK_SIZE = 128
    NUM_SAMPLES = 30
    BATCH_SIZE = 5  # Process in small batches
    DELAY = 0.5  # Delay between batches (reduce CPU load)
    
    print(f"\n⚙️  Settings:")
    print(f"  Chunk size: {CHUNK_SIZE}³")
    print(f"  Target samples: {NUM_SAMPLES}")
    print(f"  Batch size: {BATCH_SIZE} (moderate CPU)")
    print(f"  Delay: {DELAY}s between batches (moderate RAM)")
    
    print(f"\n📂 Opening: {mat_file}")
    
    with h5py.File(mat_file, 'r') as f:
        volume = f['newl']  # Memory-mapped
        shape = volume.shape
        print(f"  ✅ Volume shape: {shape}")
        print(f"  ✅ Memory-mapped (minimal RAM usage)")
        
        # Calculate valid ranges
        max_x = shape[0] - CHUNK_SIZE
        max_y = shape[1] - CHUNK_SIZE
        max_z = shape[2] - CHUNK_SIZE
        
        print(f"\n🔍 DEBUGGING MODE: Finding actual porosity range...")
        print("  (Checking first 20 random chunks to see porosity distribution)")
        
        # DEBUG: Check porosity range first
        debug_porosities = []
        for i in range(20):
            x = np.random.randint(0, max_x)
            y = np.random.randint(0, max_y)
            z = np.random.randint(0, max_z)
            
            chunk = np.array(volume[x:x+CHUNK_SIZE, y:y+CHUNK_SIZE, z:z+CHUNK_SIZE])
            
            # Try both methods
            phi_normal = np.sum(chunk > 0) / chunk.size
            phi_inverse = np.sum(chunk == 0) / chunk.size
            
            debug_porosities.append((phi_normal, phi_inverse))
            
            if i < 5:  # Print first 5
                print(f"    Sample {i+1}: φ(>0)={phi_normal:.3f}, φ(==0)={phi_inverse:.3f}")
        
        # Analyze
        normal_avg = np.mean([p[0] for p in debug_porosities])
        inverse_avg = np.mean([p[1] for p in debug_porosities])
        
        print(f"\n  📊 Average porosities:")
        print(f"     Method 1 (>0 = pore): {normal_avg:.3f}")
        print(f"     Method 2 (==0 = pore): {inverse_avg:.3f}")
        
        # Choose method with reasonable porosity (0.1-0.3 typical)
        if 0.05 < normal_avg < 0.40:
            use_inverse = False
            print(f"  ✅ Using METHOD 1 (>0 = pore)")
        else:
            use_inverse = True
            print(f"  ✅ Using METHOD 2 (==0 = pore)")
        
        # Auto-detect porosity range
        all_phis = [p[1] if use_inverse else p[0] for p in debug_porosities]
        min_phi = min(all_phis)
        max_phi = max(all_phis)
        
        # Use slightly wider range
        phi_min = max(0.01, min_phi - 0.05)
        phi_max = min(0.99, max_phi + 0.05)
        
        print(f"  📏 Detected porosity range: {min_phi:.3f} to {max_phi:.3f}")
        print(f"  🎯 Using filter range: {phi_min:.3f} to {phi_max:.3f}")
        
        print(f"\n✂️  Extracting {NUM_SAMPLES} chunks...")
        
        saved = 0
        attempts = 0
        batch_count = 0
        
        while saved < NUM_SAMPLES and attempts < NUM_SAMPLES * 10:
            # Batch processing
            for _ in range(BATCH_SIZE):
                if saved >= NUM_SAMPLES:
                    break
                
                # Random position
                x = np.random.randint(0, max_x)
                y = np.random.randint(0, max_y)
                z = np.random.randint(0, max_z)
                
                # Read chunk
                chunk = np.array(volume[x:x+CHUNK_SIZE, y:y+CHUNK_SIZE, z:z+CHUNK_SIZE])
                
                # Convert to binary
                if use_inverse:
                    chunk = (chunk == 0).astype(np.uint8)
                else:
                    chunk = (chunk > 0).astype(np.uint8)
                
                # Check porosity
                phi = np.sum(chunk) / chunk.size
                
                if phi_min < phi < phi_max:
                    output_file = os.path.join(output_dir, f"ils_{saved:03d}.npy")
                    np.save(output_file, chunk)
                    print(f"  ✅ Saved: ils_{saved:03d}.npy (φ={phi:.3f})")
                    saved += 1
                
                attempts += 1
            
            # Batch completed - pause to reduce resource usage
            batch_count += 1
            if saved < NUM_SAMPLES:
                time.sleep(DELAY)
                
                if batch_count % 5 == 0:
                    print(f"  🔄 Progress: {saved}/{NUM_SAMPLES} saved, {attempts} attempts")
    
    print(f"\n{'='*70}")
    print(f"✅ EXTRACTION COMPLETE")
    print(f"{'='*70}")
    print(f"  Samples created: {saved}")
    print(f"  Total attempts: {attempts}")
    print(f"  Output directory: {output_dir}/")
    print(f"  Success rate: {saved/attempts*100:.1f}%")
    print("="*70)


if __name__ == "__main__":
    extract_ils_moderate()
