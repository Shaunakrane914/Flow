"""
Smart ILS Extractor - Auto-detect which label is pore
ILS uses multi-label segmentation (1, 2, 3)
"""

import numpy as np
import os
import h5py

def extract_ils_smart():
    """Automatically detect which label represents pore"""
    
    print("="*70)
    print("🧠 SMART ILS EXTRACTION (Auto-detect pore label)")
    print("="*70)
    
    mat_file = "data/raw/ILS_seg_hr.mat"
    output_dir = "data/ils_chunks"
    
    os.makedirs(output_dir, exist_ok=True)
    
    CHUNK_SIZE = 128
    NUM_SAMPLES = 30
    
    print(f"\n📂 Opening: {mat_file} (memory-mapped)")
    
    with h5py.File(mat_file, 'r') as f:
        volume = f['newl']
        shape = volume.shape
        print(f"  ✅ Volume shape: {shape}")
        
        max_x = shape[0] - CHUNK_SIZE
        max_y = shape[1] - CHUNK_SIZE
        max_z = shape[2] - CHUNK_SIZE
        
        print(f"\n🔍 Testing which label is PORE (checking 10 samples)...")
        
        # Test different labels
        label_stats = {1: [], 2: [], 3: []}
        
        for i in range(10):
            x = np.random.randint(0, max_x)
            y = np.random.randint(0, max_y)
            z = np.random.randint(0, max_z)
            
            chunk = np.array(volume[x:x+CHUNK_SIZE, y:y+CHUNK_SIZE, z:z+CHUNK_SIZE])
            
            # Count each label
            total = chunk.size
            for label in [1, 2, 3]:
                frac = np.sum(chunk == label) / total
                label_stats[label].append(frac)
        
        # Analyze
        print(f"\n  📊 Average fractions per label:")
        for label in [1, 2, 3]:
            avg = np.mean(label_stats[label])
            print(f"     Label {label}: {avg:.3f}")
        
        # Pore is typically the MINORITY phase (10-30%)
        avgs = {label: np.mean(label_stats[label]) for label in [1, 2, 3]}
        
        # Find label with fraction closest to 0.15-0.25 (typical porosity)
        target_phi = 0.20
        pore_label = min(avgs.keys(), key=lambda l: abs(avgs[l] - target_phi))
        
        pore_fraction = avgs[pore_label]
        
        print(f"\n  ✅ DETECTED: Label {pore_label} is PORE (fraction={pore_fraction:.3f})")
        
        # Set porosity filter range based on detected pore
        phi_min = max(0.05, pore_fraction - 0.10)
        phi_max = min(0.40, pore_fraction + 0.15)
        
        print(f"  🎯 Using porosity range: {phi_min:.3f} to {phi_max:.3f}")
        
        print(f"\n✂️  Extracting {NUM_SAMPLES} chunks...")
        
        saved = 0
        attempts = 0
        
        while saved < NUM_SAMPLES and attempts < NUM_SAMPLES * 10:
            x = np.random.randint(0, max_x)
            y = np.random.randint(0, max_y)
            z = np.random.randint(0, max_z)
            
            chunk = np.array(volume[x:x+CHUNK_SIZE, y:y+CHUNK_SIZE, z:z+CHUNK_SIZE])
            
            # Convert: pore_label=1, all others=0
            chunk_binary = (chunk == pore_label).astype(np.uint8)
            
            # Check porosity
            phi = np.sum(chunk_binary) / chunk_binary.size
            
            if phi_min < phi < phi_max:
                output_file = os.path.join(output_dir, f"ils_{saved:03d}.npy")
                np.save(output_file, chunk_binary)
                print(f"  ✅ Saved: ils_{saved:03d}.npy (φ={phi:.3f})")
                saved += 1
            
            attempts += 1
            
            if attempts % 20 == 0:
                print(f"  🔄 Progress: {saved}/{NUM_SAMPLES} saved, {attempts} attempts")
    
    print(f"\n{'='*70}")
    print(f"✅ EXTRACTION COMPLETE")
    print(f"{'='*70}")
    print(f"  Samples created: {saved}")
    print(f"  Total attempts: {attempts}")
    print(f"  Success rate: {saved/attempts*100:.1f}%")
    print(f"  Output: {output_dir}/")
    print("="*70)


if __name__ == "__main__":
    extract_ils_smart()
