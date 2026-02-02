"""
Memory-Efficient ILS Chunk Extractor
Reads chunks directly from file without loading entire volume
"""

import numpy as np
import os
import h5py

def extract_ils_chunks_efficient():
    """Extract chunks with minimal memory footprint"""
    
    print("="*70)
    print("📦 MEMORY-EFFICIENT ILS EXTRACTION")
    print("="*70)
    
    mat_file = "data/raw/ILS_seg_hr.mat"
    output_dir = "data/ils_chunks"
    
    os.makedirs(output_dir, exist_ok=True)
    
    CHUNK_SIZE = 128
    NUM_SAMPLES = 30
    
    print(f"\n📂 Opening: {mat_file}")
    
    with h5py.File(mat_file, 'r') as f:
        volume = f['newl']  # Don't load - just reference!
        shape = volume.shape
        print(f"  ✅ Volume shape: {shape}")
        print(f"  ✅ Memory-mapped (not loaded into RAM)")
        
        # Calculate valid ranges
        max_x = shape[0] - CHUNK_SIZE
        max_y = shape[1] - CHUNK_SIZE
        max_z = shape[2] - CHUNK_SIZE
        
        print(f"\n✂️  Extracting {NUM_SAMPLES} chunks (reading directly from file)...")
        
        saved = 0
        attempts = 0
        
        while saved < NUM_SAMPLES and attempts < NUM_SAMPLES * 5:
            # Random position
            x = np.random.randint(0, max_x)
            y = np.random.randint(0, max_y)
            z = np.random.randint(0, max_z)
            
            # Read ONLY this chunk from file (memory efficient!)
            chunk = volume[x:x+CHUNK_SIZE, y:y+CHUNK_SIZE, z:z+CHUNK_SIZE]
            chunk = np.array(chunk)  # Load only this small chunk
            
            # Convert to binary - TRY INVERSE (0=pore, >0=solid)
            chunk = (chunk == 0).astype(np.uint8)  # Inverse!
            
            # Check porosity
            phi = np.sum(chunk) / chunk.size
            
            if 0.05 < phi < 0.40:
                output_file = os.path.join(output_dir, f"ils_{saved:03d}.npy")
                np.save(output_file, chunk)
                print(f"  ✅ Saved: ils_{saved:03d}.npy (φ={phi:.3f})")
                saved += 1
            
            attempts += 1
            
            if attempts % 10 == 0:
                print(f"  🔄 Progress: {saved}/{NUM_SAMPLES} saved, {attempts} attempts")
    
    print(f"\n{'='*70}")
    print(f"✅ EXTRACTION COMPLETE")
    print(f"{'='*70}")
    print(f"  Samples created: {saved}")
    print(f"  Output directory: {output_dir}/")
    print("="*70)


if __name__ == "__main__":
    extract_ils_chunks_efficient()
