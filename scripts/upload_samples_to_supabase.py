"""
Upload sample rock chunks to Supabase Storage
Run once to populate the 'rock-samples' bucket with representative .npy files
"""

import numpy as np
import os
import sys
import glob

# Try to import supabase
try:
    from supabase import create_client
except ImportError:
    print("Install supabase: pip install supabase")
    sys.exit(1)


# ═══════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════

# Load from .env or environment
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")
BUCKET_NAME = "rock-samples"


def find_sample_by_porosity(chunks_dir, pattern, target_phi, tolerance=0.05):
    """Find a chunk closest to target porosity"""
    files = sorted(glob.glob(os.path.join(chunks_dir, pattern)))
    
    best_file = None
    best_diff = float('inf')
    
    for f in files:
        try:
            chunk = np.load(f)
            phi = np.mean(chunk > 0)
            diff = abs(phi - target_phi)
            if diff < best_diff:
                best_diff = diff
                best_file = f
                best_phi = phi
        except Exception:
            continue
    
    if best_file:
        print(f"  Found: {os.path.basename(best_file)} (φ={best_phi:.3f}, target={target_phi:.3f})")
    return best_file


def prepare_savonnieres_chunks():
    """Extract sample chunks from Savonnières .dat file"""
    dat_path = "data/SAVII2_mid_1000x1000x1000x8b-3phase-cleaner.dat"
    
    if not os.path.exists(dat_path):
        print("  ⚠️ Savonnières .dat not found, skipping")
        return None, None
    
    print("  Loading Savonnières (1GB)...")
    full_rock = np.fromfile(dat_path, dtype=np.uint8).reshape((1000, 1000, 1000))
    binary = (full_rock == 0).astype(np.uint8)  # Phase 0 = pore
    
    # Sample A: region with vugs (high porosity)
    chunk_a = binary[0:128, 0:128, 0:128].copy()
    # Sample B: tighter region
    chunk_b = binary[256:384, 256:384, 256:384].copy()
    
    print(f"  Sample A: φ={np.mean(chunk_a):.3f}")
    print(f"  Sample B: φ={np.mean(chunk_b):.3f}")
    
    return chunk_a, chunk_b


def prepare_estaillades_chunks():
    """Get sample chunks from Estaillades"""
    chunks_dir = "data/estaillades_chunks"
    files = sorted(glob.glob(os.path.join(chunks_dir, "*.npy")))
    
    if len(files) < 2:
        print("  ⚠️ Not enough Estaillades chunks")
        return None, None
    
    chunk_a = np.load(files[0])
    chunk_b = np.load(files[5] if len(files) > 5 else files[1])
    
    print(f"  Sample A: φ={np.mean(chunk_a > 0):.3f}")
    print(f"  Sample B: φ={np.mean(chunk_b > 0):.3f}")
    
    return chunk_a, chunk_b


def main():
    print("="*60)
    print("☁️  UPLOAD SAMPLES TO SUPABASE")
    print("="*60)
    
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("\n❌ Set SUPABASE_URL and SUPABASE_KEY environment variables!")
        print("   export SUPABASE_URL='https://your-project.supabase.co'")
        print("   export SUPABASE_KEY='your-anon-key'")
        return
    
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    print(f"\n✅ Connected to Supabase: {SUPABASE_URL[:40]}...")
    
    # Prepare all 10 sample files
    samples = {}
    
    # 1. MEC Carbonate
    print("\n📦 MEC Carbonate...")
    mec_dir = "data/raw"
    mec_files = sorted(glob.glob(os.path.join(mec_dir, "*.npy")))
    if len(mec_files) >= 2:
        samples["MEC_Carbonate/mec_sample_a.npy"] = np.load(mec_files[0])
        samples["MEC_Carbonate/mec_sample_b.npy"] = np.load(mec_files[len(mec_files)//2])
        print(f"  ✅ 2 samples ready")
    else:
        print(f"  ⚠️ Only {len(mec_files)} files found")
    
    # 2. ILS Limestone
    print("\n📦 ILS Limestone...")
    ils_dir = "data/ils_chunks"
    ils_files = sorted(glob.glob(os.path.join(ils_dir, "*.npy")))
    if len(ils_files) >= 2:
        samples["ILS_Limestone/ils_sample_a.npy"] = np.load(ils_files[0])
        samples["ILS_Limestone/ils_sample_b.npy"] = np.load(ils_files[len(ils_files)//2])
        print(f"  ✅ 2 samples ready")
    else:
        print(f"  ⚠️ Only {len(ils_files)} files found")
    
    # 3. Synthetic Blobs
    print("\n📦 Synthetic Blobs...")
    syn_dir = "data/synthetic_raw"
    syn_files = sorted(glob.glob(os.path.join(syn_dir, "*.npy")))
    if len(syn_files) >= 2:
        samples["Synthetic_Blobs/syn_sample_a.npy"] = np.load(syn_files[0])
        samples["Synthetic_Blobs/syn_sample_b.npy"] = np.load(syn_files[len(syn_files)//2])
        print(f"  ✅ 2 samples ready")
    else:
        print(f"  ⚠️ Only {len(syn_files)} files found")
    
    # 4. Estaillades
    print("\n📦 Estaillades Carbonate...")
    est_a, est_b = prepare_estaillades_chunks()
    if est_a is not None:
        samples["Estaillades_Carbonate/est_sample_a.npy"] = est_a
        samples["Estaillades_Carbonate/est_sample_b.npy"] = est_b
        print(f"  ✅ 2 samples ready")
    
    # 5. Savonnières
    print("\n📦 Savonnières Carbonate...")
    sav_a, sav_b = prepare_savonnieres_chunks()
    if sav_a is not None:
        samples["Savonnieres_Carbonate/sav_sample_a.npy"] = sav_a
        samples["Savonnieres_Carbonate/sav_sample_b.npy"] = sav_b
        print(f"  ✅ 2 samples ready")
    
    # Upload to Supabase
    print(f"\n{'='*60}")
    print(f"☁️  UPLOADING {len(samples)} files to Supabase...")
    print(f"{'='*60}")
    
    import io
    
    for path, chunk in samples.items():
        print(f"\n  📤 {path} ({chunk.shape}, φ={np.mean(chunk > 0):.3f})...")
        
        # Convert to bytes
        buffer = io.BytesIO()
        np.save(buffer, chunk)
        file_bytes = buffer.getvalue()
        
        try:
            supabase.storage.from_(BUCKET_NAME).upload(
                path=path,
                file=file_bytes,
                file_options={"content-type": "application/octet-stream", "upsert": "true"}
            )
            print(f"  ✅ Uploaded ({len(file_bytes) / 1024:.0f} KB)")
        except Exception as e:
            print(f"  ❌ Failed: {e}")
    
    print(f"\n{'='*60}")
    print(f"✅ DONE! {len(samples)} samples uploaded to '{BUCKET_NAME}' bucket")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
