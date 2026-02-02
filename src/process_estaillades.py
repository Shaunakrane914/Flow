"""
Estaillades Carbonate Processor - The Boss Level
Famous vuggy carbonate where topology complexity might finally matter
"""

import numpy as np
import os
import sys
import torch
from torch_geometric.data import Data
from tqdm import tqdm
import porespy as ps
import multiprocessing as mp

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physics import get_permeability

# Configuration
INPUT_FILE = "data/raw/Estaillades_1000c_3p31136um.raw"
OUTPUT_DIR = "data/graphs_estaillades"
DIMENSION = 1000  # 1000x1000x1000
VOXEL_SIZE = 3.31e-6  # 3.31 microns
CHUNK_SIZE = 128


def calculate_missing_properties(network):
    """Calculate missing network properties"""
    if 'pore.diameter' not in network:
        pore_areas = network.get('pore.area', network.get('pore.equivalent_diameter', None))
        if pore_areas is not None:
            network['pore.diameter'] = 2 * np.sqrt(pore_areas / np.pi)
        else:
            network['pore.diameter'] = np.cbrt(network['pore.volume'] * 6 / np.pi)
    
    if 'throat.diameter' not in network:
        throat_areas = network.get('throat.cross_sectional_area', None)
        if throat_areas is not None:
            network['throat.diameter'] = 2 * np.sqrt(throat_areas / np.pi)
        else:
            conns = network['throat.conns']
            network['throat.diameter'] = 0.5 * (
                network['pore.diameter'][conns[:, 0]] + 
                network['pore.diameter'][conns[:, 1]]
            )
    
    if 'throat.length' not in network:
        conns = network['throat.conns']
        coords1 = network['pore.coords'][conns[:, 0]]
        coords2 = network['pore.coords'][conns[:, 1]]
        network['throat.length'] = np.linalg.norm(coords2 - coords1, axis=1)


def network_to_graph(network, permeability, source_id, log_scale=True):
    """Convert network to PyG graph"""
    pore_diameter = network['pore.diameter']
    pore_volume = network['pore.volume']
    
    x_diameter = np.log10(pore_diameter + 1e-12)
    x_volume = np.log10(pore_volume + 1e-18)
    
    x = torch.tensor(np.column_stack([x_diameter, x_volume]), dtype=torch.float32)
    
    throat_conns = network['throat.conns']
    edge_index = np.concatenate([throat_conns, throat_conns[:, [1, 0]]], axis=0)
    edge_index = torch.tensor(edge_index.T, dtype=torch.long)
    
    if log_scale:
        y = torch.tensor([np.log10(permeability + 1e-18)], dtype=torch.float32)
    else:
        y = torch.tensor([permeability], dtype=torch.float32)
    
    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        num_nodes=len(pore_diameter)
    )
    
    data.source_id = source_id
    return data


def process_chunk_worker(args):
    """Worker for parallel processing"""
    full_rock, ix, iy, iz, pore_label, output_dir = args
    
    try:
        x = ix * CHUNK_SIZE
        y = iy * CHUNK_SIZE
        z = iz * CHUNK_SIZE
        
        # Extract chunk
        chunk = full_rock[x:x+CHUNK_SIZE, y:y+CHUNK_SIZE, z:z+CHUNK_SIZE].copy()
        
        # Calculate porosity
        phi = np.sum(chunk == pore_label) / chunk.size
        
        # Filter for interesting samples (0.05-0.40 porosity)
        if phi < 0.05 or phi > 0.40:
            return {'status': 'filtered', 'phi': phi}
        
        # Extract network
        try:
            snow_output = ps.networks.snow2(chunk, voxel_size=VOXEL_SIZE)
            network = snow_output.network
        except Exception as e:
            return {'status': 'failed_snow', 'error': str(e)}
        
        # Calculate properties
        calculate_missing_properties(network)
        
        # Threshold
        num_pores = network['pore.coords'].shape[0]
        if num_pores < 30:
            return {'status': 'failed_threshold', 'pores': num_pores}
        
        # STRICT PHYSICS
        try:
            permeability = get_permeability(network, chunk.shape)
        except Exception as e:
            return {'status': 'failed_physics', 'error': f'{type(e).__name__}: {e}'}
        
        if permeability is None:
            return {'status': 'failed_physics', 'error': 'returned None'}
        
        # SUCCESS - Save
        source_id = f"est_{ix:02d}_{iy:02d}_{iz:02d}"
        graph = network_to_graph(network, permeability, source_id, log_scale=True)
        
        output_file = os.path.join(output_dir, f"{source_id}.pt")
        torch.save(graph, output_file)
        
        return {'status': 'success', 'k': permeability, 'phi': phi}
        
    except Exception as e:
        return {'status': 'error', 'error': f'{type(e).__name__}: {e}'}


def main():
    print("="*70)
    print("🏰 ESTAILLADES CARBONATE - THE VUGGY BOSS LEVEL")
    print("="*70)
    print("\nFamous for complex vugs where topology might finally matter!")
    
    # Check file
    if not os.path.exists(INPUT_FILE):
        print(f"\n❌ File not found: {INPUT_FILE}")
        print("Please download and extract Estaillades_1000c_3p31136um.raw.gz")
        return
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load volume
    print(f"\n📂 Loading Estaillades volume...")
    print(f"   File: {INPUT_FILE}")
    print(f"   Size: {os.path.getsize(INPUT_FILE) / 1e9:.2f} GB")
    
    try:
        full_rock = np.fromfile(INPUT_FILE, dtype=np.uint8)
        full_rock = full_rock.reshape((DIMENSION, DIMENSION, DIMENSION))
        print(f"✅ Loaded: {full_rock.shape}")
        
        # Detect labeling
        sample = full_rock[:100, :100, :100]
        unique_vals = np.unique(sample)
        print(f"   Unique values: {unique_vals}")
        
        # Detect pore label
        phi_test = np.sum(sample == 0) / sample.size
        if phi_test > 0.6:
            print("   High porosity detected - flipping labels")
            full_rock = 1 - full_rock
            pore_label = 1
        else:
            pore_label = 0
        
        phi_global = np.sum(full_rock == pore_label) / full_rock.size
        print(f"   Global porosity: {phi_global:.2%}")
        print(f"   Using {pore_label} as pore label")
        
    except Exception as e:
        print(f"❌ Failed to load: {e}")
        return
    
    # Prepare chunk coordinates (stride 128 for non-overlapping samples)
    nx = (DIMENSION - CHUNK_SIZE) // CHUNK_SIZE + 1
    ny = (DIMENSION - CHUNK_SIZE) // CHUNK_SIZE + 1
    nz = (DIMENSION - CHUNK_SIZE) // CHUNK_SIZE + 1
    
    print(f"\n⛏️  Chunks: {nx}×{ny}×{nz} = {nx*ny*nz} potential")
    
    # Prepare arguments for parallel processing
    args_list = []
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                args_list.append((full_rock, ix, iy, iz, pore_label, OUTPUT_DIR))
    
    # Use 4 workers
    n_workers = 4
    target_samples = 200  # Stop after 200 successful samples
    
    print(f"\n🚀 Processing with {n_workers} workers")
    print(f"   Target: {target_samples} successful samples")
    print(f"   Stopping early when target reached")
    
    # Statistics
    stats = {'success': 0, 'filtered': 0, 'failed_snow': 0, 'failed_threshold': 0, 'failed_physics': 0, 'error': 0}
    
    print(f"\n{'='*70}")
    print("🏃 PROCESSING ESTAILLADES")
    print("="*70)
    
    with mp.Pool(processes=n_workers) as pool:
        with tqdm(total=min(len(args_list), target_samples*2), desc="Processing", unit="chunk") as pbar:
            for result in pool.imap(process_chunk_worker, args_list):
                status = result['status']
                stats[status] = stats.get(status, 0) + 1
                
                if status == 'success':
                    pbar.set_postfix({
                        'Saved': stats['success'],
                        'Rate': f"{stats['success']/(sum(stats.values())-stats['filtered'])*100:.1f}%"
                    })
                
                pbar.update(1)
                
                # Stop if we hit target
                if stats['success'] >= target_samples:
                    print(f"\n✅ Reached target of {target_samples} samples!")
                    pool.terminate()
                    break
    
    # Results
    print(f"\n{'='*70}")
    print("📊 ESTAILLADES PROCESSING RESULTS")
    print("="*70)
    print(f"Chunks processed: {sum(stats.values())}")
    print(f"  Filtered (porosity): {stats['filtered']}")
    print(f"  Attempted: {sum(stats.values()) - stats['filtered']}")
    print(f"\n✅ Successful: {stats['success']}")
    print(f"❌ Failed SNOW: {stats['failed_snow']}")
    print(f"❌ Failed threshold: {stats['failed_threshold']}")
    print(f"❌ Failed physics: {stats['failed_physics']}")
    print(f"❌ Other errors: {stats.get('error', 0)}")
    
    attempted = sum(stats.values()) - stats['filtered']
    if attempted > 0:
        success_rate = stats['success'] / attempted * 100
        print(f"\n🎯 Success Rate: {success_rate:.1f}%")
    
    print(f"\n📂 Saved to: {OUTPUT_DIR}/")
    
    if stats['success'] >= 50:
        print(f"\n✅ EXCELLENT: {stats['success']} Estaillades samples!")
        print("\n🔥 THE BOSS FIGHT:")
        print("  1. python src/train_estaillades.py")
        print("  2. python src/baseline_estaillades.py")
        print("  3. FINAL VERDICT: Will GNN finally win on vuggy carbonates?")
    else:
        print(f"\n⚠️  Only {stats['success']} samples - may need more")
    
    print("="*70)


if __name__ == "__main__":
    main()
