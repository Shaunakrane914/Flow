"""
ILS Processing - PARALLEL + FIXED VERSION
Streaming HDF5 + Multi-core for max speed
"""

import numpy as np
import os
import torch
from torch_geometric.data import Data
from tqdm import tqdm
import sys
import multiprocessing as mp
import porespy as ps

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physics import get_permeability

# Configuration
ILS_FILE = "data/raw/ILS_seg_hr.mat"
OUTPUT_DIR = "data/graphs_ils"
CHUNK_SIZE = 128
VOXEL_SIZE = 1e-6


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
    """Worker for parallel processing - reads directly from HDF5"""
    h5_file, dataset_name, ix, iy, iz, pore_label, output_dir = args
    
    try:
        import h5py
        
        x = ix * CHUNK_SIZE
        y = iy * CHUNK_SIZE
        z = iz * CHUNK_SIZE
        
        # Read chunk from HDF5 (each worker opens file independently)
        with h5py.File(h5_file, 'r') as f:
            chunk = np.array(f[dataset_name][x:x+CHUNK_SIZE, y:y+CHUNK_SIZE, z:z+CHUNK_SIZE])
        
        # Calculate porosity
        phi = np.sum(chunk == pore_label) / chunk.size
        
        # Filter
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
        source_id = f"ils_{ix:02d}_{iy:02d}_{iz:02d}"
        graph = network_to_graph(network, permeability, source_id, log_scale=True)
        
        output_file = os.path.join(output_dir, f"{source_id}.pt")
        torch.save(graph, output_file)
        
        return {'status': 'success',  'k': permeability, 'phi': phi}
        
    except Exception as e:
        return {'status': 'error', 'error': f'{type(e).__name__}: {e}'}


def main():
    print("="*70)
    print("🚀 ILS PROCESSING - TURBO PARALLEL MODE")
    print("="*70)
    
    if not os.path.exists(ILS_FILE):
        print(f"\n❌ File not found: {ILS_FILE}")
        return
    
    print(f"\n📂 Found: {os.path.basename(ILS_FILE)}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Open HDF5 to get metadata
    print(f"\n📊 Reading HDF5 metadata...")
    try:
        import h5py
        with h5py.File(ILS_FILE, 'r') as f:
            rock_key = None
            for key in f.keys():
                if isinstance(f[key], h5py.Dataset) and len(f[key].shape) == 3:
                    rock_key = key
                    break
            
            if rock_key is None:
                print("❌ No 3D dataset found")
                return
            
            shape = f[rock_key].shape
            sample = np.array(f[rock_key][0:100, 0:100, 0:100])
            unique_vals = np.unique(sample)
            
            print(f"✅ Dataset: '{rock_key}' | Shape: {shape}")
            print(f"   Unique values: {unique_vals}")
            
            pore_label = int(np.min(unique_vals))
            print(f"   Using {pore_label} as pore label")
        
    except ImportError:
        print("\n❌ h5py not installed - run: pip install h5py")
        return
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return
    
    # Prepare chunk coordinates
    nx = (shape[0] - CHUNK_SIZE) // CHUNK_SIZE + 1
    ny = (shape[1] - CHUNK_SIZE) // CHUNK_SIZE + 1
    nz = (shape[2] - CHUNK_SIZE) // CHUNK_SIZE + 1
    
    print(f"\n⛏️  Chunks: {nx}×{ny}×{nz} = {nx*ny*nz} total")
    
    # Prepare arguments for parallel processing
    args_list = []
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                args_list.append((ILS_FILE, rock_key, ix, iy, iz, pore_label, OUTPUT_DIR))
    
    # Use 4 workers
    n_workers = 4
    print(f"\n🚀 Using {n_workers} parallel workers")
    print(f"   Expected speedup: ~3-4x vs sequential")
    print(f"   Estimated time: ~1-2 hours")
    
    # Statistics
    stats = {'success': 0, 'filtered': 0, 'failed_snow': 0, 'failed_threshold': 0, 'failed_physics': 0, 'error': 0}
    
    print(f"\n{'='*70}")
    print("🏃 PROCESSING...")
    print("="*70)
    
    with mp.Pool(processes=n_workers) as pool:
        with tqdm(total=len(args_list), desc="Processing", unit="chunk") as pbar:
            for result in pool.imap(process_chunk_worker, args_list):
                status = result['status']
                stats[status] = stats.get(status, 0) + 1
                
                if status == 'success':
                    pbar.set_postfix({'Saved': stats['success'], 'Rate': f"{stats['success']/(sum(stats.values())-stats['filtered'])*100:.1f}%"})
                
                pbar.update(1)
    
    # Results
    print(f"\n{'='*70}")
    print("📊 ILS PROCESSING RESULTS")
    print("="*70)
    print(f"Total chunks: {len(args_list)}")
    print(f"  Filtered (porosity): {stats['filtered']}")
    print(f"  Attempted: {len(args_list) - stats['filtered']}")
    print(f"\n✅ Successful: {stats['success']}")
    print(f"❌ Failed SNOW: {stats['failed_snow']}")
    print(f"❌ Failed threshold: {stats['failed_threshold']}")
    print(f"❌ Failed physics: {stats['failed_physics']}")
    print(f"❌ Other errors: {stats.get('error', 0)}")
    
    attempted = len(args_list) - stats['filtered']
    if attempted > 0:
        success_rate = stats['success'] / attempted * 100
        print(f"\n🎯 Success Rate: {success_rate:.1f}%")
    
    print(f"\n📂 Saved to: {OUTPUT_DIR}/")
    
    if stats['success'] >= 50:
        print(f"\n✅ EXCELLENT: {stats['success']} samples!")
        print("\nNext steps:")
        print("  1. python src/train_ils.py")
        print("  2. python src/baseline_ils.py")
        print("  3. Compare: GNN vs Baseline on ILS")
    elif stats['success'] >= 20:
        print(f"\n⚠️  MODERATE: {stats['success']} samples")
    else:
        print(f"\n💡 Only {stats['success']} samples - may need more relaxed thresholds")
    
    print("="*70)


if __name__ == "__main__":
    main()
