"""
Graph Extraction using SNOW Algorithm + PyTorch Geometric Conversion
Phase 2.2: Convert 3D pore structures to GNN-ready graphs with physics labels

Pipeline:
1. Load preprocessed chunks (.npy)
2. Extract pore network using SNOW2 algorithm (Fixed for PoreSpy 3.0+)
3. Calculate missing network properties (diameters, lengths)
4. Filter low-quality networks (< 30 pores)
5. Calculate permeability label via Stokes flow
6. Convert to PyTorch Geometric Data object
7. Save with source metadata for train/test splitting
"""

import numpy as np
import torch
from torch_geometric.data import Data
import porespy as ps
import os
import glob
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try importing from src.physics, handle potential import errors gracefully
try:
    from src.physics import get_permeability, get_permeability_geometric
except ImportError:
    # Fallback if running from root directory
    from physics import get_permeability, get_permeability_geometric

def extract_source_id(filename):
    """
    Extract rock identifier from filename
    
    Parameters:
    -----------
    filename : str
        e.g., 'synthetic_rock_A_chunk_064_064_000.npy'
    
    Returns:
    --------
    str : 'rock_A' or 'rock_B'
    """
    basename = os.path.basename(filename)
    if 'rock_A' in basename:
        return 'rock_A'
    elif 'rock_B' in basename:
        return 'rock_B'
    else:
        return 'unknown'

def network_to_pyg(network, permeability, source_id, log_scale=True):
    """
    Convert Porespy network to PyTorch Geometric Data object
    
    Parameters:
    -----------
    network : dict
        SNOW network dictionary
    permeability : float
        Ground-truth permeability (m²)
    source_id : str
        Rock identifier ('rock_A' or 'rock_B')
    log_scale : bool
        Apply log transformation to features and labels
    
    Returns:
    --------
    torch_geometric.data.Data : Graph ready for GNN training
    """
    
    # Extract pore properties as node features
    pore_diameter = network['pore.diameter']
    pore_volume = network['pore.volume']
    
    # Stack features: [diameter, volume]
    x = np.column_stack([pore_diameter, pore_volume])
    
    # Log-scale transformation (prevents numerical issues, improves learning)
    if log_scale:
        x = np.log10(x + 1e-15)  # Add epsilon to avoid log(0)
    
    # Convert to torch tensor
    x = torch.tensor(x, dtype=torch.float32)
    
    # Extract throat connectivity (edges)
    throat_conns = network['throat.conns']
    
    # CRITICAL: Make graph undirected by adding reverse edges
    # PyG expects undirected graphs for most GNN operations
    edge_index = np.concatenate([throat_conns, throat_conns[:, [1, 0]]], axis=0)
    edge_index = torch.tensor(edge_index.T, dtype=torch.long)  # Shape: [2, num_edges]
    
    # Permeability label (target for regression)
    if log_scale:
        y = torch.tensor([np.log10(permeability + 1e-18)], dtype=torch.float32)
    else:
        y = torch.tensor([permeability], dtype=torch.float32)
    
    # Create PyG Data object
    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        num_nodes=len(pore_diameter)
    )
    
    # CRITICAL METADATA: Store source_id for rock-wise train/test splitting
    data.source_id = source_id
    
    return data

def process_all_chunks(
    processed_dir="data/raw",  # Changed: chunks are directly in raw/ (skipped preprocessing)
    graphs_dir="data/graphs",
    voxel_size=2.68e-6,  # Real MEC data voxel size (2.68 microns)
    min_pores=30
):
    """
    Batch process all chunks into PyG graphs
    
    Parameters:
    -----------
    processed_dir : str
        Directory containing .npy chunks
    graphs_dir : str
        Output directory for .pt graph files
    voxel_size : float
        Physical voxel size in meters
    min_pores : int
        Minimum pore count threshold (filters degenerate graphs)
    """
    print("="*60)
    print("🧬 GRAPH EXTRACTION ENGINE: Phase 2.2 (Fixed SNOW2)")
    print("="*60)
    print(f"Configuration:")
    print(f"  SNOW Algorithm: porespy.networks.snow2")
    print(f"  Voxel Size: {voxel_size} m")
    print(f"  Minimum Pores: {min_pores}")
    print(f"  Feature Transform: Log10")
    print("="*60)
    
    # Ensure output directory exists
    os.makedirs(graphs_dir, exist_ok=True)
    
    # Find all processed chunks
    chunk_files = sorted(glob.glob(os.path.join(processed_dir, "*.npy")))
    total_chunks = len(chunk_files)
    
    if total_chunks == 0:
        print(f"⚠️  No chunks found in {processed_dir}")
        return
    
    print(f"\n📁 Found {total_chunks} chunks to process\n")
    
    # Statistics
    successful = 0
    failed_snow = 0
    failed_threshold = 0
    failed_physics = 0
    
    # Process each chunk
    for idx, chunk_file in enumerate(chunk_files, 1):
        filename = os.path.basename(chunk_file)
        source_id = extract_source_id(filename)
        
        try:
            # Load chunk
            chunk = np.load(chunk_file)
            
            # --- CRITICAL FIX START ---
            # Use snow2 and extract the dictionary from the result object
            snow_output = ps.networks.snow2(chunk, voxel_size=voxel_size)
            network = snow_output.network
            # --- CRITICAL FIX END ---
            
            # CRITICAL: SNOW2 in Porespy 3.0 doesn't return diameter/length properties
            # Calculate them manually
            if 'pore.diameter' not in network:
                # Estimate pore diameter from volume (assuming spherical pores)
                network['pore.diameter'] = 2 * (3 * network['pore.volume'] / (4 * np.pi)) ** (1/3)
            
            if 'throat.diameter' not in network or 'throat.length' not in network:
                # Calculate throat properties from pore connectivity
                conns = network['throat.conns']
                pore_coords = network['pore.coords']
                
                # Throat length: distance between connected pores
                throat_lengths = np.linalg.norm(
                    pore_coords[conns[:, 0]] - pore_coords[conns[:, 1]], 
                    axis=1
                )
                network['throat.length'] = throat_lengths
                
                # Throat diameter: average of connected pore diameters * 0.5
                throat_diameters = 0.5 * (
                    network['pore.diameter'][conns[:, 0]] + 
                    network['pore.diameter'][conns[:, 1]]
                )
                network['throat.diameter'] = throat_diameters
            
            # Quality check: Minimum number of pores
            num_pores = network['pore.coords'].shape[0]
            
            if num_pores < min_pores:
                failed_threshold += 1
                print(f"⚠️  [{idx}/{total_chunks}] {filename}: Too few pores ({num_pores} < {min_pores}) - SKIPPED")
                continue
            
            # Calculate ground-truth permeability using Stokes flow
            K = get_permeability(network, chunk.shape)
            
            # Fallback: If Stokes fails (disconnected), use geometric estimate
            if K == 0.0:
                chunk_porosity = np.sum(chunk) / chunk.size
                K = get_permeability_geometric(network, chunk.shape, chunk_porosity)
                if K == 0.0:
                    failed_physics += 1
                    print(f"⚠️  [{idx}/{total_chunks}] {filename}: Both Stokes and geometric failed - SKIPPED")
                    continue
            
            # Convert to PyG graph
            graph = network_to_pyg(network, K, source_id)
            
            # Save graph
            graph_filename = os.path.splitext(filename)[0] + ".pt"
            graph_path = os.path.join(graphs_dir, graph_filename)
            torch.save(graph, graph_path)
            
            successful += 1
            print(f"✅ [{idx}/{total_chunks}] {source_id}: {num_pores} pores, K={K:.3e} m² → {graph_filename}")
        
        except Exception as e:
            failed_snow += 1
            print(f"❌ [{idx}/{total_chunks}] {filename}: Extraction failed - {e}")
    
    # Final summary
    print(f"\n{'='*60}")
    print("📊 EXTRACTION SUMMARY")
    print(f"{'='*60}")
    print(f"Total Chunks Processed: {total_chunks}")
    print(f"✅ Successful Graphs: {successful}")
    print(f"⚠️  Failed (< {min_pores} pores): {failed_threshold}")
    print(f"⚠️  Failed (Zero K): {failed_physics}")
    print(f"❌ Failed (Error): {failed_snow}")
    
    if total_chunks > 0:
        print(f"Success Rate: {successful/total_chunks*100:.1f}%")
    
    print(f"\n💾 Saved to: {graphs_dir}/")
    print(f"\n🎯 Ready for Phase 2.3 (Verification)")

def main():
    """Execute graph extraction pipeline"""
    process_all_chunks()

if __name__ == "__main__":
    main()
