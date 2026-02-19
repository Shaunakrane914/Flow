"""
Inference Pipeline for Single Chunk Permeability Prediction
Phase 5.1: Complete end-to-end prediction with visualization

Pipeline:
1. Load chunk (.npy file)
2. Extract pore network (SNOW2)
3. Convert to PyG graph
4. Predict permeability (GNN)
5. Generate 3D visualization
"""

import torch
import numpy as np
import porespy as ps
import os
import sys
import multiprocessing
import logging
import traceback

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import TopoFlowGNN
from src.graph_extraction import network_to_pyg
from src.visualize import render_flow

# ── Logging Setup ─────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)
# Ensure logger propagates to root logger (so it uses main.py's config)
logger.propagate = True
logger.setLevel(logging.INFO)


def predict_single_chunk(chunk_path, model_path='models/best_model.pth', 
                         output_image='output_inference.png',
                         use_hybrid=False,
                         rock_type='MEC'):
    """
    Complete inference pipeline for a single chunk
    
    Parameters:
    -----------
    chunk_path : str
        Path to .npy chunk file
    model_path : str
        Path to trained model weights
    output_image : str
        Path to save visualization
    use_hybrid : bool
        Whether to use hybrid model (formula + GNN)
    rock_type : str
        Rock type for model selection
    
    Returns:
    --------
    tuple : (permeability, image_path, baseline_k)
        permeability: Predicted permeability in m²
        image_path: Path to generated visualization
        baseline_k: Kozeny-Carman baseline (None if not hybrid)
    """
    try:
        logger.info("="*60)
        logger.info("TOPO-FLOW INFERENCE PIPELINE")
        logger.info("="*60)
        logger.info(f"Input: {os.path.basename(chunk_path)}")
        logger.info(f"Model: {model_path}")
        print("="*60)  # Also print to ensure visibility
        print("TOPO-FLOW INFERENCE PIPELINE")
        print("="*60)
        
        print("="*60)
        print("TOPO-FLOW INFERENCE PIPELINE")
        print("="*60)
        print(f"Input: {os.path.basename(chunk_path)}")
        print(f"Model: {model_path}")
        
        # Step 1: Load chunk
        print("\nStep 1: Loading chunk...")
        chunk = np.load(chunk_path)
        print(f"  [OK] Chunk shape: {chunk.shape}")
        print(f"  [OK] Porosity: {np.sum(chunk) / chunk.size:.3f}")
        
        # Step 2: Extract pore network (with parallelization)
        print("\nStep 2: Extracting pore network...")
        voxel_size = 2.68e-6  # Real MEC data voxel size (2.68 microns)
        
        # Use all CPU cores for faster extraction
        ncores = max(1, multiprocessing.cpu_count() - 1)  # Leave 1 core free
        print(f"  Using {ncores} CPU cores for parallel processing")
        
        try:
            snow_output = ps.networks.snow2(
                chunk, 
                voxel_size=voxel_size,
                parallelization={'num_cores': ncores}
            )
        except (TypeError, KeyError):
            # Fallback if parallelization not supported in this Porespy version
            snow_output = ps.networks.snow2(chunk, voxel_size=voxel_size)
        
        network = snow_output.network
    
        # Add missing properties (same as graph_extraction.py)
        if 'pore.diameter' not in network:
            network['pore.diameter'] = 2 * (3 * network['pore.volume'] / (4 * np.pi)) ** (1/3)
        
        if 'throat.diameter' not in network or 'throat.length' not in network:
            conns = network['throat.conns']
            pore_coords = network['pore.coords']
            
            throat_lengths = np.linalg.norm(
                pore_coords[conns[:, 0]] - pore_coords[conns[:, 1]], 
                axis=1
            )
            network['throat.length'] = throat_lengths
            
            throat_diameters = 0.5 * (
                network['pore.diameter'][conns[:, 0]] + 
                network['pore.diameter'][conns[:, 1]]
            )
            network['throat.diameter'] = throat_diameters
        
        num_pores = network['pore.coords'].shape[0]
        num_throats = network['throat.conns'].shape[0]
        print(f"  [OK] Network: {num_pores} pores, {num_throats} throats")
        
        # Step 3: Convert to PyG graph
        print("\nStep 3: Converting to graph...")
        # Use dummy permeability (we're predicting it!)
        dummy_permeability = 1e-15
        graph = network_to_pyg(network, dummy_permeability, source_id='inference')
        print(f"  [OK] Graph: {graph.num_nodes} nodes, {graph.num_edges} edges")
        print(f"  [OK] Features: {graph.x.shape}")
        
        # Step 3.5: Calculate Baseline (if hybrid mode)
        baseline_k = None
        if use_hybrid:
            print("\nStep 3.5: Calculating Kozeny-Carman baseline...")
            OPTIMAL_C = 1.0  # Calibrated for MEC
            
            phi = np.sum(chunk) / chunk.size
            if num_pores > 0:
                avg_diameter = np.mean(network['pore.diameter'])
            else:
                avg_diameter = 1e-6
            
            baseline_k = (avg_diameter ** 2 * phi ** 3) / (OPTIMAL_C * (1 - phi) ** 2)
            print(f"  [OK] Baseline (Kozeny-Carman): {baseline_k:.4e} m^2")
        
        # Step 4: Model prediction (GPU-accelerated)
        print("\nStep 4: Running GNN prediction...")
        
        # Force GPU if available
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"  [OK] Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device('cpu')
            print(f"  [WARN] Using CPU (no GPU detected)")
        
        # Load model (hybrid or standard based on use_hybrid flag)
        if use_hybrid:
            from src.model_hybrid import HybridPhysicsGNN
            model = HybridPhysicsGNN(dropout=0.1).to(device)
            model_path = 'models/best_model_hybrid.pth'
        else:
            model = TopoFlowGNN(dropout=0.1).to(device)
            # Auto-select model based on rock type
            if rock_type:
                model_paths = {
                    'MEC': 'models/best_model.pth',
                    'Synthetic': 'models/best_model_synthetic.pth',
                    'ILS': 'models/best_model_ils.pth',
                    'Estaillades': 'models/best_model_estaillades.pth'
                }
                model_path = model_paths.get(rock_type, model_path)
        
        # Check if model file exists - if not, use baseline fallback
        use_baseline_fallback = False
        if not os.path.exists(model_path):
            logger.warning(f"Model file not found: {model_path}")
            logger.warning("Falling back to Kozeny-Carman baseline prediction")
            print(f"  [WARN] Model file not found: {model_path}")
            print(f"  [WARN] Falling back to Kozeny-Carman baseline prediction")
            use_baseline_fallback = True
        else:
            # Load weights
            try:
                logger.info(f"Loading model from: {model_path}")
                state_dict = torch.load(model_path, map_location=device, weights_only=False)
                model.load_state_dict(state_dict)
                model.eval()
                logger.info(f"Model loaded successfully from: {model_path}")
                print(f"  [OK] Model loaded from: {model_path}")
            except Exception as e:
                logger.error(f"Failed to load model from {model_path}: {str(e)}")
                logger.error(traceback.format_exc())
                print(f"  [WARN] Failed to load model: {e}")
                print(f"  [WARN] Falling back to Kozeny-Carman baseline prediction")
                use_baseline_fallback = True
        
        # Fallback to baseline if model unavailable
        if use_baseline_fallback:
            print("\nUsing Kozeny-Carman baseline (no trained model available)...")
            phi = np.sum(chunk) / chunk.size
            if num_pores > 0:
                avg_diameter = np.mean(network['pore.diameter'])
            else:
                avg_diameter = 1e-6
            
            OPTIMAL_C = 1.0  # Calibrated constant
            predicted_k = (avg_diameter ** 2 * phi ** 3) / (OPTIMAL_C * (1 - phi) ** 2)
            baseline_k = predicted_k  # Same value for baseline in fallback mode
            log_k = np.log10(predicted_k) if predicted_k > 0 else -20
            print(f"  [OK] Baseline prediction: {predicted_k:.4e} m^2")
            print(f"  [WARN] Note: Train a GNN model for improved accuracy")
            
            # Skip GNN prediction, go straight to visualization
            graph = None
        
        # Run GNN prediction (skip if using baseline fallback)
        if not use_baseline_fallback:
            # Prepare batch and move to GPU
            graph = graph.to(device)
            batch = torch.zeros(graph.num_nodes, dtype=torch.long, device=device)
            
            # Predict (no gradient computation for speed)
            with torch.no_grad():
                if use_hybrid and baseline_k is not None:
                    baseline_k_tensor = torch.tensor([[np.log10(baseline_k)]], dtype=torch.float32, device=device)
                    log_k = model(graph.x, graph.edge_index, batch, baseline_k_tensor)
                else:
                    log_k = model(graph.x, graph.edge_index, batch)
            
            # Inverse log transform
            predicted_k = 10 ** log_k.item()
            
            print(f"  [OK] Prediction:")
            print(f"     Log10(K): {log_k.item():.4f}")
            print(f"     Permeability: {predicted_k:.4e} m^2")
            if use_hybrid and baseline_k is not None:
                improvement = ((baseline_k - predicted_k) / baseline_k) * 100
                print(f"     Improvement: {improvement:+.2f}%")
        
        # Step 5: Generate visualization
        print("\nStep 5: Generating 3D visualization...")
        logger.info("Generating 3D visualization...")
        try:
            render_flow(chunk_path, output_file=output_image)
            logger.info(f"Visualization saved: {output_image}")
            print(f"  [OK] Visualization saved: {output_image}")
        except Exception as e:
            logger.warning(f"Visualization failed: {str(e)}")
            logger.debug(traceback.format_exc())
            print(f"  [WARN] Visualization failed: {e}")
            output_image = None
        
        # Summary
        print(f"\n{'='*60}")
        print("INFERENCE COMPLETE")
        print(f"{'='*60}")
        print(f"Predicted Permeability: {predicted_k:.4e} m^2")
        if baseline_k is not None:
            print(f"Baseline (Kozeny-Carman): {baseline_k:.4e} m^2")
        print(f"Visualization: {output_image if output_image else 'Not generated'}")
        print("="*60)
        
        logger.info(f"Inference completed successfully - K: {predicted_k:.4e} m^2")
        return predicted_k, output_image, baseline_k
    
    except Exception as e:
        logger.error(f"Fatal error in predict_single_chunk: {str(e)}")
        logger.error(traceback.format_exc())
        raise


def compare_with_physics(chunk_path):
    """
    Compare GNN prediction with physics-based calculation
    
    Parameters:
    -----------
    chunk_path : str
        Path to chunk file
    
    Returns:
    --------
    dict : Comparison results
    """
    from src.physics import get_permeability, get_permeability_geometric
    
    print("\n" + "="*60)
    print("PREDICTION vs PHYSICS COMPARISON")
    print("="*60)
    
    # Get GNN prediction
    predicted_k, _ = predict_single_chunk(chunk_path, output_image=None)
    
    # Get physics-based estimate
    chunk = np.load(chunk_path)
    snow_output = ps.networks.snow2(chunk, voxel_size=1e-6)
    network = snow_output.network
    
    # Add missing properties
    if 'pore.diameter' not in network:
        network['pore.diameter'] = 2 * (3 * network['pore.volume'] / (4 * np.pi)) ** (1/3)
    if 'throat.length' not in network:
        conns = network['throat.conns']
        pore_coords = network['pore.coords']
        network['throat.length'] = np.linalg.norm(
            pore_coords[conns[:, 0]] - pore_coords[conns[:, 1]], axis=1
        )
        network['throat.diameter'] = 0.5 * (
            network['pore.diameter'][conns[:, 0]] + 
            network['pore.diameter'][conns[:, 1]]
        )
    
    # Try Stokes
    physics_k = get_permeability(network, chunk.shape)
    
    # Fallback to geometric
    if physics_k == 0.0:
        porosity = np.sum(chunk) / chunk.size
        physics_k = get_permeability_geometric(network, chunk.shape, porosity)
    
    # Compare
    print(f"\nResults:")
    print(f"  GNN Prediction:    {predicted_k:.4e} m^2")
    print(f"  Physics Estimate:  {physics_k:.4e} m^2")
    print(f"  Ratio (GNN/Phys):  {predicted_k/physics_k:.2f}x")
    print(f"  Difference:        {abs(predicted_k - physics_k)/physics_k*100:.1f}%")
    print("="*60)
    
    return {
        'gnn': predicted_k,
        'physics': physics_k,
        'ratio': predicted_k / physics_k
    }


def main():
    """Test inference on a sample chunk"""
    import glob
    import random
    
    # Find a test chunk (preferably from Rock B)
    chunks = glob.glob("data/processed/*.npy")
    rock_b_chunks = [c for c in chunks if 'rock_B' in c]
    
    if rock_b_chunks:
        test_chunk = random.choice(rock_b_chunks)
    else:
        test_chunk = random.choice(chunks)
    
    print(f"Testing on: {os.path.basename(test_chunk)}\n")
    
    # Run inference
    predicted_k, image_path = predict_single_chunk(
        test_chunk,
        output_image='inference_test.png'
    )
    
    print("\nInference test complete!")
    print(f"   Prediction: {predicted_k:.4e} m^2")
    print(f"   Image: {image_path}")


if __name__ == "__main__":
    main()
