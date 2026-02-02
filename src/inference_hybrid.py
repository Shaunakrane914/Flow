"""
Hybrid Inference Module
Wraps standard inference with Kozeny-Carman baseline calculation
"""

import torch
import numpy as np
from src.physics import calculate_kozeny_carman


def predict_hybrid(chunk_path, graph, network, chunk, device):
    """
    Hybrid prediction: Kozeny-Carman + GNN residual
    
    Returns:
        (predicted_k, baseline_k)
    """
    from src.model_hybrid import HybridPhysicsGNN
    
    # Calculate Kozeny-Carman baseline
    OPTIMAL_C = 1.0  # Calibrated for MEC
    
    phi = np.sum(chunk) / chunk.size
    num_pores = network['pore.coords'].shape[0]
    
    if num_pores > 0:
        avg_diameter = np.mean(network['pore.diameter'])
    else:
        avg_diameter = 1e-6
    
    baseline_k = (avg_diameter ** 2 * phi ** 3) / (OPTIMAL_C * (1 - phi) ** 2)
    
    # Load hybrid model
    model = HybridPhysicsGNN(dropout=0.1).to(device)
    state_dict = torch.load('models/best_model_hybrid.pth', map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()
    
    # Prepare inputs
    graph = graph.to(device)
    batch = torch.zeros(graph.num_nodes, dtype=torch.long, device=device)
    baseline_k_tensor = torch.tensor([[np.log10(baseline_k)]], dtype=torch.float32, device=device)
    
    # Predict
    with torch.no_grad():
        log_k = model(graph.x, graph.edge_index, batch, baseline_k_tensor)
    
    predicted_k = 10 ** log_k.item()
    
    return predicted_k, baseline_k
