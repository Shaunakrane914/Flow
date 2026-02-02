"""
Physics-Based Permeability Calculation using Stokes Flow Simulation
Phase 2.1 (FIXED): Ground-Truth Label Generation via OpenPNM

CRITICAL FIX: Added hydraulic conductance calculation using Hagen-Poiseuille equation.
Without this, the flow solver cannot compute flow rates properly.

This module calculates permeability (K) using Darcy's Law:
    K = (Q · μ · L) / (A · ΔP)

Where:
    Q  = Flow rate through outlet pores (m³/s)
    μ  = Dynamic viscosity (Pa·s) 
    L  = Sample length (m)
    A  = Cross-sectional area (m²)
    ΔP = Pressure difference (Pa)
"""

import numpy as np
import openpnm as op


def get_permeability(network, shape):
    """
    STRICT PHYSICS-ONLY Permeability Calculation
    
    Returns real Stokes flow physics or None (no geometric fallback)
    
    Algorithm:
    1. Find all connected components
    2. Keep ONLY components that span inlet to outlet
    3. Run Stokes solver with robust settings
    4. Return K if successful, None if solver fails
    
    Parameters:
    -----------
    network : dict
        SNOW network dictionary
    shape : list or tuple
        3D dimensions [depth, height, width]
    
    Returns:
    --------
    float or None : Permeability in m² (None if physics fails)
    """
    from scipy.sparse import csgraph
    
    # Constants
    VOXEL_SIZE = 2.68e-6  # MEC data
    VISCOSITY = 0.001
    DELTA_P = 101325.0
    
    try:
        # Setup network
        pn = op.network.Network()
        pn.update(network)
        
        initial_pores = pn.Np
        initial_throats = pn.Nt
        
        # Calculate geometry
        depth, height, width = shape
        L = depth * VOXEL_SIZE
        A = (height * VOXEL_SIZE) * (width * VOXEL_SIZE)
        
        # === STEP 1: Identify Boundaries ===
        z_coords = pn['pore.coords'][:, 2]
        z_min = np.min(z_coords)
        z_max = np.max(z_coords)
        margin = VOXEL_SIZE * 0.5
        
        inlet_pores = np.where(z_coords <= (z_min + margin))[0]
        outlet_pores = np.where(z_coords >= (z_max - margin))[0]
        
        if len(inlet_pores) == 0 or len(outlet_pores) == 0:
            print(f"  ❌ No boundaries found")
            return None
        
        # === STEP 2: NUCLEAR CONNECTIVITY FIX ===
        # Find all disconnected clusters
        am = pn.create_adjacency_matrix(fmt='coo')
        n_components, labels = csgraph.connected_components(am, directed=False)
        
        # Find clusters that connect inlet to outlet
        valid_clusters = []
        for i in range(n_components):
            cluster_mask = (labels == i)
            cluster_pores = pn.Ps[cluster_mask]
            
            has_inlet = np.any(np.isin(inlet_pores, cluster_pores))
            has_outlet = np.any(np.isin(outlet_pores, cluster_pores))
            
            if has_inlet and has_outlet:
                valid_clusters.append(cluster_pores)
        
        if not valid_clusters:
            print(f"  ❌ No percolating path (0/{n_components} clusters span boundaries)")
            return None
        
        # Keep ONLY valid clusters
        pores_to_keep = np.concatenate(valid_clusters)
        pores_to_trim = np.setdiff1d(pn.Ps, pores_to_keep)
        
        # Initialize trim_pct (fixes UnboundLocalError)
        trim_pct = 0.0
        
        if len(pores_to_trim) > 0:
            op.topotools.trim(network=pn, pores=pores_to_trim)
            trim_pct = len(pores_to_trim) / initial_pores * 100
            print(f"  🔧 Nuclear trim: {len(pores_to_trim)}/{initial_pores} pores ({trim_pct:.1f}%)")
            print(f"     Kept {len(valid_clusters)} percolating cluster(s)")
        
        # Minimum network size check
        if pn.Nt < 50:
            print(f"  ❌ Network too sparse after trim: {pn.Nt} throats (min: 50)")
            return None
        
        # === STEP 3: Re-identify Boundaries ===
        z_coords = pn['pore.coords'][:, 2]
        z_min = np.min(z_coords)
        z_max = np.max(z_coords)
        
        inlet_pores = np.where(z_coords <= (z_min + margin))[0]
        outlet_pores = np.where(z_coords >= (z_max - margin))[0]
        
        if len(inlet_pores) == 0 or len(outlet_pores) == 0:
            print(f"  ❌ Lost boundaries after trim")
            return None
        
        # === STEP 4: Setup Phase ===
        water = op.phase.Phase(network=pn)
        water['pore.viscosity'] = VISCOSITY
        
        # Calculate hydraulic conductance
        throat_diameter = pn['throat.diameter']
        throat_length = pn['throat.length']
        
        R = throat_diameter / 2
        L_throat = throat_length + 1e-15
        g = (np.pi * R**4) / (8 * VISCOSITY * L_throat)
        g = g + 1e-25  # Numerical stabilizer
        water['throat.hydraulic_conductance'] = g
        
        # === STEP 5: STRICT STOKES SOLVER (No Fallback) ===
        stokes = op.algorithms.StokesFlow(network=pn, phase=water)
        stokes.set_value_BC(pores=inlet_pores, values=DELTA_P)
        stokes.set_value_BC(pores=outlet_pores, values=0.0)
        
        # Try robust solvers
        solver_success = False
        for solver_type in ['spsolve', 'cg', 'bicgstab']:
            try:
                stokes.settings['solver_family'] = 'scipy'
                stokes.settings['solver_type'] = solver_type
                stokes.settings['solver_tol'] = 1e-6
                stokes.settings['solver_maxiter'] = 5000
                stokes.run()
                solver_success = True
                break
            except Exception as e:
                continue
        
        if not solver_success:
            print(f"  ❌ All solvers failed - REJECTING sample (no fallback)")
            return None
        
        # === STEP 6: Calculate Permeability ===
        rate = stokes.rate(pores=outlet_pores)
        
        if isinstance(rate, np.ndarray):
            Q = np.sum(np.abs(rate))
        else:
            Q = abs(rate)
        
        if Q < 1e-30:
            print(f"  ❌ Zero flow rate")
            return None
        
        # Darcy's Law
        K = (Q * VISCOSITY * L) / (A * DELTA_P)
        
        print(f"  ✅ PHYSICS SUCCESS: K = {K:.2e} m² (trimmed {trim_pct:.1f}%)")
        return float(K)
    
    except Exception as e:
        # STRICT MODE: No fallback
        print(f"  ❌ Physics failed: {type(e).__name__} - REJECTING sample")
        return None


def get_permeability_geometric(network, shape, porosity=None):
    """
    Fallback permeability estimation using Kozeny-Carman equation
    
    Used when Stokes flow fails (disconnected networks)
    K ≈ (porosity³ / (1-porosity)²) × (d_pore² / 180)
    
    Parameters:
    -----------
    network : dict
        SNOW network dictionary
    shape : tuple
        3D dimensions
    porosity : float, optional
        Porosity value (calculated if not provided)
        
    Returns:
    --------
    float : Estimated permeability in m²
    """
    try:
        # Calculate mean pore diameter
        mean_diameter = np.mean(network['pore.diameter'])
        
        # Estimate porosity from network if not provided
        if porosity is None:
            total_pore_volume = np.sum(network['pore.volume'])
            sample_volume = shape[0] * shape[1] * shape[2] * (1e-6)**3
            porosity = total_pore_volume / sample_volume
        
        # Kozeny-Carman equation
        if porosity >= 0.99:
            porosity = 0.99  # Prevent division by zero
        
        K = (porosity**3 / (1 - porosity)**2) * (mean_diameter**2 / 180)
        
        return float(K)
    
    except:
        return 1e-15  # Minimum reasonable value


if __name__ == "__main__":
    """
    Standalone test: Generate a simple synthetic network and test physics
    """
    print("="*60)
    print("🔬 PHYSICS ENGINE TEST (FIXED VERSION)")
    print("="*60)
    
    # Create a realistic test network with proper throat properties
    # This simulates what SNOW would extract
    test_network = {
        'pore.coords': np.array([
            [64e-6, 64e-6, 10e-6],   # Inlet pore
            [64e-6, 64e-6, 64e-6],   # Middle pore
            [64e-6, 64e-6, 118e-6]   # Outlet pore
        ]),
        'pore.diameter': np.array([5e-6, 5e-6, 5e-6]),
        'pore.volume': np.array([65e-18, 65e-18, 65e-18]),
        'throat.conns': np.array([[0, 1], [1, 2]]),
        'throat.diameter': np.array([3e-6, 3e-6]),
        'throat.length': np.array([50e-6, 50e-6])  # CRITICAL: Added throat lengths
    }
    
    test_shape = [128, 128, 128]
    
    print(f"Test Network:")
    print(f"  Pores: {test_network['pore.coords'].shape[0]}")
    print(f"  Throats: {test_network['throat.conns'].shape[0]}")
    print(f"  Shape: {test_shape}")
    print(f"  Throat Diameters: {test_network['throat.diameter']}")
    print(f"  Throat Lengths: {test_network['throat.length']}")
    
    K = get_permeability(test_network, test_shape)
    
    print(f"\n📊 Results:")
    print(f"  Calculated Permeability: {K:.6e} m²")
    
    if K > 0:
        print("✅ Physics engine working correctly!")
        print("   Non-zero permeability indicates proper flow calculation.")
    else:
        print("❌ Zero permeability - check network connectivity or conductance")
    
    print("="*60)
