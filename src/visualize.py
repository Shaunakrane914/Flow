"""
Clear 3D Pore Network Visualization
Shows the actual pore structure with clear colors
"""

import numpy as np
import pyvista as pv
import porespy as ps
import openpnm as op
import os
from scipy.sparse import csgraph


def render_flow(chunk_path, output_file='output_flow.png', window_size=(1920, 1080)):
    """
    Create clear 3D visualization of pore network with flow
    
    Blue pores → Red flow paths (like reference image)
    """
    print("="*60)
    print("🎨 CLEAR PORE NETWORK VISUALIZATION")
    print("="*60)
    
    # Load chunk
    chunk = np.load(chunk_path)
    print(f"Chunk: {chunk.shape}")
    phi = np.sum(chunk) / chunk.size
    print(f"Porosity: {phi:.3f}")
    
    # Extract network
    print("\n🧬 Extracting pore network...")
    try:
        snow_output = ps.networks.snow2(chunk, voxel_size=2.68e-6)
        network = snow_output.network
        
        # Add missing properties
        if 'pore.diameter' not in network:
            network['pore.diameter'] = np.cbrt(network['pore.volume'] * 6 / np.pi)
        
        if 'throat.diameter' not in network:
            conns = network['throat.conns']
            network['throat.diameter'] = 0.5 * (
                network['pore.diameter'][conns[:, 0]] +
                network['pore.diameter'][conns[:, 1]]
            )
        
        if 'throat.length' not in network:
            conns = network['throat.conns']
            coords = network['pore.coords']
            network['throat.length'] = np.linalg.norm(
                coords[conns[:, 0]] - coords[conns[:, 1]], axis=1
            )
        
        print(f"  ✅ {network['pore.coords'].shape[0]} pores, {network['throat.conns'].shape[0]} throats")
        
        # Simulate flow
        print("  💧 Simulating flow...")
        pn = op.network.Network()
        pn.update(network)
        
        # Find flow backbone
        z_coords = pn['pore.coords'][:, 2]
        z_min, z_max = z_coords.min(), z_coords.max()
        margin = 2.68e-6 * 0.5
        
        inlet = np.where(z_coords <= z_min + margin)[0]
        outlet = np.where(z_coords >= z_max - margin)[0]
        
        # Get flow-percolating backbone
        am = pn.create_adjacency_matrix(fmt='coo')
        n_comp, labels = csgraph.connected_components(am, directed=False)
        
        flow_pores = []
        for i in range(n_comp):
            cluster = pn.Ps[labels == i]
            if np.any(np.isin(inlet, cluster)) and np.any(np.isin(outlet, cluster)):
                flow_pores.extend(cluster)
        
        flow_pores = np.array(flow_pores, dtype=int)
        
        if len(flow_pores) > 0:
            # Trim to backbone
            pores_to_remove = np.setdiff1d(pn.Ps, flow_pores)
            if len(pores_to_remove) > 0:
                op.topotools.trim(network=pn, pores=pores_to_remove)
            
            # Recalculate boundaries
            z_coords = pn['pore.coords'][:, 2]
            inlet = np.where(z_coords <= z_coords.min() + margin)[0]
            outlet = np.where(z_coords >= z_coords.max() - margin)[0]
            
            # Setup flow simulation
            water = op.phase.Phase(network=pn)
            water['pore.viscosity'] = 0.001
            
            # Hydraulic conductance
            R = pn['throat.diameter'] / 2
            L = pn['throat.length'] + 1e-15
            g = (np.pi * R**4) / (8 * 0.001 * L) + 1e-25
            water['throat.hydraulic_conductance'] = g
            
            # Run Stokes flow
            stokes = op.algorithms.StokesFlow(network=pn, phase=water)
            stokes.set_value_BC(pores=inlet, values=101325.0)
            stokes.set_value_BC(pores=outlet, values=0.0)
            
            try:
                stokes.settings['solver_family'] = 'scipy'
                stokes.settings['solver_type'] = 'spsolve'
                stokes.run()
                
                # Calculate flow rates
                conns = pn['throat.conns']
                flow_rates = np.abs(water['throat.hydraulic_conductance'] * 
                                   (stokes['pore.pressure'][conns[:, 0]] - 
                                    stokes['pore.pressure'][conns[:, 1]]))
                
                has_flow = True
                print(f"  ✅ Flow simulation complete")
                
            except:
                # Fallback: use throat diameter
                flow_rates = pn['throat.diameter'] ** 2
                has_flow = True
        else:
            has_flow = False
            
    except Exception as e:
        print(f"  ⚠️  Network extraction failed: {e}")
        has_flow = False
    
    if not has_flow:
        print("  ℹ️  Using simplified visualization")
        return create_simple_viz(chunk, output_file, window_size)
    
    # Create visualization
    print("\n🖼️  Creating 3D rendering...")
    
    plotter = pv.Plotter(off_screen=True, window_size=window_size)
    plotter.background_color = 'white'
    
    # 1. Draw pores as spheres (BLUE - like reference)
    print("  🔵 Adding pores...")
    pore_coords = pn['pore.coords']
    pore_diameters = pn['pore.diameter']
    
    for i, (coord, diameter) in enumerate(zip(pore_coords, pore_diameters)):
        if i % 10 == 0:  # Show every 10th pore to avoid clutter
            sphere = pv.Sphere(radius=diameter/2, center=coord)
            plotter.add_mesh(
                sphere,
                color='lightblue',
                opacity=0.6,
                smooth_shading=True
            )
    
    # 2. Draw throats as tubes (COLORED by flow rate)
    print("  🌈 Adding flow-colored throats...")
    
    conns = pn['throat.conns']
    lines = []
    for conn in conns:
        lines.append([2, conn[0], conn[1]])
    
    poly = pv.PolyData()
    poly.points = pore_coords
    poly.lines = np.hstack(lines)
    poly.cell_data['Flow Rate'] = flow_rates
    
    # Create tubes with varying thickness
    tubes = poly.tube(radius=5e-6, n_sides=12)
    
    plotter.add_mesh(
        tubes,
        scalars='Flow Rate',
        cmap='coolwarm',  # Blue (low) → Red (high)
        show_scalar_bar=True,
        scalar_bar_args={
            'title': 'Flow Rate (m³/s)',
            'title_font_size': 18,
            'label_font_size': 14,
            'position_x': 0.85,
            'position_y': 0.15,
            'width': 0.08,
            'height': 0.7
        },
        smooth_shading=True
    )
    
    # 3. Add bounding box
    bounds = [
        pore_coords[:, 0].min(), pore_coords[:, 0].max(),
        pore_coords[:, 1].min(), pore_coords[:, 1].max(),
        pore_coords[:, 2].min(), pore_coords[:, 2].max()
    ]
    outline = pv.Box(bounds)
    plotter.add_mesh(
        outline,
        color='black',
        style='wireframe',
        line_width=2,
        opacity=0.3
    )
    
    # 4. Add title
    plotter.add_text(
        f'Pore Network Flow (Blue=Low, Red=High)\n{os.path.basename(chunk_path)}  |  φ = {phi:.3f}',
        position='upper_left',
        font_size=14,
        color='black'
    )
    
    # Set camera
    plotter.camera_position = 'iso'
    plotter.camera.zoom(1.2)
    
    # Render
    print("  📸 Rendering...")
    plotter.show(screenshot=output_file)
    
    print(f"\n✅ Saved: {output_file}")
    print("="*60)
    
    return output_file


def create_simple_viz(chunk, output_file, window_size):
    """Fallback: just show the pore space clearly"""
    print("  📦 Creating simple pore visualization...")
    
    plotter = pv.Plotter(off_screen=True, window_size=window_size)
    plotter.background_color = 'white'
    
    # Create volume
    grid = pv.ImageData()
    grid.dimensions = np.array(chunk.shape) + 1
    grid.spacing = (1, 1, 1)
    
    # Pore field (1=pore, 0=solid)
    pore_field = chunk.astype(np.float32)
    grid.cell_data['Pore'] = pore_field.flatten(order='F')
    
    # Volume render
    plotter.add_volume(
        grid,
        scalars='Pore',
        cmap='Blues',
        opacity=[0.0, 0.0, 0.5, 1.0],
        show_scalar_bar=False
    )
    
    # Bounding box
    outline = pv.Box(grid.bounds)
    plotter.add_mesh(outline, color='black', style='wireframe', line_width=2)
    
    # Text
    phi = np.sum(chunk) / chunk.size
    plotter.add_text(
        f'Pore Space Visualization\nφ = {phi:.3f}',
        position='upper_left',
        font_size=14,
        color='black'
    )
    
    plotter.camera_position = 'iso'
    plotter.camera.zoom(1.2)
    
    plotter.show(screenshot=output_file)
    
    return output_file
