"""Explicit cylindrical-throat PNM reference for pilot data, not voxel-resolved DNS."""
from time import perf_counter
import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import spsolve


def solve_permeability(coords, conns, diameter, length, inlet, outlet, shape,
                       voxel_size, flow_axis=0, viscosity=.001, pressure_drop=1., return_fields=False):
    """Solve mass conservation with Hagen–Poiseuille conductances and fixed pressures.

    Geometry is supplied in SI units. Inlet/outlet masks are supplied explicitly;
    disconnected clusters are excluded from the solve, not given artificial conductance.
    """
    coords, conns = np.asarray(coords), np.asarray(conns, dtype=int)
    diameter, length = np.asarray(diameter), np.asarray(length)
    inlet, outlet = np.asarray(inlet, dtype=bool), np.asarray(outlet, dtype=bool)
    n = len(coords)
    if (coords.shape != (n, 3) or conns.ndim != 2 or conns.shape[1] != 2
            or diameter.shape != (len(conns),) or length.shape != diameter.shape
            or inlet.shape != (n,) or outlet.shape != (n,)):
        raise ValueError('Inconsistent network dimensions')
    if (not np.isfinite(coords).all() or not np.isfinite(diameter).all()
            or not np.isfinite(length).all() or np.any(diameter <= 0) or np.any(length <= 0)
            or not np.isfinite([voxel_size, viscosity, pressure_drop]).all()
            or min(voxel_size, viscosity, pressure_drop) <= 0
            or flow_axis not in (0, 1, 2) or len(shape) != 3 or min(shape) <= 0):
        raise ValueError('Invalid physical geometry or solver settings')
    if not conns.size or conns.min() < 0 or conns.max() >= n:
        raise ValueError('Invalid throat connections')
    if np.any(inlet & outlet) or not inlet.any() or not outlet.any():
        raise ValueError('Invalid boundary masks')
    adjacency = coo_matrix((np.ones(len(conns)), (conns[:, 0], conns[:, 1])), shape=(n, n))
    _, components = connected_components(adjacency, directed=False)
    spanning = np.intersect1d(components[inlet], components[outlet])
    if not len(spanning):
        raise ValueError('Non-percolating network')
    active = np.isin(components, spanning)
    keep_throats = active[conns].all(axis=1)
    g = np.pi * (diameter[keep_throats] / 2) ** 4 / (8 * viscosity * length[keep_throats])
    if not np.isfinite(g).all() or (g <= 0).any():
        raise ValueError('Invalid conductance')
    a, b = conns[keep_throats].T
    weights = coo_matrix((np.r_[g, g], (np.r_[a, b], np.r_[b, a])), shape=(n, n)).tocsr()
    laplacian = diags(np.asarray(weights.sum(axis=1)).ravel()) - weights
    boundary = active & (inlet | outlet)
    internal = np.flatnonzero(active & ~boundary)
    fixed = np.flatnonzero(boundary)
    pressure = np.zeros(n)
    pressure[inlet & active] = pressure_drop
    if len(internal):
        matrix = laplacian[internal][:, internal]
        rhs = -(laplacian[internal][:, fixed] @ pressure[fixed])
        pressure[internal] = spsolve(matrix, rhs)
    if not np.isfinite(pressure[active]).all():
        raise ValueError('Nonfinite pressure solution')
    flux = np.asarray(laplacian @ pressure)
    q_in, q_out = float(flux[inlet & active].sum()), float(-flux[outlet & active].sum())
    if min(q_in, q_out) <= 0:
        raise ValueError('Nonpositive flow')
    balance = abs(q_in - q_out) / max(q_in, q_out)
    residual = float(np.max(np.abs(flux[internal]))) / max(q_in, q_out) if len(internal) else 0.
    if balance > 1e-7 or residual > 1e-7:
        raise ValueError('Mass balance failed')
    physical_shape = np.asarray(shape) * voxel_size
    area = float(np.prod(np.delete(physical_shape, flow_axis)))
    k = q_out * viscosity * physical_shape[flow_axis] / (area * pressure_drop)
    diagnostics = {'relative_mass_imbalance': balance, 'relative_internal_residual': residual,
                      'active_pores': int(active.sum()), 'excluded_pores': int((~active).sum()),
                      'flow_rate_m3_s': q_out, 'pressure_drop_pa': pressure_drop,
                      'viscosity_pa_s': viscosity, 'flow_axis': flow_axis}
    if return_fields:
        conductance=np.zeros(len(conns));conductance[keep_throats]=g
        throat_drop=pressure[conns[:,0]]-pressure[conns[:,1]]
        dissipation=conductance*throat_drop**2
        sensitivity=dissipation/(q_out*pressure_drop)
        diagnostics['fields']={'pressure_pa':pressure,'throat_conductance':conductance,
                               'throat_flow_m3_s':conductance*throat_drop,
                               'dissipation_fraction':sensitivity,
                               'active_pores_mask':active}
    return float(k), diagnostics


def extract_pilot_graph(image, voxel_size, return_network=False):
    """SNOW2 with x boundaries; use physical pore features and document the closure."""
    import porespy as ps
    import torch
    from torch_geometric.data import Data
    if image.ndim != 3 or image.dtype != bool:
        raise ValueError('Expected an explicitly binarized boolean pore image')
    started = perf_counter()
    snow = ps.networks.snow2(image, voxel_size=voxel_size, boundary_width=[3, 0, 0],
                            parallel_kw={'divs': 1}, sigma=.4, r_max=4)
    snow_finished = perf_counter()
    net = snow.network
    required = ('pore.xmin', 'pore.xmax', 'pore.boundary', 'throat.inscribed_diameter', 'throat.direct_length')
    if any(key not in net for key in required):
        raise ValueError('Expected SNOW2 boundary labels and physical throat properties are missing')
    k, diagnostics = solve_permeability(net['pore.coords'], net['throat.conns'],
                                        net['throat.inscribed_diameter'], net['throat.direct_length'],
                                        net['pore.xmin'], net['pore.xmax'], image.shape, voxel_size)
    solve_finished = perf_counter()
    physical = ~net['pore.boundary']
    diameter, volume = net['pore.equivalent_diameter'][physical], net['pore.volume'][physical]
    if len(diameter) < 2 or (diameter <= 0).any() or (volume <= 0).any():
        raise ValueError('Invalid physical pore geometry')
    remap = np.full(len(physical), -1, dtype=int)
    remap[physical] = np.arange(physical.sum())
    conns = net['throat.conns']
    conns = remap[conns[physical[conns].all(axis=1)]]
    if not conns.size:
        raise ValueError('No physical pore connections')
    edges = np.r_[conns, conns[:, ::-1]].T.copy()
    graph = Data(x=torch.tensor(np.column_stack([np.log10(diameter), np.log10(volume)]), dtype=torch.float32),
                 edge_index=torch.tensor(edges, dtype=torch.long), y=torch.tensor([np.log10(k)], dtype=torch.float32))
    diagnostics.update(node_diameter='equivalent diameter', throat_diameter='inscribed diameter',
                       throat_length='direct center distance including boundary conduits',
                       boundary_nodes_in_gnn=False, sigma=.4, r_max=4,
                       reference_limitation='Cylindrical-throat PNM approximation; boundary-conduit geometry is not DNS validated')
    finished = perf_counter()
    diagnostics['timing_seconds'] = {'snow2': snow_finished - started,
                                     'pnm_solve': solve_finished - snow_finished,
                                     'graph_conversion': finished - solve_finished,
                                     'total_extraction_and_reference': finished - started}
    if return_network:
        return graph, diagnostics, net
    return graph, diagnostics
