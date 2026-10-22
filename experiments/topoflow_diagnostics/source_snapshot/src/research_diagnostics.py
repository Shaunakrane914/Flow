"""Post-hoc diagnostics; these functions do not train or select a predictor."""
import numpy as np
import networkx as nx
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra, maximum_flow
from src.research_pnm import solve_permeability


def topology_masks(net):
    conns = np.asarray(net['throat.conns'], dtype=int)
    n = len(net['pore.coords'])
    adjacency = coo_matrix((np.ones(len(conns)), conns.T), shape=(n, n)).tocsr()
    _, labels = connected_components(adjacency, directed=False)
    spanning = np.intersect1d(labels[net['pore.xmin']], labels[net['pore.xmax']])
    active = np.isin(labels, spanning)
    if not active.any():
        raise ValueError('No spanning component')
    return active, active[conns].all(1)


def network_diagnostics(net, shape, voxel):
    """Geometry-only predictors plus clearly separated flow-derived diagnostics."""
    coords = np.asarray(net['pore.coords'], dtype=float)
    conns = np.asarray(net['throat.conns'], dtype=int)
    diameter = np.asarray(net['throat.inscribed_diameter'], dtype=float)
    length = np.asarray(net['throat.direct_length'], dtype=float)
    active, active_edges = topology_masks(net)
    physical = active & ~net['pore.boundary']
    internal = np.flatnonzero(physical[conns].all(1))
    a, b = conns.T
    delta = coords[b] - coords[a]
    cosine2 = delta[:, 0] ** 2 / np.sum(delta ** 2, axis=1)
    g = diameter ** 4 / length
    starts = np.flatnonzero(net['pore.xmin'] & active)
    stops = np.flatnonzero(net['pore.xmax'] & active)
    span = abs(coords[stops, 0].mean() - coords[starts, 0].mean())
    weighted = coo_matrix((np.r_[length[active_edges], length[active_edges]],
                          (np.r_[a[active_edges], b[active_edges]], np.r_[b[active_edges], a[active_edges]])),
                         shape=(len(coords), len(coords))).tocsr()
    distance = dijkstra(weighted, directed=False, indices=starts, min_only=True)
    geodesic = float(distance[stops].min()) / span
    graph = nx.Graph()
    graph.add_nodes_from(np.flatnonzero(physical))
    graph.add_edges_from(conns[internal].tolist())
    degrees = np.asarray([degree for _, degree in graph.degree()], dtype=float)
    bridge_fraction = len(list(nx.bridges(graph))) / max(1, len(internal))
    articulation_fraction = len(list(nx.articulation_points(graph))) / max(1, len(graph))
    near_in = np.unique(conns[np.asarray(net['pore.xmin'])[conns].any(1)])
    near_out = np.unique(conns[np.asarray(net['pore.xmax'])[conns].any(1)])
    near_in = near_in[physical[near_in]]
    near_out = near_out[physical[near_out]]
    if np.intersect1d(near_in, near_out).size:
        cut = None
    else:
        n = len(coords)
        cap = len(internal) + 1
        edges = conns[internal]
        rows = np.r_[edges[:, 0], edges[:, 1], np.full(len(near_in), n), near_out]
        cols = np.r_[edges[:, 1], edges[:, 0], near_in, np.full(len(near_out), n + 1)]
        capacity = np.r_[np.ones(2 * len(edges), dtype=np.int64),
                         np.full(len(near_in) + len(near_out), cap, dtype=np.int64)]
        matrix = coo_matrix((capacity, (rows, cols)), shape=(n + 2, n + 2)).tocsr()
        cut = int(maximum_flow(matrix, n, n + 1).flow_value)
    k, diag = solve_permeability(coords, conns, diameter, length, net['pore.xmin'], net['pore.xmax'],
                                shape, voxel, return_fields=True)
    fields = diag['fields']
    actual = fields['pressure_pa']
    affine = 1 - (coords[:, 0] - coords[starts, 0].mean()) / span
    # Enforce exactly the same Dirichlet boundary values and ignore inactive edges.
    affine[net['pore.xmin']] = 1
    affine[net['pore.xmax']] = 0
    drops = affine[a] - affine[b]
    energy = float(np.sum(fields['throat_conductance'] * drops ** 2))
    optimum = diag['flow_rate_m3_s']
    return {'physical_active_pores': int(physical.sum()), 'eligible_throats': len(internal),
            'geodesic_tortuosity_lower_bound': geodesic,
            'axial_alignment_conductance_weighted': float(np.average(cosine2[internal], weights=g[internal])),
            'degree_cv': float(degrees.std() / degrees.mean()),
            'throat_diameter_cv': float(diameter[internal].std() / diameter[internal].mean()),
            'log10_conductance_std': float(np.log10(g[internal]).std()),
            'internal_bridge_fraction': bridge_fraction, 'articulation_fraction': articulation_fraction,
            'internal_min_edge_cut': cut,
            'min_cut_per_eligible_throat': cut / len(internal) if cut is not None else None,
            'pnm_k_m2_diagnostic_only': k,
            'flow_derived_affine_energy_excess': energy / optimum - 1,
            'flow_derived_affine_pressure_rmse': float(np.sqrt(np.mean((actual[physical] - affine[physical]) ** 2))),
            'flow_derived_boundary_dissipation_fraction': float(fields['dissipation_fraction'][~physical[conns].all(1)].sum())}


def loss_group(loss):
    """Fixed descriptive groups: thresholding outcomes never changes selection."""
    if not np.isfinite(loss):
        raise ValueError('Nonfinite outcome')
    return 'low_under_0.1pp' if loss < .001 else ('moderate_0.1_to_1pp' if loss < .01 else 'high_at_least_1pp')


def remove_boundary_nodes(net):
    """Diagnostic zero-resistance reservoir limit, not a validated boundary model."""
    conns = np.asarray(net['throat.conns'], dtype=int)
    physical = ~net['pore.boundary']
    remap = np.full(len(physical), -1, dtype=int)
    remap[physical] = np.arange(physical.sum())
    result = {}
    edge_mask = physical[conns].all(1)
    for key, value in net.items():
        if key.startswith('pore.'):
            result[key] = np.asarray(value)[physical].copy()
        elif key.startswith('throat.'):
            result[key] = np.asarray(value)[edge_mask].copy()
    result['throat.conns'] = remap[conns[edge_mask]]
    for label in ['xmin', 'xmax']:
        touching = np.unique(conns[np.asarray(net['pore.' + label])[conns].any(1)])
        neighbors = remap[touching[physical[touching]]]
        result['pore.' + label] = np.isin(np.arange(physical.sum()), neighbors)
    if np.any(result['pore.xmin'] & result['pore.xmax']):
        raise ValueError('Reservoir contraction produces overlapping boundaries')
    return result


def solve_given_conductances(net, conductance, shape, voxel):
    """Adapter to the frozen solver: equivalent diameters reproduce supplied g."""
    g = np.asarray(conductance, dtype=float)
    length = np.asarray(net['throat.direct_length'], dtype=float)
    if np.any(g <= 0) or not np.isfinite(g).all():
        raise ValueError('Nonpositive or nonfinite conductance')
    effective_diameter = (128 * .001 * length * g / np.pi) ** .25
    return solve_permeability(net['pore.coords'], net['throat.conns'], effective_diameter, length,
                              net['pore.xmin'], net['pore.xmax'], shape, voxel, return_fields=True)


def openpnm_crosscheck(net, conductance, shape, voxel):
    """Independent OpenPNM assembly with identical g, domain and boundary values."""
    import openpnm as op
    active, edges = topology_masks(net)
    remap = np.full(len(active), -1, dtype=int)
    remap[active] = np.arange(active.sum())
    pn = op.network.Network(coords=net['pore.coords'][active], conns=remap[net['throat.conns'][edges]])
    phase = op.phase.Phase(network=pn)
    phase['throat.hydraulic_conductance'] = np.asarray(conductance)[edges]
    flow = op.algorithms.StokesFlow(network=pn, phase=phase)
    inlet = np.flatnonzero(net['pore.xmin'][active])
    outlet = np.flatnonzero(net['pore.xmax'][active])
    flow.set_value_BC(pores=inlet, values=1.)
    flow.set_value_BC(pores=outlet, values=0.)
    flow.run(solver=op.solvers.ScipySpsolve())
    rate = float(flow.rate(pores=outlet, mode='group')[0])
    size = np.asarray(shape) * voxel
    value = abs(rate) * .001 * size[0] / np.prod(size[1:])
    op.Workspace().close_project(pn.project)
    return value


def parse_native_network(files):
    """Read original Statoil geometry, preserving endpoint/length correspondence.

    Reservoir throats are removed and adjacent physical pores receive Dirichlet
    labels, matching OpenPNM's importer convention. This changes reservoir
    resistance and is explicitly not a certified reproduction of the old code.
    """
    from io import StringIO
    text = {key: value.decode('ascii') for key, value in files.items()}
    link1 = np.loadtxt(StringIO(text['link1']), skiprows=1)
    link2 = np.loadtxt(StringIO(text['link2']))
    node2 = np.loadtxt(StringIO(text['node2']))
    lines = [line.split() for line in text['node1'].splitlines() if line.strip()]
    dimensions = np.asarray(lines[0][1:4], dtype=float)
    n = int(lines[0][0])
    nodes = np.asarray([[float(v) for v in row[:4]] for row in lines[1:]])
    if len(nodes) != n or not np.array_equal(link1[:, 0], link2[:, 0]):
        raise ValueError('Inconsistent original network records')
    np.testing.assert_array_equal(nodes[:, 0], np.arange(1, n + 1))
    np.testing.assert_array_equal(node2[:, 0], np.arange(1, n + 1))
    np.testing.assert_array_equal(link1[:, 1:3], link2[:, 1:3])
    raw_conns = link1[:, 1:3].astype(int)
    keep = (raw_conns > 0).all(1)
    original = raw_conns[keep] - 1
    swapped = original[:, 0] > original[:, 1]
    lengths = link2[keep][:, [3, 5, 4]].copy()  # pore1, throat, pore2
    lengths[swapped] = lengths[swapped][:, ::-1]
    conns = np.sort(original, axis=1)
    result = {'pore.coords': nodes[:, 1:4], 'throat.conns': conns,
              'pore.radius': node2[:, 2], 'pore.shape_factor': node2[:, 3],
              'throat.radius': link1[keep, 3], 'throat.shape_factor': link1[keep, 4],
              'throat.direct_length': link1[keep, 5], 'throat.conduit_lengths': lengths,
              'pore.xmin': np.zeros(n, dtype=bool), 'pore.xmax': np.zeros(n, dtype=bool),
              'pore.boundary': np.zeros(n, dtype=bool)}
    for reservoir, label in [(-1, 'xmin'), (0, 'xmax')]:
        boundary = raw_conns[(raw_conns == reservoir).any(1)]
        physical = np.unique(boundary[boundary > 0]) - 1
        result['pore.' + label][physical] = True
    if min(dimensions) <= 0 or np.any(lengths < 0):
        raise ValueError('Invalid original geometry')
    return result, dimensions, {'physical_pores': n, 'internal_throats': len(conns),
                               'removed_reservoir_throats': int((~keep).sum()),
                               'endpoint_lengths_reordered_with_connections': int(swapped.sum())}


def native_conductances(net):
    """Fixed circular and shape-factor closures on the same original network."""
    import openpnm as op
    conns = net['throat.conns']
    radius = net['throat.radius']
    lengths = net['throat.conduit_lengths']
    pore_g_per_length = np.pi * net['pore.radius'] ** 4 / (8 * .001)
    throat_g_per_length = np.pi * radius ** 4 / (8 * .001)
    circular_series = 1 / (lengths[:, 0] / pore_g_per_length[conns[:, 0]]
                            + lengths[:, 1] / throat_g_per_length
                            + lengths[:, 2] / pore_g_per_length[conns[:, 1]])
    pn = op.network.Network(coords=net['pore.coords'], conns=conns)
    pn['pore.shape_factor'] = net['pore.shape_factor']
    pn['throat.shape_factor'] = net['throat.shape_factor']
    pn['pore.area'] = net['pore.radius'] ** 2 / (4 * net['pore.shape_factor'])
    pn['throat.cross_sectional_area'] = radius ** 2 / (4 * net['throat.shape_factor'])
    pn['throat.conduit_lengths'] = lengths
    phase = op.phase.Phase(network=pn)
    phase['pore.viscosity'] = .001
    phase['throat.viscosity'] = .001
    shaped = op.models.physics.hydraulic_conductance.valvatne_blunt(phase)
    op.Workspace().close_project(pn.project)
    return {'circular_total_length': throat_g_per_length / net['throat.direct_length'],
            'circular_pore_throat_series': circular_series, 'valvatne_blunt_shape_factor': shaped}
