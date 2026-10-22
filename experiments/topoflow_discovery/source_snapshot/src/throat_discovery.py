"""Separate image-derived inputs from flow-derived outcomes for blind discovery."""
import numpy as np
import torch
from scipy.sparse import coo_matrix, diags
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import splu
from torch_geometric.data import Data


def geometry_graph(net, shape, voxel):
    """Use the pilot's exact feature recipe, without reading any hydraulic labels."""
    coords = np.asarray(net['pore.coords'], dtype=float)
    conns = np.asarray(net['throat.conns'], dtype=int)
    diameter = np.asarray(net['throat.inscribed_diameter'], dtype=float)
    length = np.asarray(net['throat.direct_length'], dtype=float)
    physical = ~np.asarray(net['pore.boundary'], dtype=bool)
    pore_d = np.asarray(net['pore.equivalent_diameter'], dtype=float)
    volume = np.asarray(net['pore.volume'], dtype=float)
    if physical.sum() < 2 or np.any(pore_d[physical] <= 0):
        raise ValueError('Invalid physical pores')
    if np.any(diameter <= 0) or np.any(length <= 0):
        raise ValueError('Invalid throat geometry')
    scale = 10 ** np.log10(pore_d[physical]).mean()
    node = np.column_stack([
        np.log10(np.maximum(pore_d, .01 * voxel) / scale),
        np.log10(np.maximum(volume, 1e-6 * voxel ** 3) / scale ** 3),
        coords / (np.asarray(shape) * voxel), net['pore.xmin'], net['pore.xmax']])
    a, b = conns.T
    difference = np.abs(coords[a] - coords[b])
    direction = difference / np.maximum(np.linalg.norm(difference, axis=1, keepdims=True), voxel * 1e-4)
    edge = np.column_stack([np.log10(diameter / scale), np.log10(length / scale),
                            np.log10((diameter / scale) ** 4 / (length / scale)), direction])
    graph = Data(x=torch.tensor(node, dtype=torch.float32),
                 edge_index=torch.tensor(np.r_[conns, conns[:, ::-1]].T.copy(), dtype=torch.long),
                 edge_attr=torch.tensor(np.r_[edge, edge], dtype=torch.float32),
                 throat_index=torch.tensor(conns.T.copy(), dtype=torch.long),
                 throat_attr=torch.tensor(edge, dtype=torch.float32))
    if not torch.isfinite(graph.x).all() or not torch.isfinite(graph.edge_attr).all():
        raise ValueError('Nonfinite geometry features')
    adjacency = coo_matrix((np.ones(len(conns)), (a, b)), shape=(len(coords), len(coords)))
    _, labels = connected_components(adjacency, directed=False)
    spanning = np.intersect1d(labels[np.asarray(net['pore.xmin'], dtype=bool)],
                             labels[np.asarray(net['pore.xmax'], dtype=bool)])
    if not len(spanning):
        raise ValueError('No topology-spanning component')
    # This mask depends only on connectivity and boundary membership, never flow.
    eligible = np.flatnonzero(physical[conns].all(1) & np.isin(labels[a], spanning))
    return graph, eligible


def ranked_selection(scores, eligible, k=3):
    """Stable edge-index tie breaking; top and bottom are disjoint."""
    eligible = np.asarray(eligible, dtype=int)
    if len(eligible) < 2 * k:
        raise ValueError('Too few eligible throats')
    values = np.asarray(scores)
    if not np.isfinite(values[eligible]).all():
        raise ValueError('Nonfinite ranking')
    top = eligible[np.lexsort((eligible, -values[eligible]))[:k]]
    bottom = eligible[np.lexsort((eligible, values[eligible]))[:k]]
    # Constant scores can otherwise select identical indices for both groups.
    bottom = np.asarray([e for e in eligible[np.lexsort((eligible, values[eligible]))]
                         if e not in set(top)], dtype=int)[:k]
    return {'top': top, 'low': bottom}


def analytic_scores(net):
    """Two solver-free controls; axial energy assumes an affine pressure field."""
    coords = np.asarray(net['pore.coords'], dtype=float)
    a, b = np.asarray(net['throat.conns']).T
    conductance = np.asarray(net['throat.inscribed_diameter'], dtype=float) ** 4 / net['throat.direct_length']
    axial_energy = conductance * (coords[a, 0] - coords[b, 0]) ** 2
    return {'conductance': conductance / conductance.sum(),
            'affine_pressure': axial_energy / axial_energy.sum()}


def exact_response(net, shape, voxel, doses=(.1, .25, .5)):
    """Solver-based oracle, including a base solve and a Dirichlet factorization.

    The finite-change identity is exact only under the declared linear PNM.
    This routine is forbidden during selection; the runner enforces two phases.
    """
    from src.research_pnm import solve_permeability
    coords = np.asarray(net['pore.coords'], dtype=float)
    conns = np.asarray(net['throat.conns'], dtype=int)
    base, diag = solve_permeability(coords, conns, net['throat.inscribed_diameter'],
                                   net['throat.direct_length'], net['pore.xmin'], net['pore.xmax'],
                                   shape, voxel, return_fields=True)
    fields = diag.pop('fields')
    g, sensitivity = fields['throat_conductance'], fields['dissipation_fraction']
    internal = np.flatnonzero(fields['active_pores_mask'] & ~net['pore.xmin'] & ~net['pore.xmax'])
    a, b = conns.T
    weights = coo_matrix((np.r_[g, g], (np.r_[a, b], np.r_[b, a])), shape=(len(coords), len(coords))).tocsr()
    lap = diags(np.asarray(weights.sum(1)).ravel()) - weights
    leverage = np.zeros(len(conns))
    if len(internal):
        lu = splu(lap[internal][:, internal].tocsc())
        remap = np.full(len(coords), -1, dtype=int)
        remap[internal] = np.arange(len(internal))
        for start in range(0, len(conns), 128):
            stop = min(start + 128, len(conns))
            incidence = np.zeros((len(internal), stop - start))
            for sign, nodes in [(1, a[start:stop]), (-1, b[start:stop])]:
                valid = remap[nodes] >= 0
                incidence[remap[nodes[valid]], np.flatnonzero(valid)] += sign
            leverage[start:stop] = g[start:stop] * np.sum(incidence * lu.solve(incidence), axis=0)
    if abs(sensitivity.sum() - 1) > 1e-6 or np.min(leverage) < -1e-7 or np.max(sensitivity + leverage) > 1 + 1e-6:
        raise ValueError('Hydraulic identity validation failed')
    sensitivity = np.maximum(sensitivity, 0)
    leverage = np.clip(leverage, 0, 1 - sensitivity)
    response = response_array(sensitivity, leverage, doses)
    return base, sensitivity, leverage, response, diag


def response_array(sensitivity, leverage, doses=(.1, .25, .5)):
    change = (1 - np.asarray(doses)) ** 4 - 1
    return -change[None, :] * np.asarray(sensitivity)[:, None] / (1 + change[None, :] * np.asarray(leverage)[:, None])
