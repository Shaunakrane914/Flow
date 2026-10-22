"""Checks for outcome strata, boundary contraction and independent flow assembly."""
import unittest
import numpy as np
from src.research_diagnostics import (loss_group, remove_boundary_nodes,
    solve_given_conductances, openpnm_crosscheck, parse_native_network, native_conductances, network_diagnostics)


def series():
    return {'pore.coords': np.array([[0,0,0], [1,0,0], [2,0,0], [3,0,0]], dtype=float) * 1e-4,
            'throat.conns': np.array([[0,1], [1,2], [2,3]]),
            'throat.direct_length': np.ones(3) * 1e-4,
            'throat.inscribed_diameter': np.ones(3) * 1e-5,
            'pore.boundary': np.array([1,0,0,1], dtype=bool),
            'pore.xmin': np.array([1,0,0,0], dtype=bool),
            'pore.xmax': np.array([0,0,0,1], dtype=bool)}


class DiagnosticTests(unittest.TestCase):
    def test_straight_uniform_path_has_unit_tortuosity_and_exact_affine_pressure(self):
        result = network_diagnostics(series(), [30,10,10], 1e-5)
        self.assertAlmostEqual(result['geodesic_tortuosity_lower_bound'], 1, places=12)
        self.assertAlmostEqual(result['axial_alignment_conductance_weighted'], 1, places=12)
        self.assertEqual(result['degree_cv'], 0)
        self.assertEqual(result['internal_min_edge_cut'], 1)
        self.assertAlmostEqual(result['flow_derived_affine_energy_excess'], 0, places=12)

    def test_strata_have_fixed_exclusive_thresholds(self):
        self.assertEqual(loss_group(0), 'low_under_0.1pp')
        self.assertEqual(loss_group(.001), 'moderate_0.1_to_1pp')
        self.assertEqual(loss_group(.01), 'high_at_least_1pp')
        self.assertEqual(loss_group(1), 'high_at_least_1pp')
        with self.assertRaises(ValueError):
            loss_group(float('nan'))

    def test_supplied_conductance_adapter_matches_series_formula(self):
        net = series()
        conductances = np.array([1,2,3]) * 1e-12
        _, diagnostics = solve_given_conductances(net, conductances, [30,10,10], 1e-5)
        expected_flow = 1 / np.sum(1 / conductances)
        self.assertAlmostEqual(diagnostics['flow_rate_m3_s'] / expected_flow, 1., places=12)
        np.testing.assert_allclose(diagnostics['fields']['throat_conductance'], conductances, rtol=1e-12)

    def test_boundary_contraction_is_zero_boundary_resistance_limit(self):
        net = series()
        contracted = remove_boundary_nodes(net)
        np.testing.assert_array_equal(contracted['throat.conns'], [[0,1]])
        np.testing.assert_array_equal(contracted['pore.xmin'], [True, False])
        np.testing.assert_array_equal(contracted['pore.xmax'], [False, True])
        base = solve_given_conductances(net, np.ones(3) * 1e-12, [30,10,10], 1e-5)[0]
        changed = solve_given_conductances(contracted, np.ones(1) * 1e-12, [30,10,10], 1e-5)[0]
        self.assertAlmostEqual(changed / base, 3, places=12)

    def test_openpnm_assembly_matches_independent_series_answer(self):
        net = series()
        g = np.array([1,2,3]) * 1e-12
        reference = solve_given_conductances(net, g, [30,10,10], 1e-5)[0]
        independent = openpnm_crosscheck(net, g, [30,10,10], 1e-5)
        self.assertAlmostEqual(independent / reference, 1, places=12)

    def test_original_endpoint_lengths_follow_endpoint_reordering(self):
        files = {'node1': b'2 1 1 1\n1 0 0 0 1\n2 1 0 0 1\n',
                 'node2': b'1 1 0.1 0.07 0\n2 1 0.2 0.07 0\n',
                 'link1': b'3\n1 -1 1 0.01 0.07 1\n2 2 1 0.01 0.07 1\n3 2 0 0.01 0.07 1\n',
                 'link2': b'1 -1 1 0 0.1 0.9 1 0\n2 2 1 0.2 0.1 0.7 1 0\n3 2 0 0.2 0 0.8 1 0\n'}
        net, _, meta = parse_native_network(files)
        np.testing.assert_array_equal(net['throat.conns'], [[0,1]])
        np.testing.assert_allclose(net['throat.conduit_lengths'], [[.1,.7,.2]])
        self.assertEqual(meta['endpoint_lengths_reordered_with_connections'], 1)

    def test_shape_factor_closure_reduces_to_circular_uniform_tube(self):
        net = series()
        net['pore.radius'] = np.ones(4) * 5e-6
        net['throat.radius'] = np.ones(3) * 5e-6
        net['pore.shape_factor'] = np.full(4, 1 / (4 * np.pi))
        net['throat.shape_factor'] = np.full(3, 1 / (4 * np.pi))
        net['throat.conduit_lengths'] = np.tile([.25e-4, .5e-4, .25e-4], (3,1))
        values = native_conductances(net)
        for value in values.values():
            np.testing.assert_allclose(value, values['circular_total_length'], rtol=1e-12)


if __name__ == '__main__':
    unittest.main()
