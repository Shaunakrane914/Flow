"""Guard blind geometry selection and independently check exact PNM updates."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch
from src.research_pnm import solve_permeability
from src.throat_discovery import geometry_graph, ranked_selection, exact_response


def toy_network():
    # A spanning series path plus a geometrically large disconnected component.
    return {'pore.coords': np.array([[0,0,0], [1,0,0], [2,0,0], [3,0,0], [1,2,0], [2,2,0]], dtype=float) * 1e-4,
            'throat.conns': np.array([[0,1], [1,2], [2,3], [4,5]]),
            'throat.inscribed_diameter': np.array([1, 1.5, 1, 10]) * 1e-5,
            'throat.direct_length': np.ones(4) * 1e-4,
            'pore.boundary': np.array([1,0,0,1,0,0], dtype=bool),
            'pore.xmin': np.array([1,0,0,0,0,0], dtype=bool),
            'pore.xmax': np.array([0,0,0,1,0,0], dtype=bool),
            'pore.equivalent_diameter': np.ones(6) * 2e-5,
            'pore.volume': np.ones(6) * 1e-15}


class DiscoveryTests(unittest.TestCase):
    def test_inputs_ignore_hydraulic_fields_and_exclude_disconnected_edges(self):
        net = toy_network()
        first, eligible = geometry_graph(net, (30, 30, 30), 1e-5)
        np.testing.assert_array_equal(eligible, [1])
        net['pore.pressure'] = np.full(6, np.nan)
        net['throat.flow'] = np.full(4, 1e30)
        second, _ = geometry_graph(net, (30, 30, 30), 1e-5)
        for key, value in first:
            torch.testing.assert_close(value, second[key], atol=0, rtol=0)
        self.assertEqual(first.x.shape, (6, 7))
        self.assertEqual(first.throat_attr.shape, (4, 6))

    def test_exact_update_matches_fresh_solves_for_all_edges_and_doses(self):
        net = toy_network()
        base, s, leverage, response, _ = exact_response(net, (30, 30, 30), 1e-5)
        self.assertEqual(s[-1], 0)
        self.assertEqual(leverage[-1], 0)
        for edge in range(4):
            for j, dose in enumerate([.1, .25, .5]):
                diameter = net['throat.inscribed_diameter'].copy()
                diameter[edge] *= 1 - dose
                changed, _ = solve_permeability(net['pore.coords'], net['throat.conns'], diameter,
                    net['throat.direct_length'], net['pore.xmin'], net['pore.xmax'], (30, 30, 30), 1e-5)
                self.assertAlmostEqual(response[edge, j], 1 - changed / base, places=10)

    def test_constant_rankings_have_reproducible_disjoint_controls(self):
        groups = ranked_selection(np.ones(12), [1,3,5,7,9,11])
        np.testing.assert_array_equal(groups['top'], [1,3,5])
        np.testing.assert_array_equal(groups['low'], [7,9,11])

    def test_tampered_selection_is_rejected_before_outcome_solver(self):
        from scripts import run_throat_discovery as runner
        with TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'protocol.json').write_text('{}')
            (root / 'selection_seal.json').write_text(json.dumps({'protocol_sha256': 'tampered', 'files': {}}))
            with patch.object(runner, 'OUT', root), patch.object(runner, 'checked_protocol', return_value={}), \
                 patch('src.throat_discovery.exact_response') as flow:
                with self.assertRaisesRegex(ValueError, 'Protocol changed'):
                    runner.evaluate()
                flow.assert_not_called()
            self.assertFalse((root / 'evaluation_started.json').exists())


if __name__ == '__main__':
    unittest.main()
