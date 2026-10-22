"""Scientific invariants and end-to-end wiring; fixtures are not benchmark evidence."""
import argparse
import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch_geometric.data import Data

from src.experiment_data import decode_log, load_manifest
from src.experiment_metrics import paired_loss, paired_parent_interval, regression_metrics
from src.experiment_models import make_model
from src.research_pnm import solve_permeability
from scripts.run_experiments import run


def fixture_manifest(directory):
    samples = []
    for i in range(12):
        diameter = np.asarray([1, 1.2 + .03 * i, 2 + .2 * i]) * 1e-5
        volume = np.pi / 6 * diameter ** 3
        x = torch.tensor(np.column_stack([np.log10(diameter), np.log10(volume)]), dtype=torch.float32)
        graph = Data(x=x, edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
                     y=torch.tensor([-12 + .05 * i], dtype=torch.float32))
        path = directory / f'graph{i}.pt'
        torch.save(graph, path)
        split = 'train' if i < 6 else 'validation' if i < 8 else 'test'
        samples.append(dict(id=f'sample{i}', graph=path.name, parent_id=f'parent{i // 2}',
                            formation='fixture_A' if i % 4 < 2 else 'fixture_B', split=split,
                            porosity=.15 + .002 * i, label_source='test fixture only'))
    manifest = dict(schema_version=1, purpose='pilot',
                    encoding=dict(diameter_epsilon_m=0, volume_epsilon_m3=0, target_epsilon_m2=0),
                    samples=samples)
    path = directory / 'manifest.json'
    path.write_text(json.dumps(manifest), encoding='utf-8')
    return path, manifest


class MetricsTests(unittest.TestCase):
    def test_loss_compares_pairs_and_rejects_mismatched_counts(self):
        pred = torch.tensor([[1.], [2.]], requires_grad=True)
        loss = paired_loss(pred, torch.tensor([1., 2.]), torch.nn.MSELoss())
        self.assertEqual(loss.item(), 0.)
        loss.backward()
        self.assertTrue(torch.equal(pred.grad, torch.zeros_like(pred)))
        with self.assertRaises(ValueError):
            paired_loss(pred, torch.ones(3), torch.nn.MSELoss())

    def test_metric_spaces_and_constant_target_r2(self):
        result = regression_metrics(np.array([-12., -11.]), np.array([-12., -11.]))
        self.assertEqual(result['r2_log10_m2'], 1.)
        self.assertEqual(result['mae_m2'], 0.)
        self.assertEqual(result['factor_of_two_accuracy'], 1.)
        self.assertIsNone(regression_metrics(np.array([-12., -12.]), np.array([-12., -12.]))['r2_log10_m2'])

    def test_inverse_transform_removes_epsilon(self):
        actual = decode_log(np.log10(np.array([1e-16, 2e-16]) + 1e-15), 1e-15, 'volume')
        np.testing.assert_allclose(actual, [1e-16, 2e-16], rtol=1e-12, atol=0)
        with self.assertRaises(ValueError):
            decode_log([-16.], 1e-15, 'volume')

    def test_bootstrap_counts_parents_not_seeds_or_crops(self):
        rows = [{'parent_id': 'a', 'squared_log_error_difference': 1.}] * 10
        self.assertIsNone(paired_parent_interval(rows, 50)['ci95'])
        rows += [{'parent_id': 'b', 'squared_log_error_difference': -1.}]
        result = paired_parent_interval(rows, 50)
        self.assertEqual(result['parent_balanced_effect'], 0.)
        self.assertEqual(result['parents'], 2)


class DatasetTests(unittest.TestCase):
    def test_parent_leakage_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path, manifest = fixture_manifest(Path(directory))
            manifest['samples'][-1]['parent_id'] = manifest['samples'][0]['parent_id']
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'Parent leakage'):
                load_manifest(path)

    def test_cv_comes_from_node_sizes_and_formation_holdout_is_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            path, manifest = fixture_manifest(Path(directory))
            _, records, _ = load_manifest(path)
            diameter = np.array([1., 1.2, 2.])
            self.assertAlmostEqual(records[0]['pore_radius_cv'], diameter.std() / diameter.mean(), places=5)
            with self.assertRaisesRegex(ValueError, 'Formation holdout'):
                load_manifest(path, 'formation-holdout')

    def test_identical_tensors_with_different_file_names_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path, manifest = fixture_manifest(Path(directory))
            root = Path(directory)
            graph = torch.load(root / 'graph0.pt', weights_only=False)
            torch.save(graph, root / 'graph1.pt')
            with self.assertRaisesRegex(ValueError, 'Duplicate graph'):
                load_manifest(path)


class ModelTests(unittest.TestCase):
    def test_uniform_attention_does_not_depend_on_attention_parameters(self):
        model = make_model('uniform_gat', dropout=0).eval()
        x = torch.tensor([[-5., -15.], [-4.8, -14.9], [-4.6, -14.5]])
        edges, batch = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]), torch.zeros(3, dtype=torch.long)
        first = model(x, edges, batch)
        with torch.no_grad():
            model.gat1.att_src.fill_(1000)
            model.gat1.att_dst.fill_(-1000)
        torch.testing.assert_close(first, model(x, edges, batch))

    def test_pore_set_control_ignores_edges(self):
        model = make_model('pore_set', dropout=0).eval()
        x, batch = torch.randn(3, 2), torch.zeros(3, dtype=torch.long)
        a = model(x, torch.tensor([[0, 1], [1, 0]]), batch)
        b = model(x, torch.tensor([[0, 2], [2, 0]]), batch)
        torch.testing.assert_close(a, b)


class PhysicsTests(unittest.TestCase):
    def solve_tube(self, scale=1., viscosity=.001, pressure_drop=1.):
        coords = np.array([[0., 0., 0.], [5e-5, 0., 0.], [1e-4, 0., 0.]]) * scale
        return solve_permeability(coords, [[0, 1], [1, 2]], np.full(2, 1e-5) * scale,
                                  np.full(2, 5e-5) * scale, [True, False, False], [False, False, True],
                                  (10, 10, 10), 1e-5 * scale, viscosity=viscosity, pressure_drop=pressure_drop)

    def test_tube_matches_poiseuille_and_is_independent_of_pressure_viscosity(self):
        k, diagnostics = self.solve_tube()
        expected = np.pi * (5e-6) ** 4 / (8 * (1e-4) ** 2)
        self.assertAlmostEqual(k / expected, 1., places=10)
        changed, _ = self.solve_tube(viscosity=.002, pressure_drop=3.)
        self.assertAlmostEqual(changed / k, 1., places=10)
        self.assertLess(diagnostics['relative_mass_imbalance'], 1e-10)

    def test_permeability_scales_with_length_squared(self):
        k, _ = self.solve_tube()
        doubled, _ = self.solve_tube(scale=2.)
        self.assertAlmostEqual(doubled / k, 4., places=10)

    def test_nonpercolating_network_does_not_get_a_fallback_label(self):
        with self.assertRaisesRegex(ValueError, 'Non-percolating'):
            solve_permeability(np.zeros((4, 3)), [[0, 1], [2, 3]], [1., 1.], [1., 1.],
                               [True, False, False, False], [False, False, False, True], (4, 4, 4), 1.)


class IntegrationTests(unittest.TestCase):
    def test_full_run_and_test_labels_do_not_select_checkpoints(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, manifest = fixture_manifest(root)
            args = argparse.Namespace(manifest=str(manifest_path), output=str(root / 'first'),
                                      purpose='pilot', protocol='grouped', models=['gat', 'sage', 'uniform_gat', 'pore_set'],
                                      seeds=[7], epochs=2, patience=2, batch_size=4, learning_rate=.001,
                                      weight_decay=5e-4, dropout=0., bootstrap_repeats=20, threads=1, device='cpu')
            run(args)
            first = root / 'first'
            with (first / 'predictions.csv').open(encoding='utf-8') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 12 * 7)
            self.assertEqual(json.loads((first / 'run.json').read_text())['status'], 'complete')
            self.assertFalse(json.loads((first / 'heterogeneity.json').read_text())['threshold_estimated'])
            for sample in manifest['samples']:
                if sample['split'] == 'test':
                    p = root / sample['graph']
                    graph = torch.load(p, weights_only=False)
                    graph.y += 2
                    torch.save(graph, p)
            args.output = str(root / 'second')
            run(args)
            for name in args.models:
                a = torch.load(first / f'{name}_seed7.pth', weights_only=False)
                b = torch.load(root / 'second' / f'{name}_seed7.pth', weights_only=False)
                self.assertEqual(a['selected_epoch'], b['selected_epoch'])
                for key in a['state_dict']:
                    torch.testing.assert_close(a['state_dict'][key], b['state_dict'][key], rtol=0, atol=0)
            with self.assertRaisesRegex(ValueError, 'Output already exists'):
                run(args)


if __name__ == '__main__':
    unittest.main()
