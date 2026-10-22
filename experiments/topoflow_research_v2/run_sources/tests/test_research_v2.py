"""Scientific invariants for the nested evaluation and dimensional normalization."""
import unittest
import numpy as np
import torch

from tests.test_experiments import fixture_manifest
from pathlib import Path
import tempfile
from src.experiment_data import load_manifest
from src.research_evaluation import TrainTransform, parent_weights, parent_macro_mse
from scripts.run_research_v2 import fit_neural, predict, CONFIGS


class ResearchEvaluationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_physical_scaling_preserves_dimensionless_inputs_and_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, _ = fixture_manifest(Path(tmp))
            _, records, graphs = load_manifest(path)
            transform = TrainTransform.fit(records, graphs, list(range(6)), True)
            original = transform.apply(records[8], graphs[8])
            graph = graphs[8].clone()
            graph.x += torch.tensor([1., 3.])  # tenfold geometry scaling
            record = dict(records[8], true_log10_k_m2=records[8]['true_log10_k_m2'] + 2.)
            changed = transform.apply(record, graph)
            # Volume dimension in record descriptors must scale with the geometry.
            record['descriptors'] = list(record['descriptors'])
            record['descriptors'][1] += 1.
            record['descriptors'][3] += 3.
            changed = transform.apply(record, graph)
            torch.testing.assert_close(original.x, changed.x, atol=3e-4, rtol=1e-4)
            torch.testing.assert_close(original.geometry, changed.geometry, atol=3e-4, rtol=1e-4)
            torch.testing.assert_close(original.y, changed.y, atol=3e-5, rtol=1e-4)
            self.assertAlmostEqual((changed.offset-original.offset).item(), 2., places=5)

    def test_parent_balance_is_independent_of_crop_count(self):
        records = [{'parent_id':'a','true_log10_k_m2':0.}] * 10 + [
            {'parent_id':'b','true_log10_k_m2':0.}]
        weights = parent_weights(records, list(range(11)))
        self.assertAlmostEqual(weights[:10].sum(), weights[10])
        self.assertAlmostEqual(parent_macro_mse(records, list(range(11)), [0.]*10+[2.]), 2.)

    def test_outer_labels_and_features_cannot_affect_training_or_normalization(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, _ = fixture_manifest(Path(tmp))
            _, records, graphs = load_manifest(path)
            train, validation = list(range(6)), [6,7]
            a, ta, ia = fit_neural('gat_physics', records, graphs, train, validation,
                                   17, CONFIGS[0], 3, 2)
            for i in range(8,12):
                records[i]['true_log10_k_m2'] += 20.
                graphs[i].x *= 3
            b, tb, ib = fit_neural('gat_physics', records, graphs, train, validation,
                                   17, CONFIGS[0], 3, 2)
            self.assertEqual(ia,ib)
            np.testing.assert_array_equal(ta.node_mean,tb.node_mean)
            self.assertEqual(ta.target_mean,tb.target_mean)
            for key in a.state_dict():
                torch.testing.assert_close(a.state_dict()[key],b.state_dict()[key],rtol=0,atol=0)


if __name__=='__main__':unittest.main()
