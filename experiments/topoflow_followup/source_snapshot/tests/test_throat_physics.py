"""Closed-form checks for hydraulic-dissipation sensitivity and independent ablations."""
import unittest
import numpy as np
import torch
from src.research_pnm import solve_permeability
from src.research_evaluation import ScaleAwareRegressor
from src.critical_throat_model import CriticalThroatRegressor,fractional_response
from torch_geometric.data import Data,Batch


class ThroatPhysicsTests(unittest.TestCase):
    def solve(self,diameters):
        return solve_permeability(np.array([[0,0,0],[1,0,0],[2,0,0]],dtype=float)*1e-4,
             [[0,1],[1,2]],np.asarray(diameters,dtype=float),[1e-4,1e-4],
             [True,False,False],[False,False,True],(20,10,10),1e-5,return_fields=True)

    def test_dissipation_is_exact_conductance_sensitivity(self):
        base,diag=self.solve([1e-5,1e-5])
        values=diag['fields']['dissipation_fraction']
        np.testing.assert_allclose(values,[.5,.5],rtol=1e-12)
        eps=1e-5
        changed,_=self.solve([1e-5*np.exp(eps/4),1e-5])
        self.assertAlmostEqual(np.log(changed/base)/eps,.5,places=5)

    def test_constriction_matches_series_resistance(self):
        base,_=self.solve([1e-5,1e-5])
        changed,_=self.solve([.5e-5,1e-5])
        self.assertAlmostEqual(changed/base,2/17,places=12)

    def test_geometry_head_is_independent_of_physical_scaling_choice(self):
        for physics in [False,True]:
            for descriptors in [False,True]:
                model=ScaleAwareRegressor('gat',physics=physics,geometry=descriptors)
                self.assertEqual(hasattr(model,'geometry_head'),descriptors)
        torch.manual_seed(7);old=ScaleAwareRegressor('gat',physics=True)
        torch.manual_seed(7);explicit=ScaleAwareRegressor('gat',physics=True,geometry=True)
        for k in old.state_dict():
            torch.testing.assert_close(old.state_dict()[k],explicit.state_dict()[k],atol=0,rtol=0)

    def test_critical_readout_bounds_and_monotonicity(self):
        importance=torch.tensor([.01,.3,.8]);leverage=torch.tensor([.8,.5,.1])
        response=fractional_response(importance,leverage,[0,.1,.25,.5,1.])
        self.assertTrue(torch.all(response>=0) and torch.all(response<=1))
        self.assertTrue(torch.all(response[:,1:]>=response[:,:-1]))
        torch.testing.assert_close(response[:,0],torch.zeros(3))

    def test_critical_models_use_geometry_only_and_batch_throat_indices_correctly(self):
        graph=Data(x=torch.randn(3,7),edge_index=torch.tensor([[0,1,1,2],[1,0,2,1]]),
                   edge_attr=torch.randn(4,6),throat_index=torch.tensor([[0,1],[1,2]]),
                   throat_attr=torch.randn(2,6),sensitivity=torch.tensor([.4,.6]),response_target=torch.rand(2,3))
        batch=Batch.from_data_list([graph,graph.clone()])
        torch.testing.assert_close(batch.throat_index[:,2:],graph.throat_index+3)
        for architecture in ['edge_gat','edge_sage','edge_mlp']:
            model=CriticalThroatRegressor(architecture).eval()
            first=model(batch)
            batch.sensitivity=torch.randn(4)*100;batch.response_target=torch.randn(4,3)*100
            second=model(batch)
            for a,b in zip(first,second):torch.testing.assert_close(a,b,atol=0,rtol=0)
            torch.testing.assert_close(first[0][:2].sum(),torch.tensor(1.))
            torch.testing.assert_close(first[0][2:].sum(),torch.tensor(1.))


if __name__=='__main__':unittest.main()
