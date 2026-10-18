import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec = importlib.util.spec_from_file_location('round1', Path(__file__).parents[2] / 'scripts/revision/round1.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

class PopulationTests(unittest.TestCase):
    def test_common_population_and_separate_coverage(self):
        x=np.array([[1.,3.,100.],[2.,4.,9.],[3.,5.,8.]])
        valid=np.array([[1,1,0],[1,0,1],[0,1,1]],dtype=bool)
        result=m.compare(x,np.array([2.,3.,4.]),valid,[0,1])
        self.assertEqual(result['n_common'],1)
        self.assertEqual(result['n_retained_all'],3)
        self.assertEqual(result['n_retained_reduced'],1)
        self.assertAlmostEqual(result['all_filtered_rmse'],0.)
        self.assertAlmostEqual(result['reduced_filtered_rmse'],0.)
        self.assertAlmostEqual(result['all_unfiltered_rmse'],98/3)
    def test_insufficient_common_population_fails(self):
        with self.assertRaisesRegex(ValueError,'common'):
            m.compare(np.ones((1,3)),np.ones(1),np.array([[1,0,1]],bool),[0,1])
    def test_scatter_is_descriptive_not_fitted(self):
        d=m.error_summary(np.array([2.,4.]),np.array([1.,1.]))
        self.assertAlmostEqual(d['bias'],2.)
        self.assertAlmostEqual(d['centered_rmse'],1.)
        self.assertAlmostEqual(d['rmse']**2,d['bias']**2+d['centered_rmse']**2)
    def test_missing_and_nonfinite_inputs_rejected(self):
        with self.assertRaises(ValueError):
            m.compare(np.array([[1.,float('nan'),3.]]),np.ones(1),np.ones((1,3),bool),[0,1])
    def test_zero_denominator_is_explicit(self):
        self.assertIsNone(m.percent_reduction(0.,1.))

if __name__=='__main__':unittest.main()
