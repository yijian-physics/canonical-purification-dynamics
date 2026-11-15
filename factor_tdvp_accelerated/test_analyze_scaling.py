"""Regression checks for scientific accuracy selection, without plotting."""
import unittest
from factor_tdvp_accelerated.analyze_scaling import summarize_local, frontier


class AccuracySelectionTests(unittest.TestCase):
    def rows(self):
        return [dict(config_id='D3', family='D', n=3, method='matrix_free', steps=5,
                     requested_repeats=3, repeat=k+1, total=.1, status='completed', directory=f'r{k}',
                     global_cp_error=1e-8, global_cp_bound=None, total_seconds=t,
                     wall_seconds=t+.2) for k,t in enumerate((1.,2.,3.))]

    def qualifies(self, rows):
        values=frontier(summarize_local(rows))
        return next(r for r in values if r['method']=='matrix_free' and r['target']==1e-7)

    def test_each_repeat_must_qualify_even_with_mixed_evidence(self):
        rows=self.rows()
        rows[2].update(global_cp_error=None,global_cp_bound=1e-4)
        self.assertFalse(self.qualifies(rows)['attained'])

    def test_nonfinite_or_negative_evidence_cannot_qualify(self):
        for value in (float('nan'),float('inf'),-1e-10,None):
            with self.subTest(value=value):
                rows=self.rows();rows[2]['global_cp_error']=value
                self.assertFalse(self.qualifies(rows)['attained'])

    def test_failed_or_missing_repeat_cannot_qualify(self):
        rows=self.rows();rows[2]['status']='timeout'
        self.assertFalse(self.qualifies(rows)['attained'])
        self.assertFalse(self.qualifies(rows[:2])['attained'])

    def test_missing_or_nonfinite_runtime_cannot_qualify(self):
        for value in (None,float('nan'),float('inf'),0.,-1.):
            rows=self.rows();rows[2]['total_seconds']=value
            self.assertFalse(self.qualifies(rows)['attained'])

    def test_duplicate_repeats_do_not_supply_three_observations(self):
        rows=self.rows();rows[2]['repeat']=2
        self.assertFalse(self.qualifies(rows)['attained'])

    def test_different_endpoints_have_separate_frontiers(self):
        a=self.rows();b=self.rows()
        for row in b:
            row.update(config_id='D3later',total=1.,total_seconds=.01)
        values=[r for r in frontier(summarize_local(a+b)) if r['method']=='matrix_free' and r['target']==1e-7]
        self.assertEqual(len(values),2)
        self.assertEqual({r['total']:r['seconds'] for r in values},{.1:2.,1.:.01})

    def test_median_cost_is_selected_and_mixed_evidence_labeled(self):
        rows=self.rows();rows[2].update(global_cp_error=None,global_cp_bound=2e-8)
        value=self.qualifies(rows)
        self.assertTrue(value['attained'])
        self.assertEqual(value['seconds'],2.)
        self.assertEqual(value['evidence'],'mixed_measured_and_bound')


if __name__=='__main__':
    unittest.main()
