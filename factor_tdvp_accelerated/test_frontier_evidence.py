"""Reject invalid timing/error evidence in repeated benchmark frontiers."""
import unittest
from factor_tdvp_accelerated.repeated_scaling import repeated_frontier


class FrontierEvidenceTests(unittest.TestCase):
    def rows(self, **overrides):
        return [dict(config_id='case', family='D', n=3, method='matrix_free',
                     total=.1, steps=5, requested_repeats=3, repeat=i,
                     status='completed', global_cp_error=1e-8,
                     total_seconds=float(i), **overrides) for i in (1, 2, 3)]

    def assert_rejected(self, rows):
        self.assertFalse(repeated_frontier(rows, 1e-7)[0]['attained'])

    def test_nonfinite_negative_or_missing_accuracy_is_rejected(self):
        for invalid in (float('nan'), float('inf'), -float('inf'), -1., None):
            with self.subTest(invalid=invalid):
                rows = self.rows()
                rows[1]['global_cp_error'] = invalid
                self.assert_rejected(rows)

    def test_invalid_actual_is_not_hidden_by_a_good_bound(self):
        rows = self.rows()
        rows[1].update(global_cp_error=float('nan'), global_cp_bound=1e-9)
        self.assert_rejected(rows)

    def test_nonfinite_or_nonpositive_runtime_is_rejected(self):
        for invalid in (float('nan'), float('inf'), -float('inf'), -1., 0., None):
            with self.subTest(invalid=invalid):
                rows = self.rows()
                rows[1]['total_seconds'] = invalid
                self.assert_rejected(rows)

    def test_missing_requested_repeat_and_duplicate_repeat_are_rejected(self):
        rows = self.rows()
        for row in rows:
            row['requested_repeats'] = 4
        self.assert_rejected(rows)
        rows = self.rows()
        rows[-1]['repeat'] = 2
        self.assert_rejected(rows)

    def test_mixed_evidence_requires_every_repeat_to_qualify(self):
        rows = self.rows()
        rows[1].update(global_cp_error=None, global_cp_bound=1e-5)
        self.assert_rejected(rows)
        rows[1]['global_cp_bound'] = 2e-8
        result = repeated_frontier(rows, 1e-7)[0]
        self.assertTrue(result['attained'])
        self.assertEqual(result['evidence'], 'mixed_measured_and_bound')
        self.assertEqual(result['median_seconds'], 2.)

    def test_different_physical_endpoints_are_separate_comparisons(self):
        rows = self.rows()
        later = self.rows()
        for row in later:
            row.update(config_id='later', total=.2, total_seconds=10*row['total_seconds'])
        result = repeated_frontier(rows + later, 1e-7)
        self.assertEqual(len(result), 2)
        self.assertEqual({row['total'] for row in result}, {.1, .2})
        self.assertTrue(all(row['attained'] for row in result))


if __name__ == '__main__':
    unittest.main()
