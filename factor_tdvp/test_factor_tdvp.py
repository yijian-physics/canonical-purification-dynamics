"""Small unit/regression tests for direct H-MPO tangent evolution.

From repository root:
    python -m unittest factor_tdvp.test_factor_tdvp -v
or:
    python factor_tdvp/test_factor_tdvp.py -v
"""
from __future__ import annotations

import unittest
import numpy as np
from scipy.linalg import expm

if __package__:
    from . import tangent_solver as ts
    from . import rank_bridge as rb
    from . import factor_tdvp as driver
else:
    import tangent_solver as ts
    import rank_bridge as rb
    import factor_tdvp as driver


I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], complex)
Y = np.array([[0, -1j], [1j, 0]], complex)
Z = np.diag([1., -1.]).astype(complex)
LOWER = np.array([[0, 1], [0, 0]], complex)


def density(h):
    x = h @ h.conj().T
    return x @ x


class FactorJacobianTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(819)

    def random_matrix(self, d):
        return self.rng.normal(size=(d, d)) + 1j*self.rng.normal(size=(d, d))

    def test_jacobian_finite_difference_and_mpo(self):
        h, k = self.random_matrix(4), self.random_matrix(4)
        h /= np.linalg.norm(h)
        k /= np.linalg.norm(k)
        exact = ts.jacobian_dense(h, k)
        eps = 2e-6
        difference = (density(h+eps*k)-density(h-eps*k))/(2*eps)
        np.testing.assert_allclose(exact, difference, rtol=2e-8, atol=2e-11)
        mpo_result = ts.matrix(ts.jacobian_mpo(ts.matrix_to_mpo(h, 2), ts.matrix_to_mpo(k, 2)))
        np.testing.assert_allclose(exact, mpo_result, rtol=2e-12, atol=2e-13)

    def test_jacobian_is_real_linear_not_complex_linear(self):
        h, k, ell = (self.random_matrix(2) for _ in range(3))
        actual = ts.jacobian_dense(h, .37*k-.21*ell)
        expected = .37*ts.jacobian_dense(h, k)-.21*ts.jacobian_dense(h, ell)
        np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-13)
        self.assertGreater(np.linalg.norm(ts.jacobian_dense(h, 1j*k)-1j*ts.jacobian_dense(h, k)), 1.)

    def test_right_unitary_gauge_is_null(self):
        h = self.random_matrix(4)
        a = self.random_matrix(4)
        omega = (a-a.conj().T)/2
        k = h@omega
        self.assertLess(np.linalg.norm(ts.jacobian_dense(h, k)), 2e-12)
        hm, km = ts.matrix_to_mpo(h, 2), ts.matrix_to_mpo(k, 2)
        self.assertLess(np.linalg.norm(ts.matrix(ts.jacobian_mpo(hm, km))), 2e-11)
        unitary = expm(.2*omega)
        np.testing.assert_allclose(density(h@unitary), density(h), atol=2e-12)

    def test_one_site_tangent_orthonormality_and_existing_projector(self):
        h = ts.matrix_to_mpo(self.random_matrix(8), 3, chi=2)
        force = ts.matrix_to_mpo(self.random_matrix(8), 3)
        basis = ts.tangent_basis(h)
        basis_matrix = np.column_stack([b.dense() for b in basis])
        np.testing.assert_allclose(basis_matrix.conj().T@basis_matrix, np.eye(len(basis)), atol=2e-14)
        projection = basis_matrix@(basis_matrix.conj().T@force.dense())
        existing = ts.tn.tangent_project(h, force).dense()
        np.testing.assert_allclose(projection, existing, atol=2e-13)
        self.assertLess(len(basis), 4**3)

    def test_full_tangent_horizontal_and_positive_gauge_oracles(self):
        h = np.diag([.9, .6]).astype(complex)
        h /= np.trace(density(h)).real**.25
        ham, jumps = .23*X+.17*Z, [.41*LOWER]
        velocity, info = ts.factor_velocity(ts.matrix_to_mpo(h, 1),
            ts.matrix_to_mpo(ham, 1), [ts.matrix_to_mpo(j, 1) for j in jumps])
        k = ts.matrix(velocity)
        g = rb.lindblad_rhs(density(h), ham, jumps)
        eigen = np.diag(h).real
        denominator = eigen[:, None]**2+eigen[None, :]**2
        positive_k = g/((eigen[:, None]+eigen[None, :])*denominator)
        horizontal_k = eigen[None, :]*g/(denominator**2)
        np.testing.assert_allclose(k, horizontal_k, atol=2e-13)
        np.testing.assert_allclose(ts.jacobian_dense(h, positive_k), g, atol=2e-13)
        np.testing.assert_allclose(k@h+h@k.conj().T,
                                   positive_k@h+h@positive_k.conj().T, atol=2e-13)
        self.assertLess(np.linalg.norm(h.conj().T@k-k.conj().T@h), 2e-13)
        self.assertGreater(np.linalg.norm(k-positive_k), 1e-3)
        self.assertEqual(info["ls_rank"], 4)
        self.assertEqual(info["ls_nullity"], 4)
        self.assertLess(info["force_residual"], 2e-13)

    def test_restricted_tangent_stationarity_does_not_hide_physical_residual(self):
        rho = np.kron((I+.4*X)/2, (I+.3*X)/2)
        h = ts.matrix_to_mpo(driver.positive_power(rho, .25), 2, chi=1)
        velocity, info = ts.factor_velocity(h, ts.matrix_to_mpo(np.kron(Z, Z), 2), [])
        self.assertTrue(info["least_squares_stationary"])
        self.assertLess(info["ls_normal_relative"], 1e-10)
        self.assertGreater(info["force_residual"], .1)
        self.assertFalse(info["physical_residual_small"])
        g = rb.lindblad_rhs(density(ts.matrix(h)), np.kron(Z, Z), [])
        true_residual = np.linalg.norm(ts.jacobian_dense(ts.matrix(h), ts.matrix(velocity))-g)
        self.assertAlmostEqual(true_residual, info["force_residual"], places=13)


class FactorRetractionTests(unittest.TestCase):
    def test_physical_normalization_is_not_frobenius_factor_normalization(self):
        state, info = ts.normalize_factor(ts.matrix_to_mpo(np.eye(4), 2))
        h = ts.matrix(state)
        self.assertAlmostEqual(np.trace(density(h)).real, 1., places=13)
        self.assertAlmostEqual(np.trace(h.conj().T@h).real, 2., places=13)
        self.assertAlmostEqual(info["trace_before"], 4., places=13)
        self.assertAlmostEqual(info["root_normalization_change"], 1., places=13)

    def test_nonhermitian_factor_positivity_and_reconstruction(self):
        h = np.array([[.8, .2j], [.1, .5]], complex)
        state, _ = ts.normalize_factor(ts.matrix_to_mpo(h, 1))
        h = ts.matrix(state)
        root = ts.matrix(ts.root_mpo(state))
        self.assertGreater(np.linalg.norm(h-h.conj().T), .1)
        np.testing.assert_allclose(root, h@h.conj().T, atol=2e-15)
        self.assertGreater(np.linalg.eigvalsh(root)[0], 0.)
        np.testing.assert_allclose(root, driver.positive_power(density(h), .5), atol=2e-14)

    def test_retraction_reports_root_error(self):
        rng = np.random.default_rng(17)
        h = rng.normal(size=(4, 4))+1j*rng.normal(size=(4, 4))
        state = ts.matrix_to_mpo(h, 2)
        result, info = ts.retract_factor(state, 1)
        old_root, new_root = ts.matrix(ts.root_mpo(state)), ts.matrix(ts.root_mpo(result))
        self.assertLessEqual(np.linalg.norm(old_root-new_root), info["root_retraction_error_bound"]+2e-12)
        self.assertGreater(info["factor_discarded_bound"], 0.)
        self.assertAlmostEqual(np.trace(new_root@new_root).real, 1., places=12)
        self.assertGreaterEqual(np.linalg.eigvalsh(new_root)[0], -2e-14)


class FactorEvolutionTests(unittest.TestCase):
    def test_full_rank_nonhermitian_gauge_refines(self):
        rho0 = (I+.37*X+.19*Z)/2
        ham, jumps = .43*X+.19*Z, [np.sqrt(.4)*LOWER]
        initial = driver.positive_power(rho0, .25)@expm(.21j*Y)
        target = driver.reference_density(rho0, ham, jumps, .2)
        errors = []
        for steps in (4, 8, 16):
            state = ts.matrix_to_mpo(initial, 1)
            result, records = driver.evolve(state, ham, jumps, np.linspace(0, .2, steps+1), 1)
            errors.append(driver.score(result, target)["global_cp_root_error"])
            self.assertTrue(all(r["method"] == "factor_tangent_midpoint" for r in records))
            self.assertLess(abs(np.trace(density(ts.matrix(result))).real-1), 2e-13)
        self.assertLess(errors[1], .3*errors[0])
        self.assertLess(errors[2], .3*errors[1])
        self.assertLess(errors[-1], 1e-6)

    def test_first_opening_bridge_then_factor_tangent(self):
        pure = (I+X)/2
        state = ts.matrix_to_mpo(pure, 1)
        times = np.r_[0., np.linspace(.02, .08, 9)]
        result, records = driver.evolve(state, np.zeros((2, 2)), [Z/np.sqrt(2)], times, 1,
                                        clock="quartic")
        self.assertEqual(records[0]["method"], "channel_bridge")
        self.assertEqual(records[0]["event"], "first_order_opening")
        self.assertTrue(all(r["method"] == "factor_tangent_midpoint" for r in records[1:]))
        target = (I+np.exp(-.08)*X)/2
        score = driver.score(result, target)
        self.assertLess(score["global_cp_root_error"], 2e-5)
        self.assertGreater(score["minimum_root_eigenvalue"], 0.)
        self.assertLess(score["normalization_error"], 2e-13)
        self.assertLessEqual(score["global_cp_root_error"], score["gap_free_root_bound"]+2e-13)
        self.assertLessEqual(score["global_cp_root_error"], score["inverse_weighted_root_bound"]+2e-13)

    def test_spurious_truncation_rank_loss_is_guarded(self):
        # Rank-two H needs spatial bond two. Bond-one retraction discards the
        # second physical support sector even though the generator is zero.
        h = np.diag([1., 0., 0., .3]).astype(complex)
        state, _ = ts.normalize_factor(ts.matrix_to_mpo(h, 2, chi=2))
        with self.assertRaises(rb.RankOpeningError) as caught:
            driver.evolve(state, np.zeros((4, 4)), [], [0., .01], 1)
        self.assertIn("rank loss", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
