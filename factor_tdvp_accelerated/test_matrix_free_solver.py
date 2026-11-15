"""Independent finite-system oracles for the MPO-output factor tangent solver.

Dense arrays appear only in these validation tests, never in the production
Jacobian/LSQR code. Run from repository root with python3 -m unittest
factor_tdvp_accelerated.test_matrix_free_solver -v.
"""
from __future__ import annotations

import unittest
from unittest.mock import patch
import numpy as np
from scipy.linalg import expm

from factor_tdvp import tangent_solver as ts
from factor_tdvp import rank_bridge as rb
from factor_tdvp import factor_tdvp as fd
from factor_tdvp_accelerated import matrix_free_solver as mf


I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], complex)
Y = np.array([[0, -1j], [1j, 0]], complex)
Z = np.diag([1., -1.]).astype(complex)
LOWER = np.array([[0, 1], [0, 0]], complex)
GROUND = np.diag([1., 0.]).astype(complex)


def density(h):
    root = h @ h.conj().T
    return root @ root


def random_matrix(rng, size):
    return rng.normal(size=(size, size)) + 1j*rng.normal(size=(size, size))


def smooth_problem(n=2, chi=None):
    rng = np.random.default_rng(710+n)
    h = np.eye(2**n) + .06*random_matrix(rng, 2**n)
    factor, _ = ts.normalize_factor(ts.matrix_to_mpo(h, n, chi))
    ham = .17*fd.embed(X, 0, n) + .23*fd.embed(Z, n-1, n)
    if n > 1:
        ham += .19*fd.embed(Z, 0, n) @ fd.embed(Z, 1, n)
    jumps = [.37*fd.embed(LOWER, j, n) for j in range(n)]
    return factor, ts.matrix_to_mpo(ham, n), [ts.matrix_to_mpo(j, n) for j in jumps]


def dense_coordinate_problem(factor, ham, jumps):
    """Independent explicit Jacobian assembled from the established basis."""
    h = ts.matrix(factor)
    basis = ts.tangent_basis(factor)
    columns = []
    for b in basis:
        for phase in (1., 1j):
            a = ts.jacobian_dense(h, phase*ts.matrix(b)).ravel()
            columns.append(np.r_[a.real, a.imag])
    g = rb.lindblad_rhs(density(h), ts.matrix(ham), [ts.matrix(j) for j in jumps]).ravel()
    return np.column_stack(columns), np.r_[g.real, g.imag]


class TangentCoordinateTests(unittest.TestCase):
    def test_decode_and_environment_projection_match_old_basis(self):
        rng = np.random.default_rng(12)
        for n, chi in ((1, 1), (2, 2), (3, 2)):
            with self.subTest(n=n):
                h = ts.matrix_to_mpo(random_matrix(rng, 2**n), n, chi)
                tangent = mf.TangentCoordinates(h)
                basis = np.column_stack([b.dense() for b in ts.tangent_basis(h)])
                coefficients = rng.normal(size=tangent.dimension)
                actual = tangent.decode(coefficients)
                z = coefficients[::2] + 1j*coefficients[1::2]
                np.testing.assert_allclose(actual.dense(), basis@z, atol=5e-13)
                np.testing.assert_allclose(tangent.encode(actual), coefficients, atol=5e-13)
                force = ts.matrix_to_mpo(random_matrix(rng, 2**n), n)
                projected = tangent.decode(tangent.encode(force)).dense()
                expected = basis @ (basis.conj().T @ force.dense())
                np.testing.assert_allclose(projected, expected, atol=5e-13)
                np.testing.assert_allclose(projected, ts.tn.tangent_project(h, force).dense(), atol=5e-13)
                self.assertLessEqual(actual.bond_dimension(), 2*h.bond_dimension())

    def test_real_coordinate_pairing(self):
        rng = np.random.default_rng(61)
        h = ts.matrix_to_mpo(random_matrix(rng, 8), 3, 2)
        tangent = mf.TangentCoordinates(h)
        c = rng.normal(size=tangent.dimension)
        force = ts.matrix_to_mpo(random_matrix(rng, 8), 3)
        lhs = np.vdot(tangent.decode(c).dense(), force.dense()).real
        rhs = c @ tangent.encode(force)
        self.assertAlmostEqual(lhs, rhs, places=11)


class AdjointTests(unittest.TestCase):
    def test_complex_nonhermitian_h_k_r_adjoint_identity(self):
        rng = np.random.default_rng(41)
        h, k, residual = (random_matrix(rng, 4) for _ in range(3))
        h /= np.linalg.norm(h)
        hm, km, rm = (ts.matrix_to_mpo(a, 2) for a in (h, k, residual))
        jac = mf.Jacobian(hm, 4)
        actual = ts.matrix(jac.adjoint(rm))
        hermitian_r = (residual+residual.conj().T)/2
        root = h @ h.conj().T
        expected = 2*(root@hermitian_r+hermitian_r@root)@h
        np.testing.assert_allclose(actual, expected, rtol=3e-12, atol=3e-13)
        np.testing.assert_allclose(ts.matrix(jac.apply(km)), ts.jacobian_dense(h, k), atol=5e-13)
        lhs = np.vdot(ts.jacobian_dense(h, k), residual).real
        rhs = np.vdot(k, actual).real
        self.assertAlmostEqual(lhs, rhs, places=11)
        antihermitian = ts.matrix_to_mpo((residual-residual.conj().T)/2, 2)
        self.assertLess(np.linalg.norm(ts.matrix(jac.adjoint(antihermitian))), 3e-12)

    def test_real_linearity_and_physical_gauge_null(self):
        rng = np.random.default_rng(9)
        h, k, a = (random_matrix(rng, 4) for _ in range(3))
        h /= np.linalg.norm(h)
        jac = mf.Jacobian(ts.matrix_to_mpo(h, 2), 4)
        omega = (a-a.conj().T)/2
        null = ts.matrix(jac.apply(ts.matrix_to_mpo(h@omega, 2)))
        self.assertLess(np.linalg.norm(null), 5e-12)
        jk = ts.matrix(jac.apply(ts.matrix_to_mpo(k, 2)))
        jik = ts.matrix(jac.apply(ts.matrix_to_mpo(1j*k, 2)))
        self.assertGreater(np.linalg.norm(jik-1j*jk), .1)
        np.testing.assert_allclose(density(h@expm(.13*omega)), density(h), atol=3e-13)


class SolverTests(unittest.TestCase):
    def test_full_and_restricted_minimum_norm_match_explicit(self):
        for chi in (4, 2):
            with self.subTest(chi=chi):
                factor, ham, jumps = smooth_problem(2, chi)
                explicit, old = ts.factor_velocity(factor, ham, jumps)
                result, info = mf.factor_velocity(factor, ham, jumps, tolerance=1e-10)
                self.assertTrue(info['converged'])
                np.testing.assert_allclose(ts.matrix(result), ts.matrix(explicit), rtol=3e-7, atol=3e-9)
                self.assertLess(abs(info['force_residual']-old['force_residual']), 3e-9)
                a, g = dense_coordinate_problem(factor, ham, jumps)
                c = np.asarray(info['coefficients'])
                relative = np.linalg.norm(a.T@(a@c-g))/np.linalg.norm(a.T@g)
                self.assertLess(relative, 3e-9)
                self.assertAlmostEqual(info['ls_normal_relative'], relative, delta=3e-10)

    def test_stationarity_does_not_claim_small_physical_residual(self):
        rho = np.kron((I+.4*X)/2, (I+.3*X)/2)
        factor = ts.matrix_to_mpo(fd.positive_power(rho, .25), 2, 1)
        ham = ts.matrix_to_mpo(np.kron(Z, Z)+.2*np.kron(Y, I), 2)
        velocity, info = mf.factor_velocity(factor, ham, [], tolerance=1e-9)
        g = rb.lindblad_rhs(density(ts.matrix(factor)), ts.matrix(ham), [])
        residual = ts.jacobian_dense(ts.matrix(factor), ts.matrix(velocity))-g
        self.assertTrue(info['least_squares_stationary'])
        self.assertFalse(info['physical_residual_small'])
        self.assertGreater(info['force_residual'], .1)
        self.assertAlmostEqual(np.linalg.norm(residual), info['force_residual'], places=10)

    def test_production_force_forbids_dense_export_and_explicit_basis(self):
        factor, ham, jumps = smooth_problem(2, 2)
        with patch.object(ts.tn.MPS, 'dense', side_effect=AssertionError('Dense export forbidden')), \
             patch.object(ts, 'tangent_basis', side_effect=AssertionError('Explicit basis forbidden')):
            velocity, info = mf.factor_velocity(factor, ham, jumps, tolerance=1e-9)
        self.assertTrue(info['converged'])
        self.assertEqual(velocity.n, 2)

    def test_iteration_limit_is_guarded_and_diagnostics_retained(self):
        factor, ham, jumps = smooth_problem()
        with self.assertRaises(mf.SolverError) as caught:
            mf.factor_velocity(factor, ham, jumps, maxiter=1, tolerance=1e-12)
        self.assertFalse(caught.exception.diagnostics['converged'])
        self.assertEqual(caught.exception.diagnostics['iterations'], 1)
        _, info = mf.factor_velocity(factor, ham, jumps, maxiter=1, tolerance=1e-12, strict=False)
        self.assertFalse(info['least_squares_stationary'])
        self.assertGreater(info['ls_normal_relative'], 1e-12)

    def test_inadequate_work_bond_is_guarded(self):
        factor, ham, jumps = smooth_problem()
        with self.assertRaises(mf.CompressionError) as caught:
            mf.factor_velocity(factor, ham, jumps, work_bond=1)
        info = caught.exception.diagnostics
        self.assertGreater(info['relative_discard'], info['compression_tolerance'])

    def test_damping_preconditioning_matches_augmented_dense_oracle(self):
        factor, ham, jumps = smooth_problem(1)
        a, g = dense_coordinate_problem(factor, ham, jumps)
        damping = .1
        expected = np.linalg.lstsq(np.vstack([a, damping*np.eye(a.shape[1])]),
                                 np.r_[g, np.zeros(a.shape[1])], rcond=None)[0]
        p = np.linspace(.7, 1.3, a.shape[1])
        _, info = mf.factor_velocity(factor, ham, jumps, damping=damping,
                                     preconditioner=p, tolerance=1e-11)
        np.testing.assert_allclose(info['coefficients'], expected, atol=3e-10)
        self.assertTrue(info['preconditioned'])
        self.assertEqual(info['damping'], damping)

    def test_zero_generator_returns_zero_velocity(self):
        factor, _, _ = smooth_problem(1)
        velocity, info = mf.factor_velocity(factor, ts.matrix_to_mpo(0*X, 1), [])
        self.assertTrue(info['converged'])
        self.assertLess(np.linalg.norm(ts.matrix(velocity)), 1e-14)
        self.assertEqual(info['iterations'], 0)


class IntegratedEvolutionTests(unittest.TestCase):
    options = {'tolerance': 1e-9}

    def evolve(self, *args, **kwargs):
        return fd.evolve(*args, velocity_solver=mf.factor_velocity,
                         solver_options=self.options, **kwargs)

    def test_smooth_nonhermitian_factor_second_order_and_positivity(self):
        rho = (I+.37*X+.19*Z)/2
        initial = fd.positive_power(rho, .25) @ expm(.21j*Y)
        ham, jumps = .43*X+.19*Z, [np.sqrt(.4)*LOWER]
        target = fd.reference_density(rho, ham, jumps, .2)
        errors = []
        for steps in (4, 8, 16):
            result, records = self.evolve(ts.matrix_to_mpo(initial, 1), ham, jumps,
                                          np.linspace(0, .2, steps+1), 1)
            score = fd.score(result, target)
            errors.append(score['global_cp_root_error'])
            self.assertLess(score['normalization_error'], 5e-13)
            self.assertGreater(score['minimum_root_eigenvalue'], 0.)
            self.assertTrue(all(r['solver']['force1']['backend']=='mpo_output_real_lsqr' for r in records))
        self.assertLess(errors[1], .3*errors[0])
        self.assertLess(errors[2], .3*errors[1])
        self.assertLess(errors[-1], 1e-6)

    def test_first_order_opening_bridge_then_matrix_free_tangent(self):
        pure = (I+X)/2
        result, records = self.evolve(ts.matrix_to_mpo(pure, 1), 0*X,
            [Z/np.sqrt(2)], np.r_[0., np.linspace(.02, .08, 9)], 1, clock='quartic')
        self.assertEqual(records[0]['method'], 'channel_bridge')
        self.assertEqual(records[0]['event'], 'first_order_opening')
        self.assertTrue(all(r['method']=='factor_tangent_midpoint' for r in records[1:]))
        target = (I+np.exp(-.08)*X)/2
        score = fd.score(result, target)
        self.assertLess(score['global_cp_root_error'], 2e-5)
        self.assertGreater(score['minimum_root_eigenvalue'], 0.)

    def test_repeated_opening_after_generator_change(self):
        initial = np.kron(GROUND, GROUND)
        zero = np.zeros((4, 4), complex)
        first = [np.kron(X, I)]
        second = [np.kron(I, X)]
        # Each segment bridges a fresh support sector, then invokes LSQR.
        times = np.array([0., .02, .021])
        state, history1 = self.evolve(ts.matrix_to_mpo(initial, 2), zero, first, times, 4)
        state, history2 = self.evolve(state, zero, second, times+.021, 4)
        self.assertEqual(history1[0]['rank_before'], 1)
        self.assertEqual(history1[0]['rank_after'], 2)
        self.assertEqual(history2[0]['rank_before'], 2)
        self.assertEqual(history2[0]['rank_after'], 4)
        self.assertEqual(history1[-1]['solver']['force1']['backend'], 'mpo_output_real_lsqr')
        self.assertEqual(history2[-1]['solver']['force1']['backend'], 'mpo_output_real_lsqr')
        target = fd.reference_density(fd.reference_density(initial, zero, first, .021), zero, second, .021)
        self.assertLess(fd.score(state, target)['global_cp_root_error'], 1e-5)

    def test_delayed_support_opening_is_bridged(self):
        state, records = self.evolve(ts.matrix_to_mpo(GROUND, 1), X, [LOWER], [0., .02], 1)
        self.assertEqual(records[0]['event'], 'delayed_opening')
        self.assertEqual(records[0]['method'], 'channel_bridge')
        self.assertEqual(records[0]['rank_after'], 2)
        target = fd.reference_density(GROUND, X, [LOWER], .02)
        self.assertLess(fd.score(state, target)['density_frobenius_error'], 3e-13)

    def test_truncation_rank_loss_remains_guarded(self):
        h = np.diag([1., 0., 0., .3]).astype(complex)
        state, _ = ts.normalize_factor(ts.matrix_to_mpo(h, 2, 2))
        with self.assertRaisesRegex(rb.RankOpeningError, 'rank loss'):
            self.evolve(state, np.zeros((4, 4)), [], [0., .01], 1)

    def test_retraction_keeps_positive_root_and_physical_normalization(self):
        rng = np.random.default_rng(19)
        raw = ts.matrix_to_mpo(random_matrix(rng, 4), 2)
        state, info = ts.retract_factor(raw, 1)
        root = ts.matrix(ts.root_mpo(state))
        self.assertGreaterEqual(np.linalg.eigvalsh(root)[0], -1e-13)
        self.assertAlmostEqual(np.trace(root@root).real, 1., places=12)
        true_change = np.linalg.norm(ts.matrix(ts.root_mpo(raw))-root)
        self.assertLessEqual(true_change, info['root_retraction_error_bound']+1e-12)


if __name__ == '__main__':
    unittest.main()
