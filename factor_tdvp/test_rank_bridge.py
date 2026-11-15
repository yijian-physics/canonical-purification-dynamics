"""Focused dense channel/support tests; run from the repository root."""
import unittest

import numpy as np
from scipy.linalg import expm

if __package__:
    from .rank_bridge import (RankOpeningError, audit_support, bridge_factor,
        bridge_from_kraus, exact_channel_kraus, lindblad_rhs, lindblad_superoperator)
    from . import factor_tdvp as driver
    from . import tangent_solver as ts
else:
    from rank_bridge import (RankOpeningError, audit_support, bridge_factor,
        bridge_from_kraus, exact_channel_kraus, lindblad_rhs, lindblad_superoperator)
    import factor_tdvp as driver
    import tangent_solver as ts


X = np.array([[0, 1], [1, 0]], complex)
Z = np.diag([1., -1.]).astype(complex)
I = np.eye(2, dtype=complex)
LOWER = np.array([[0, 1], [0, 0]], complex)
GROUND = np.diag([1., 0.]).astype(complex)
PLUS = np.ones((2, 2), complex)/2


def density(factor):
    root = factor@factor.conj().T
    return root@root


class RankBridgeTests(unittest.TestCase):
    def test_superoperator_convention(self):
        rho = np.array([[.7, .1+.03j], [.1-.03j, .3]])
        ham = .3*X+.2*Z
        jumps = [.4*LOWER]
        actual = (lindblad_superoperator(ham,jumps)@rho.reshape(-1,order="F")).reshape((2,2),order="F")
        np.testing.assert_allclose(actual,lindblad_rhs(rho,ham,jumps),atol=1e-15)

    def test_first_opening_fourth_root_scale(self):
        singular = []
        for step in (1e-4, 1e-6):
            seed, info = bridge_factor(PLUS,0*X,[Z],step)
            expected = (I+np.exp(-2*step)*X)/2
            np.testing.assert_allclose(density(seed),expected,atol=3e-14)
            value = np.linalg.svd(seed,compute_uv=False)[-1]
            singular.append(value)
            self.assertAlmostEqual(value,(-np.expm1(-2*step)/2)**.25,places=11)
            self.assertTrue(info["safe_handoff"])
        self.assertAlmostEqual(singular[0]/singular[1],100**.25,delta=1e-4)

    def test_delayed_and_unitary_classification(self):
        self.assertEqual(audit_support(GROUND,X,[LOWER],.02)["mode"],"delayed_opening")
        self.assertEqual(audit_support(GROUND,X,[],.02)["mode"],"smooth_rank_preserving")
        self.assertEqual(audit_support(GROUND,0*X,[LOWER],.02)["mode"],"smooth_rank_preserving")
        seed, info = bridge_factor(GROUND,X,[LOWER],.02)
        self.assertEqual(info["support_rank_after"],2)
        self.assertTrue(info["safe_handoff"])
        expected = (expm(.02*lindblad_superoperator(X,[LOWER]))@GROUND.reshape(-1,order="F")).reshape((2,2),order="F")
        np.testing.assert_allclose(density(seed),expected,atol=2e-13)

    def test_repeated_generator_change(self):
        initial = np.kron(GROUND,GROUND)
        zero = np.zeros((4,4),complex)
        first = [np.kron(X,I)]
        second = [np.kron(I,X)]
        seed1, info1 = bridge_factor(initial,zero,first,.01)
        self.assertEqual(info1["support_rank_after"],2)
        self.assertTrue(info1["safe_handoff"])
        self.assertTrue(audit_support(seed1,zero,second,.01)["first_order_opening"])
        seed2, info2 = bridge_factor(seed1,zero,second,.01)
        self.assertEqual(info2["support_rank_after"],4)
        p = -np.expm1(-.02)/2
        expected = np.kron(np.diag([1-p,p]),np.diag([1-p,p]))
        np.testing.assert_allclose(density(seed2),expected,atol=2e-14)

    def test_correlated_three_spin_dephasing(self):
        ket = np.ones(8,complex)/np.sqrt(8)
        # Linear cluster state: CZ_01 CZ_12 |+++>.
        for n in range(8):
            b = [(n>>k)&1 for k in (2,1,0)]
            ket[n] *= (-1)**(b[0]*b[1]+b[1]*b[2])
        initial = np.outer(ket,ket.conj())
        jumps = [np.kron(np.kron(Z if j==0 else I,Z if j==1 else I),Z if j==2 else I) for j in range(3)]
        seed, info = bridge_factor(initial,np.zeros((8,8)),jumps,.01)
        self.assertEqual(info["support_rank_after"],8)
        self.assertTrue(info["safe_handoff"])
        self.assertEqual(info["channel_backend"],"exact_commuting_involutions")
        expected = initial.copy()
        for jump in jumps:
            p = -np.expm1(-.02)/2
            expected = (1-p)*expected+p*jump@expected@jump
        np.testing.assert_allclose(density(seed),expected,atol=2e-14)

    def test_kraus_completeness_and_arbitrary_factor(self):
        kraus, info = exact_channel_kraus(.3*X+.1*Z,[.7*LOWER],.13)
        self.assertLess(info["channel_completeness_error"],2e-14)
        self.assertLess(info["channel_superoperator_error"],2e-13)
        factor = np.array([[1.,.2j],[.3,.4]],complex)
        factor /= np.trace(density(factor)).real**.25
        seed, reconstruction = bridge_from_kraus(factor,kraus)
        expected = sum(k@density(factor)@k.conj().T for k in kraus)
        np.testing.assert_allclose(density(seed),expected,atol=2e-14)
        self.assertGreaterEqual(np.linalg.eigvalsh(seed@seed.conj().T)[0],0.)
        self.assertLess(reconstruction["bridge_density_reconstruction_error"],2e-14)

    def test_unresolved_tiny_delayed_bridge_is_guarded(self):
        with self.assertRaises(RankOpeningError) as caught:
            bridge_factor(GROUND,X,[LOWER],1e-10)
        self.assertFalse(caught.exception.diagnostics["safe_handoff"])

    def test_full_rank_stiffness_is_not_opening(self):
        factor = np.diag(np.array([1-1e-12,1e-12])**.25).astype(complex)
        info = audit_support(factor,0*X,[X],.01)
        self.assertEqual(info["rank"],2)
        self.assertEqual(info["mode"],"full_rank_ill_conditioned")
        self.assertFalse(info["first_order_opening"])

    def test_smooth_mpo_truncation_cannot_silently_lose_support(self):
        # The smaller sector is invariant under this zero generator: a
        # post-compression leakage test alone would incorrectly allow it.
        factor = np.diag([1.,0.,0.,.1]).astype(complex)
        factor /= np.trace(density(factor)).real**.25
        state = ts.matrix_to_mpo(factor,2,chi=2)
        zero = np.zeros((4,4),complex)
        before = audit_support(ts.matrix(state),zero,[],.01)
        self.assertEqual(before["rank"],2)
        self.assertTrue(before["safe_handoff"])
        truncated, _ = ts.retract_factor(state,1)
        after = audit_support(ts.matrix(truncated),zero,[],.01)
        self.assertEqual(after["rank"],1)
        self.assertTrue(after["safe_handoff"])
        with self.assertRaisesRegex(RankOpeningError,"numerical rank loss"):
            driver.evolve(state,zero,[],[0.,.01],chi=1)


if __name__ == "__main__":
    unittest.main()
