"""Regression checks for references, metric definitions and benchmark fairness."""
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.linalg import expm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from factor_tdvp import tangent_solver as ts
from factor_tdvp.rank_bridge import lindblad_superoperator
if __package__:
    from . import comparison_metrics as cm
else:
    import comparison_metrics as cm


def embed(op, site, n):
    return cm.kron_all([op if j == site else cm.I for j in range(n)])


class ReferenceTests(unittest.TestCase):
    def test_cluster_analytic_root_against_independent_channel(self):
        for n in (2, 3, 4):
            rates = .7+.15*np.cos(np.arange(n))**2
            initial, _ = cm.controlled_phase_reference(n,np.pi,rates,0)
            rho, root = cm.controlled_phase_reference(n,np.pi,rates,.37)
            direct = cm.direct_dephasing_density(initial,rates,.37)
            np.testing.assert_allclose(rho,direct,atol=5e-16)
            np.testing.assert_allclose(root,cm.positive_root(direct),atol=2e-14)
            np.testing.assert_allclose(root@root,rho,atol=3e-15)
            self.assertAlmostEqual(np.linalg.norm(root),1.,places=13)
            self.assertGreater(np.linalg.eigvalsh(root).min(),0.)

    def test_nonclifford_inhomogeneous_angles(self):
        n=4
        angles=.27+.03*np.arange(n-1)
        rates=.7+.15*np.cos(np.arange(n))**2
        initial, _=cm.controlled_phase_reference(n,angles,rates,0)
        rho, root=cm.controlled_phase_reference(n,angles,rates,1.)
        direct=cm.direct_dephasing_density(initial,rates,1.)
        np.testing.assert_allclose(rho,direct,atol=4e-16)
        np.testing.assert_allclose(root,cm.positive_root(direct),atol=5e-15)
        self.assertGreater(np.linalg.norm(initial.imag),.01)

    def test_tilted_semigroup_endpoint_matches_finite_channel(self):
        n=3; site=1; probability=.1; total=1.
        rho0,_=cm.controlled_phase_reference(n,.37,np.ones(n),0)
        gamma=-np.log1p(-2*probability)/total
        for theta in (0.,.05,.1,.2):
            pauli=embed(np.cos(theta)*cm.Z+np.sin(theta)*cm.X,site,n)
            jump=np.sqrt(gamma/2)*pauli
            liouv=lindblad_superoperator(np.zeros_like(rho0),[jump])
            endpoint=(expm(total*liouv)@rho0.reshape(-1,order='F')).reshape(rho0.shape,order='F')
            channel=(1-probability)*rho0+probability*pauli@rho0@pauli
            np.testing.assert_allclose(endpoint,channel,atol=8e-16)

    def test_mixed_chain_preparation_matches_existing_witness(self):
        path=ROOT/'scalable_cp_witness/code/scalable_cp_witness.py'
        spec=importlib.util.spec_from_file_location('_comparison_witness',path)
        module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        for n in (2,3):
            expected,_=cm.correlated_mixed_reference(n,time=0)
            actual=ts.matrix(module.prepared_power(n,1.,.12,.35))
            np.testing.assert_allclose(actual,expected,atol=2e-15)
            expected_t,_=cm.correlated_mixed_reference(n,time=.1)
            actual_t=ts.matrix(module.channel(module.prepared_power(n,1.,.12,.35),.1))
            np.testing.assert_allclose(actual_t,expected_t,atol=2e-15)


class MetricTests(unittest.TestCase):
    def test_right_unitary_gauge_is_not_removed_from_cp_error(self):
        rho=np.diag([.7,.3]).astype(complex)
        root=np.diag(np.sqrt([.7,.3])).astype(complex)
        unitary=expm(.4j*cm.X)
        a=root@unitary
        metrics=cm.amplitude_metrics(a,rho,root)
        self.assertLess(metrics['density_error'],1e-15)
        self.assertGreater(metrics['global_cp_error'],.1)
        self.assertGreater(metrics['canonical_branch_defect'],.1)
        self.assertLess(metrics['normalization_defect'],1e-15)

    def test_factor_metrics_use_squared_positive_root(self):
        rho=np.array([[.7,.1j],[-.1j,.3]])
        vals,vecs=np.linalg.eigh(rho)
        h=(vecs*vals**.25)@vecs.conj().T@expm(.3j*cm.X)
        root=h@h.conj().T
        metrics=cm.amplitude_metrics(root,rho)
        self.assertLess(metrics['global_cp_error'],2e-15)
        self.assertLess(metrics['density_error'],2e-15)
        self.assertGreater(abs(np.trace(h.conj().T@h)-1),.1)

    def test_scoring_never_normalizes_a_bad_output(self):
        rho=cm.I/2
        metrics=cm.amplitude_metrics(.9*cm.I/np.sqrt(2),rho)
        self.assertAlmostEqual(metrics['global_cp_error'],.1,places=13)
        self.assertAlmostEqual(metrics['normalization_defect'],.19,places=13)

    def test_reported_error_categories_stay_distinct(self):
        self.assertEqual(cm.classify_error({'status':'ok','global_cp_error':1e-5}),'measured_global')
        self.assertEqual(cm.classify_error({'status':'ok','global_cp_bound':1e-5}),'global_bound')
        self.assertEqual(cm.classify_error({'status':'ok','error_kind':'local_diagnostic','local_cp_error':1e-5}),'local_diagnostic')
        self.assertEqual(cm.classify_error({'status':'timeout','global_cp_error':1e-5}),'unavailable')
        self.assertEqual(cm.classify_error({'status':'unsupported','global_cp_bound':1e-5}),'unavailable')
        self.assertEqual(cm.classify_error({'status':'ok'}),'unavailable')
        self.assertEqual(cm.classify_error({'status':'ok','error_kind':'oracle_lower_bound','global_cp_error':1e-6}),'oracle_optimum')

    def test_local_root_fit_does_not_determine_global_extension(self):
        if __package__:
            from .audit_local_unitary import run_audit
        else:
            from audit_local_unitary import run_audit
        rows = run_audit()
        partial, complete = rows[1], rows[2]
        self.assertLess(partial['physical_cache_error'],1e-13)
        self.assertLess(partial['dense_application_error'],1e-13)
        self.assertLess(partial['local_fit_residual'],1e-10)
        self.assertGreater(partial['global_cp_error'],.09)
        self.assertAlmostEqual(partial['global_source_outside_local_span'],1/np.sqrt(2),places=12)
        self.assertLess(partial['exact_target_same_support_error'],1e-12)
        self.assertLess(complete['global_cp_error'],1e-10)
        self.assertLess(complete['global_source_outside_local_span'],1e-13)

    def test_nonpositive_reference_is_rejected(self):
        with self.assertRaises(ValueError):
            cm.positive_root(np.diag([1.1,-.1]))


if __name__=='__main__':
    unittest.main()
