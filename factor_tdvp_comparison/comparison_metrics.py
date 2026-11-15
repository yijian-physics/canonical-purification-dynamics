"""Independent finite-size references and unambiguous comparison metrics.

These helpers score outputs; they never provide future roots to an integrator.
A general purification amplitude A represents physical density A A†. For the
factor method, supply A=H H†, which represents density (H H†)^2.
"""
from __future__ import annotations

import numpy as np

I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], complex)
Z = np.diag([1., -1.]).astype(complex)


def kron_all(terms):
    out = np.ones((1, 1), complex)
    for term in terms:
        out = np.kron(out, term)
    return out


def positive_root(rho):
    """Dense validation root; negative roundoff is clipped, positive modes kept."""
    rho = np.asarray(rho, complex)
    herm = (rho + rho.conj().T)/2
    values, vectors = np.linalg.eigh(herm)
    if values[0] < -1e-12 * max(1., np.linalg.norm(herm, 2)):
        raise ValueError("Reference density is not positive")
    return (vectors*np.sqrt(np.maximum(values, 0)))@vectors.conj().T


def controlled_phase_diagonal(n, angles):
    angles = np.broadcast_to(np.asarray(angles, float), (max(0, n-1),))
    bits = (np.arange(2**n)[:, None] >> np.arange(n-1, -1, -1)) & 1
    return np.exp(1j*np.sum(bits[:, :-1]*bits[:, 1:]*angles, axis=1))


def controlled_phase_reference(n, angles, rates, time):
    """Exact A/B density and canonical root by commuting circuit covariance."""
    if n < 1 or time < 0:
        raise ValueError("Expected n >= 1 and time >= 0")
    rates = np.broadcast_to(np.asarray(rates, float), (n,))
    if np.any(rates < 0):
        raise ValueError("Rates must be nonnegative")
    q = np.exp(-rates*time)
    locals_rho = [(I+value*X)/2 for value in q]
    # Closed formula retains exact zero at t=0, avoiding spurious sqrt(eps).
    locals_root = []
    for value in q:
        a, b = np.sqrt((1+value)/2), np.sqrt((1-value)/2)
        locals_root.append(((a+b)*I+(a-b)*X)/2)
    phase = controlled_phase_diagonal(n, angles)
    wrap = lambda a: phase[:, None]*a*phase.conj()[None, :]
    return wrap(kron_all(locals_rho)), wrap(kron_all(locals_root))


def direct_dephasing_density(rho0, rates, time):
    """Independent computational-basis channel, not a square-root formula."""
    n = int(round(np.log2(len(rho0))))
    if rho0.shape != (2**n, 2**n):
        raise ValueError("Density dimension must be a qubit power")
    rates = np.broadcast_to(np.asarray(rates, float), (n,))
    bits = (np.arange(2**n)[:, None] >> np.arange(n-1, -1, -1)) & 1
    decay = np.exp(-time*np.sum((bits[:, None, :] != bits[None, :, :])*rates, axis=2))
    return np.asarray(rho0)*decay


def correlated_mixed_reference(n, polarization=.12, angle=.35, time=0.):
    """Benchmark D: X-product -> controlled phase -> all-site Hadamards."""
    local = (I+polarization*X)/2
    raw = kron_all([local]*n)
    phase = controlled_phase_diagonal(n, angle)
    raw = phase[:, None]*raw*phase.conj()[None, :]
    hadamard = np.array([[1., 1.], [1., -1.]])/np.sqrt(2)
    unitary = kron_all([hadamard]*n)
    initial = unitary@raw@unitary.conj().T
    target = direct_dephasing_density(initial, np.ones(n), time)
    return target, positive_root(target)


def amplitude_metrics(amplitude, reference_density, reference_root=None):
    """Global Frobenius scores without purification-gauge optimization.

    Branch defect is ||A-sqrt(A A†)||F; it is measured independently from
    physical density error. No normalization is applied silently for scoring.
    """
    a = np.asarray(amplitude, complex)
    rho = a@a.conj().T
    target = positive_root(reference_density) if reference_root is None else reference_root
    # Polar root from A itself avoids squaring small singular values first.
    u, singular, _ = np.linalg.svd(a, full_matrices=False)
    own_root = (u*singular)@u.conj().T
    return {
        "global_cp_error": float(np.linalg.norm(a-target)),
        "density_error": float(np.linalg.norm(rho-reference_density)),
        "canonical_branch_defect": float(np.linalg.norm(a-own_root)),
        "amplitude_hermiticity_defect": float(np.linalg.norm(a-a.conj().T)),
        "normalization_defect": float(abs(np.trace(rho)-1)),
        "error_kind": "measured_global",
    }


def classify_error(row):
    """Never silently promote a local residual or bound to measured global error."""
    if row.get("status", "ok") not in ("ok", "success", "completed"):
        return "unavailable"
    kind = row.get("error_kind")
    if kind in ("oracle_lower_bound", "oracle_optimum"):
        return "oracle_optimum"
    if kind in ("local_diagnostic", "local_bound", "global_bound"):
        return kind
    if row.get("global_cp_error") is not None:
        return "measured_global"
    if row.get("global_cp_bound") is not None:
        return "global_bound"
    if row.get("local_cp_error") is not None:
        return "local_diagnostic"
    return "unavailable"
