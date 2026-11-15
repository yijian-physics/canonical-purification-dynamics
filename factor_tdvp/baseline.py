"""Thin full-rank control using the repository's existing root-TDVP engine.

This baseline intentionally retains the old dense Sylvester-force backend.  It
is a small-system comparison, not a component of the new factor solver.  The
time grid and one-site tangent choice match the factor benchmark; a canonical
root and its factor generally need different MPO bonds.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from time import perf_counter

import numpy as np


def _legacy_module():
    """Load the existing implementation without copying its algorithms.

    The legacy scripts use sibling imports rather than a package.  Prepend
    their own directory only while importing.  Their tdvp_core.py is identical
    to scalable_cp_witness/code/tdvp_core.py at the base commit.
    """
    key = "_factor_tdvp_legacy_root_evolution"
    if key in sys.modules:
        return sys.modules[key]
    directory = Path(__file__).resolve().parents[1] / "cp_tdvp"
    old_path = sys.path[:]
    try:
        sys.path.insert(0, str(directory))
        spec = importlib.util.spec_from_file_location(
            key, directory / "cp_tdvp_evolution.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = old_path
    return module


def _liouvillian(hamiltonian, jumps):
    """Dense column-vectorization generator, exclusively for this baseline."""
    d = len(hamiltonian)
    identity = np.eye(d, dtype=complex)
    out = -1j * (np.kron(identity, hamiltonian)
                 - np.kron(hamiltonian.T, identity))
    for jump in jumps:
        number = jump.conj().T @ jump
        out += (np.kron(jump.conj(), jump)
                - .5 * np.kron(identity, number)
                - .5 * np.kron(number.T, identity))
    return out


def evolve_root(rho0, hamiltonian, jumps, times, chi):
    """Return ``(root_mpo, rows)`` for legacy one-site root-TDVP.

    ``times`` includes zero.  Only full-rank trajectories above the legacy
    force's cutoff are supported; a rank boundary raises explicitly instead
    of using its zero-denominator fallback.  Exact future states are not used.
    Rows report stage null forcing, branch correction, MPO rounding, and time.
    The returned root is the legacy stored amplitude, so independent scoring
    should use its actual matrix rather than silently canonicalizing it.
    """
    rho0 = np.asarray(rho0, dtype=complex)
    hamiltonian = np.asarray(hamiltonian, dtype=complex)
    jumps = [np.asarray(j, dtype=complex) for j in jumps]
    times = np.asarray(times, dtype=float)
    d = len(rho0)
    n = int(round(np.log2(d)))
    if rho0.shape != (d, d) or d != 2**n or n < 1:
        raise ValueError("rho0 must act on one or more qubits")
    if times.ndim != 1 or len(times) < 1 or times[0] != 0:
        raise ValueError("times must be a one-dimensional grid starting at zero")
    if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
        raise ValueError("times must be finite and strictly increasing")
    if chi < 1:
        raise ValueError("chi must be positive")
    if np.linalg.norm(rho0-rho0.conj().T) > 1e-12:
        raise ValueError("rho0 must be Hermitian")
    if abs(np.trace(rho0)-1) > 1e-12:
        raise ValueError("rho0 must have trace one")
    if np.min(np.linalg.eigvalsh(rho0)) <= 1e-12:
        raise ValueError("the legacy baseline is restricted to full-rank input")
    legacy = _legacy_module()
    tn = legacy.tn
    root = legacy.positive_root(rho0)
    state = tn.from_dense(tn.to_interleaved(root, n), n, chi)
    dense_force = legacy.DenseForce(n, _liouvillian(hamiltonian, jumps),
                                    clock="physical")

    def guarded_force(t, current):
        a = tn.to_matrix(current.dense(), n)
        singular = np.linalg.svd(a, compute_uv=False)
        if singular[-1] <= 1e-12*singular[0]:
            raise RuntimeError("legacy root baseline encountered a rank boundary")
        return dense_force(t, current)

    retraction = legacy.polar_retractor(n, chi)
    rows = []
    started = perf_counter()
    for left, right in zip(times[:-1], times[1:]):
        state, info = tn.midpoint_step(
            state, float(left), float(right-left), guarded_force, chi,
            retract=retraction, mode="one")
        rows.append({
            "time": float(right),
            "dt": float(right-left),
            "bond": state.bond_dimension(),
            "max_null_forcing": max(info["force1"]["null_forcing"],
                                    info["force2"]["null_forcing"]),
            "min_stage_root_singular": min(
                info["force1"]["smallest_root_eigenvalue"],
                info["force2"]["smallest_root_eigenvalue"]),
            "retraction_discarded_bound": (
                info["mid_retraction"]["discarded_bound"]
                + info["end_retraction"]["discarded_bound"]),
            "polar_correction": (info["mid_retraction"]["polar_correction"]
                                 + info["end_retraction"]["polar_correction"]),
            "seconds": perf_counter()-started,
        })
    return state, rows
