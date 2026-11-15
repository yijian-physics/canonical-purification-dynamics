"""Finite-size factor-MPO McLachlan solver.

The state is an MPO H, never a separately evolved root.  The complex one-site
MPS tangent basis is represented by MPOs.  J_H(B) and J_H(i B) are constructed
with MPO products, then exported as small dense columns for real SVD least
squares.  This deliberately explicit backend is not a large-N solver.  It
never solves a dense Sylvester equation to obtain a root force.

Physical/reference indices are paired as 00, 01, 10, 11, exactly as in the
existing scalable_cp_witness tensor core.  Intermediate rounding is reported
separately; the stored physical root is the uncompressed product H H^dagger.
"""
from __future__ import annotations

import copy
from pathlib import Path
import sys
from time import perf_counter

_CORE = Path(__file__).resolve().parents[1] / "scalable_cp_witness" / "code"
if str(_CORE) not in sys.path:
    sys.path.insert(0, str(_CORE))
import tdvp_core as tn
import mpo_cp_force as mpo
import numpy as np


def matrix_to_mpo(a, n, chi=None):
    """Small-system input conversion, using the existing TT-SVD routine."""
    a = np.asarray(a, dtype=complex)
    if n < 1 or a.shape != (2**n, 2**n):
        raise ValueError("Expected a square 2**n matrix with n >= 1")
    return tn.from_dense(tn.to_interleaved(a, n), n, chi)


def matrix(state):
    """Dense export for this finite-size solver and independent diagnostics."""
    return tn.to_matrix(state.dense(), state.n)


def root_mpo(factor):
    """Exact represented root; no independent compression or branch choice."""
    return mpo.multiply(factor, mpo.adjoint(factor))


def jacobian_dense(h, k):
    """Independent dense arithmetic identity, not a force inversion."""
    x = h @ h.conj().T
    dx = k @ h.conj().T + h @ k.conj().T
    return x @ dx + dx @ x


class _Algebra:
    """Round intermediates, collecting LOCAL discard allowances only.

    These are not propagated global error bounds.  Default exact finite-chain
    capacity prevents bond-cap truncation but ordinary floating point remains.
    """
    def __init__(self, n, work_bond=None):
        self.bond = 4**(n // 2) if work_bond is None else int(work_bond)
        if self.bond < 1:
            raise ValueError("work_bond must be positive")
        self.discards = []

    def round(self, a):
        out, info = tn.compress(a, self.bond, tolerance=0., normalize=False)
        self.discards.append(info["discarded_bound"])
        return out

    def product(self, a, b):
        return self.round(mpo.multiply(a, b))

    def add(self, a, b, alpha=1., beta=1.):
        return self.round(tn.add(a, b, alpha=alpha, beta=beta))


class _Jacobian:
    def __init__(self, h, algebra):
        self.h, self.algebra = h, algebra
        self.hdag = mpo.adjoint(h)
        self.x = algebra.product(h, self.hdag)

    def apply(self, k):
        a = self.algebra
        dx = a.add(a.product(k, self.hdag),
                   a.product(self.h, mpo.adjoint(k)))
        return a.add(a.product(self.x, dx), a.product(dx, self.x))


def jacobian_mpo(h, k, work_bond=None, return_info=False):
    """Apply the REAL-linear Jacobian with MPO products and measured rounding."""
    if h.n != k.n:
        raise ValueError("Mismatched chain sizes")
    algebra = _Algebra(h.n, work_bond)
    result = _Jacobian(h, algebra).apply(k)
    info = {"local_discard_sum": float(sum(algebra.discards)),
            "local_discard_max": float(max(algebra.discards, default=0.))}
    return (result, info) if return_info else result


def generator_mpo(rho, hamiltonian, jumps, work_bond=None, return_info=False):
    """Apply a general finite-system Lindblad generator using MPO products."""
    algebra = _Algebra(rho.n, work_bond)
    result = algebra.add(algebra.product(hamiltonian, rho),
                         algebra.product(rho, hamiltonian), -1j, 1j)
    for jump in jumps:
        adj = mpo.adjoint(jump)
        number = algebra.product(adj, jump)
        gain = algebra.product(algebra.product(jump, rho), adj)
        loss = algebra.add(algebra.product(number, rho),
                           algebra.product(rho, number), -.5, -.5)
        result = algebra.add(result, algebra.add(gain, loss))
    info = {"local_discard_sum": float(sum(algebra.discards)),
            "local_discard_max": float(max(algebra.discards, default=0.))}
    return (result, info) if return_info else result


def tangent_basis(state):
    """Orthonormal COMPLEX basis for the full one-site fixed-bond tangent.

    At every nonfinal site use the orthogonal complement of its left-isometry
    columns, and the right-isometric suffix.  The last site is unrestricted.
    This removes MPS coordinate gauge, not necessarily all physical right-unitary
    gauge.  Those additional real null directions are handled by SVD least
    squares.  Bases B and iB together span the real tangent used by the solver.
    """
    right = copy.deepcopy(state)
    right.center = state.n - 1
    right.move_center(0)
    left = copy.deepcopy(right)
    left.move_center(state.n - 1)
    basis = []
    for j in range(state.n):
        dl, _, dr = left.tensors[j].shape
        if j < state.n - 1:
            flat = left.tensors[j].reshape(dl * 4, dr)
            q, _ = np.linalg.qr(flat, mode="complete")
            complement = q[:, dr:]
        else:
            complement = np.eye(dl * 4, dtype=complex)
        for column in range(complement.shape[1]):
            for r in range(dr):
                b = np.zeros((dl * 4, dr), dtype=complex)
                b[:, r] = complement[:, column]
                vector = tn.MPS(state.n, max_bond=state.max_bond)
                vector.tensors = ([a.copy() for a in left.tensors[:j]]
                                  + [b.reshape(dl, 4, dr)]
                                  + [a.copy() for a in right.tensors[j+1:]])
                vector.center = j
                basis.append(vector)
    return basis


def _real_vector(vector):
    return np.concatenate((vector.real, vector.imag))


def factor_velocity(h, hamiltonian_mpo, jumps_mpo, rcond=1e-11,
                    work_bond=None, max_real_columns=4096):
    """Solve min_{K in T_H} ||J_H(K)-L((HH*)**2)||_F by real SVD.

    No positivity or Hermiticity constraint is imposed on H or K.  The default
    rank-revealing cutoff is explicit in the diagnostics.  A small normal
    residual means LS stationarity only; it does NOT mean the physical residual
    is small.  Dense reference arithmetic checks the final physical residual
    independently of intermediate MPO rounding in this small-system backend.
    """
    started = perf_counter()
    if not 0 < rcond < 1:
        raise ValueError("rcond must lie strictly between zero and one")
    if h.n != hamiltonian_mpo.n or any(j.n != h.n for j in jumps_mpo):
        raise ValueError("Mismatched chain sizes")
    algebra = _Algebra(h.n, work_bond)
    jac = _Jacobian(h, algebra)
    rho = algebra.product(jac.x, jac.x)
    g, generator_info = generator_mpo(rho, hamiltonian_mpo, jumps_mpo,
                                      work_bond, return_info=True)
    basis = tangent_basis(h)
    if 2 * len(basis) > max_real_columns:
        raise RuntimeError("Explicit finite-size Jacobian exceeds max_real_columns; "
                           "a scalable iterative tangent solver is not implemented")
    columns = []
    for b in basis:
        columns.append(_real_vector(jac.apply(b).dense()))
        # J is real-linear: J(iB) cannot be obtained by multiplying J(B) by i.
        columns.append(_real_vector(jac.apply(tn.scale(b, 1j)).dense()))
    design = np.column_stack(columns)
    target = _real_vector(g.dense())
    if not (np.isfinite(design).all() and np.isfinite(target).all()):
        raise FloatingPointError("Nonfinite factor Jacobian or generator")
    coefficients, _, rank, singular = np.linalg.lstsq(design, target, rcond=rcond)
    velocity = tn.scale(h, 0.)
    velocity_algebra = _Algebra(h.n)  # preserve the entire tangent result
    dense_expected = np.zeros(4**h.n, dtype=complex)
    for j, b in enumerate(basis):
        c = coefficients[2*j] + 1j * coefficients[2*j+1]
        if c != 0:
            velocity = velocity_algebra.add(velocity, b, beta=c)
            dense_expected += c * b.dense()
    ls_residual = design @ coefficients - target
    normal = design.T @ ls_residual
    gnorm = float(np.linalg.norm(target))
    normal_scale = max(float(singular[0]) * gnorm, 1e-300)
    hm, km = matrix(h), matrix(velocity)
    xm = hm @ hm.conj().T
    rhom = xm @ xm
    ham = matrix(hamiltonian_mpo)
    gdense = -1j * (ham @ rhom - rhom @ ham)
    for jump in jumps_mpo:
        jm = matrix(jump)
        number = jm.conj().T @ jm
        gdense += jm @ rhom @ jm.conj().T - .5 * (number @ rhom + rhom @ number)
    actual_applied = jacobian_dense(hm, km)
    actual_residual = actual_applied - gdense
    density_trace = float(np.trace(rhom).real)
    trace_speed = float(np.trace(actual_applied).real)
    normalization_speed = trace_speed / (4. * density_trace)
    normalized_residual = actual_residual - 4. * normalization_speed * rhom
    actual_norm = float(np.linalg.norm(actual_residual))
    actual_gnorm = float(np.linalg.norm(gdense))
    retained_min = float(singular[rank-1]) if rank else 0.
    info = {
        "backend": "explicit_real_jacobian_mpo_tangent",
        "tangent": "one_site_fixed_bond",
        "real_tangent_dimension": int(design.shape[1]),
        "ls_rank": int(rank),
        "ls_nullity": int(design.shape[1] - rank),
        "ls_rcond": float(rcond),
        "ls_singular_max": float(singular[0]) if len(singular) else 0.,
        "ls_retained_singular_min": retained_min,
        "ls_retained_condition": float(singular[0] / retained_min) if rank else None,
        "ls_residual": float(np.linalg.norm(ls_residual)),
        "ls_normal_residual": float(np.linalg.norm(normal)),
        "ls_normal_relative": float(np.linalg.norm(normal) / normal_scale),
        "least_squares_stationary": bool(np.linalg.norm(normal) / normal_scale < 1e-8
                                          or np.linalg.norm(normal) < 1e-14),
        "force_residual": actual_norm,
        "relative_force_residual": actual_norm / max(actual_gnorm, 1e-300),
        "physical_residual_small": bool(actual_norm <= 1e-9 * max(1., actual_gnorm)),
        "generator_norm": actual_gnorm,
        "velocity_norm": float(np.linalg.norm(km)),
        "trace_velocity": trace_speed,
        "normalization_tangent_correction_norm": float(abs(normalization_speed) * np.linalg.norm(hm)),
        "normalized_force_residual": float(np.linalg.norm(normalized_residual)),
        "jacobian_column_error": float(np.linalg.norm(design @ coefficients - _real_vector(tn.to_interleaved(actual_applied, h.n)))),
        "horizontal_defect": float(np.linalg.norm(hm.conj().T @ km - km.conj().T @ hm)),
        "generator_mpo_error": float(np.linalg.norm(matrix(g) - gdense)),
        "velocity_reconstruction_error": float(np.linalg.norm(velocity.dense() - dense_expected)),
        "product_local_discard_sum": float(sum(algebra.discards)
                                            + generator_info["local_discard_sum"]),
        "product_local_discard_max": float(max(algebra.discards
                                              + [generator_info["local_discard_max"]], default=0.)),
        "velocity_local_discard_sum": float(sum(velocity_algebra.discards)),
        "factor_bond": h.bond_dimension(),
        "root_product_bond": root_mpo(h).bond_dimension(),
        "work_bond": algebra.bond,
        "seconds": perf_counter() - started,
    }
    if not info["least_squares_stationary"]:
        raise RuntimeError("Real tangent least squares failed its stationarity check")
    return velocity, info


def normalize_factor(h):
    """Normalize Tr[(HH*)**2]=1 by H -> H / sqrt(||HH*||_F)."""
    root_norm = tn.norm(root_mpo(h))
    if not np.isfinite(root_norm) or root_norm <= 0:
        raise FloatingPointError("Cannot normalize a zero or nonfinite factor")
    scaling = 1. / np.sqrt(root_norm)
    out = tn.scale(h, scaling)
    return out, {
        "trace_before": float(root_norm**2),
        "factor_scaling": float(scaling),
        "factor_normalization_change": float(abs(scaling - 1.) * tn.norm(h)),
        "root_normalization_change": float(abs(root_norm - 1.)),
    }


def retract_factor(raw, chi, tolerance=1e-13):
    """Compress H then normalize physically; HH* stays positive identically.

    Root truncation change is measured with MPO contractions.  It includes
    nonlinear propagation of the H truncation but excludes normalization,
    which is reported separately and can be added by the triangle inequality.
    """
    if chi < 1:
        raise ValueError("chi must be positive")
    compressed, compression = tn.compress(raw, chi, tolerance=tolerance, normalize=False)
    root_truncation = tn.distance(root_mpo(raw), root_mpo(compressed))
    result, normalization = normalize_factor(compressed)
    info = {
        "factor_discarded_bound": compression["discarded_bound"],
        "factor_truncation_error": tn.distance(raw, compressed),
        "root_truncation_error": root_truncation,
        "root_retraction_error_bound": root_truncation + normalization["root_normalization_change"],
        **normalization,
    }
    return result, info


def midpoint_step(h, hamiltonian_mpo, jumps_mpo, dt, chi, *, rcond=1e-11,
                  work_bond=None, stage_scales=(1., 1.), guard=None,
                  velocity_solver=None, solver_options=None):
    """Retracted explicit midpoint on the H-MPO manifold.

    For physical time use stage_scales=(1,1). For t=t_*+u**4, pass dt=du
    and (4*u**3, 4*(u+du/2)**3). A clock does not change the tangent space;
    the caller must bridge any rank event before entering this routine.
    Optional guard(H) runs before BOTH velocity solves and may raise.
    An optional velocity_solver uses the same (H, Hamiltonian, jumps, rcond,
    work_bond) interface. Rank guards and retractions apply to every backend.
    """
    if dt <= 0 or not np.isfinite(dt):
        raise ValueError("dt must be finite and positive")
    if len(stage_scales) != 2 or not np.isfinite(stage_scales).all():
        raise ValueError("Expected two finite clock speeds")
    if guard is not None:
        guard(h)
    solve = factor_velocity if velocity_solver is None else velocity_solver
    options = {} if solver_options is None else dict(solver_options)
    k1, i1 = solve(h, hamiltonian_mpo, jumps_mpo, rcond, work_bond, **options)
    midpoint, rm = retract_factor(tn.add(h, k1, beta=.5 * dt * stage_scales[0]), chi)
    if guard is not None:
        guard(midpoint)
    k2, i2 = solve(midpoint, hamiltonian_mpo, jumps_mpo, rcond, work_bond, **options)
    result, re = retract_factor(tn.add(h, k2, beta=dt * stage_scales[1]), chi)
    return result, {"force1": i1, "force2": i2,
                    "mid_retraction": rm, "end_retraction": re,
                    "stage_scales": [float(x) for x in stage_scales]}
