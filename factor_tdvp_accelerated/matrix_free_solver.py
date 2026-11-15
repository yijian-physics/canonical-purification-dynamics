"""MPO-output LSQR for the original physical-density factor-TDVP objective.

The Krylov codomain consists of MPOs, not 4**N arrays. The domain is a real
vector of gauge-fixed one-site tangent coordinates. Fixed canonical frames
and left/right contraction environments replace full-chain basis columns.

Bond truncation makes arithmetic inexact and potentially nonlinear. By default
any single rounding discard exceeding ``compression_tolerance`` times the
operand norm raises CompressionError. Reported local allowances are NOT a
propagated global error certificate. Rank audits/bridges belong to the caller.
"""
from __future__ import annotations

import copy
from time import perf_counter
import numpy as np
from factor_tdvp import tangent_solver as ts

tn, mpo = ts.tn, ts.mpo


class SolverError(RuntimeError):
    """Guarded failure retaining measured diagnostics."""
    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


class CompressionError(SolverError):
    pass


class TangentCoordinates:
    """Orthonormal one-site tangent coordinates, with real/imaginary interleave.

    This removes MPS representation gauge, not all physical H -> H U gauge.
    The full-chain tangent vector is a single block MPO of bond at most 2 chi.
    """
    def __init__(self, h):
        self.n = h.n
        self.right = copy.deepcopy(h)
        self.right.center = h.n - 1
        self.right.move_center(0)
        self.left = copy.deepcopy(self.right)
        self.left.move_center(h.n - 1)
        self.complements, self.shapes, self.offsets = [], [], [0]
        for j, a in enumerate(self.left.tensors):
            dl, _, dr = a.shape
            if j < self.n - 1:
                q, _ = np.linalg.qr(a.reshape(dl * 4, dr), mode="complete")
                q = q[:, dr:]
            else:
                q = np.eye(dl * 4, dtype=complex)
            shape = (q.shape[1], dr)
            self.complements.append(q)
            self.shapes.append(shape)
            self.offsets.append(self.offsets[-1] + int(np.prod(shape)))
        self.complex_dimension = self.offsets[-1]
        self.dimension = 2 * self.complex_dimension

    def decode(self, coefficients):
        c = np.asarray(coefficients, dtype=float)
        if c.shape != (self.dimension,) or not np.isfinite(c).all():
            raise ValueError("Expected finite real tangent coordinates")
        z = c[::2] + 1j * c[1::2]
        variations = []
        for j, q in enumerate(self.complements):
            local = z[self.offsets[j]:self.offsets[j+1]].reshape(self.shapes[j])
            variations.append((q @ local).reshape(self.left.tensors[j].shape))
        out = tn.MPS(self.n, max_bond=2 * self.left.max_bond)
        if self.n == 1:
            out.tensors = variations
            out.center = 0
            return out
        out.tensors = [np.concatenate([self.left.tensors[0], variations[0]], axis=2)]
        for j in range(1, self.n-1):
            a, b, r = self.left.tensors[j], variations[j], self.right.tensors[j]
            block = np.zeros((a.shape[0]+r.shape[0], 4, a.shape[2]+r.shape[2]), complex)
            block[:a.shape[0], :, :a.shape[2]] = a
            block[:a.shape[0], :, a.shape[2]:] = b
            block[a.shape[0]:, :, a.shape[2]:] = r
            out.tensors.append(block)
        out.tensors.append(np.concatenate([variations[-1], self.right.tensors[-1]], axis=0))
        out.center = self.n-1
        return out

    def encode(self, force):
        """B* force by reusable frames and two environment sweeps; no columns."""
        if force.n != self.n:
            raise ValueError("Mismatched chain sizes")
        renv = [None] * (self.n + 1)
        renv[self.n] = np.ones((1, 1), complex)
        for j in range(self.n-1, 0, -1):
            renv[j] = np.einsum('ru,asr,bsu->ab', renv[j+1],
                               self.right.tensors[j].conj(), force.tensors[j], optimize=True)
        lenv = np.ones((1, 1), complex)
        blocks = []
        for j, q in enumerate(self.complements):
            local = np.einsum('al,lsr,br->asb', lenv, force.tensors[j], renv[j+1], optimize=True)
            block = q.conj().T @ local.reshape(q.shape[0], -1)
            blocks.append(block.ravel())
            if j < self.n-1:
                lenv = np.einsum('ab,asr,bsu->ru', lenv,
                                 self.left.tensors[j].conj(), force.tensors[j], optimize=True)
        z = np.concatenate(blocks)
        c = np.empty(self.dimension)
        c[::2], c[1::2] = z.real, z.imag
        return c


class Algebra:
    def __init__(self, work_bond, compression_tolerance=1e-10, strict=True):
        if int(work_bond) != work_bond or work_bond < 1:
            raise ValueError("work_bond must be a positive integer")
        if not np.isfinite(compression_tolerance) or compression_tolerance < 0:
            raise ValueError("compression_tolerance must be finite and nonnegative")
        self.bond = int(work_bond)
        self.compression_tolerance, self.strict = compression_tolerance, strict
        self.discards, self.relative_discards = [], []
        self.products = self.roundings = 0
        self.product_seconds = self.round_seconds = 0.

    def round(self, value, cancellation_scale=None):
        started = perf_counter()
        out, info = tn.compress(value, self.bond, tolerance=1e-14, normalize=False)
        discard = float(info['discarded_bound'])
        # Orthogonal truncation has norm(input) <= norm(output)+discard.
        scale = max(float(info['norm_before']) + discard, np.finfo(float).tiny)
        relative = discard / scale
        if relative > self.compression_tolerance and cancellation_scale is not None:
            # A near-zero subtraction can lose digits relative to its result
            # while staying backward accurate relative to its operands.
            scale = max(scale, float(cancellation_scale()))
            relative = discard / max(scale, np.finfo(float).tiny)
        self.discards.append(discard)
        self.relative_discards.append(relative)
        self.roundings += 1
        self.round_seconds += perf_counter()-started
        if self.strict and relative > self.compression_tolerance:
            raise CompressionError("MPO work bond discarded too much for a reliable Krylov operation", {
                'work_bond': self.bond, 'discarded_bound': discard,
                'relative_discard': relative, 'compression_tolerance': self.compression_tolerance})
        return out

    def product(self, a, b):
        started = perf_counter()
        raw = mpo.multiply(a, b)
        self.product_seconds += perf_counter()-started
        self.products += 1
        return self.round(raw)

    def add(self, a, b, alpha=1., beta=1.):
        return self.round(tn.add(a, b, alpha=alpha, beta=beta),
                          lambda: abs(alpha)*tn.norm(a)+abs(beta)*tn.norm(b))


class Jacobian:
    """Real Jacobian and adjoint, including Hermitian projection of residuals."""
    def __init__(self, h, work_bond, compression_tolerance=1e-10, strict=True, algebra=None):
        self.h = h
        self.algebra = algebra or Algebra(work_bond, compression_tolerance, strict)
        self.hdag = mpo.adjoint(h)
        self.x = self.algebra.product(h, self.hdag)
        self.applications = self.adjoint_applications = 0

    def apply(self, k):
        self.applications += 1
        a = self.algebra
        dx = a.add(a.product(k, self.hdag), a.product(self.h, mpo.adjoint(k)))
        return a.add(a.product(self.x, dx), a.product(dx, self.x))

    def adjoint(self, residual):
        self.adjoint_applications += 1
        a = self.algebra
        rh = a.add(residual, mpo.adjoint(residual), .5, .5)
        sym = a.add(a.product(self.x, rh), a.product(rh, self.x))
        return tn.scale(a.product(sym, self.h), 2.)


def jacobian_adjoint_mpo(h, residual, work_bond):
    return Jacobian(h, work_bond).adjoint(residual)


def _generator(rho, ham, jumps, algebra):
    a = algebra
    result = a.add(a.product(ham, rho), a.product(rho, ham), -1j, 1j)
    for jump in jumps:
        adj = mpo.adjoint(jump)
        number = a.product(adj, jump)
        gain = a.product(a.product(jump, rho), adj)
        loss = a.add(a.product(number, rho), a.product(rho, number), -.5, -.5)
        result = a.add(result, a.add(gain, loss))
    return result


def factor_velocity(h, hamiltonian_mpo, jumps_mpo, rcond=1e-11, work_bond=None, *,
                    tolerance=1e-8, maxiter=None, damping=0., preconditioner=None,
                    initial=None, strict=True, compression_tolerance=1e-10, check_every=5):
    """Solve original density-residual tangent LS using MPO-output LSQR.

    ``damping>0`` adds damping**2 * ||K||_F**2, explicitly changing the solve.
    A positive diagonal ``preconditioner`` rescales real tangent coordinates;
    a converged undamped LS objective is unchanged but its null-space
    minimum-norm convention can change. Default zero start and no scaling
    select the Euclidean minimum-norm solution in exact arithmetic.

    ``initial`` is a real coordinate vector in THIS state's tangent frame.
    Do not recycle coefficients from a different state's frame without a
    projection. Use TangentCoordinates(h).encode(previous_velocity) instead.

    ``rcond`` is accepted for integrator compatibility, not used as an SVD
    cutoff; LSQR has a distinct stopping tolerance. A default bond is allowed
    only for N<=8 finite validation; larger runs must declare their resources.
    ``strict=False`` retains failed diagnostics for research, not safe evolution.
    """
    started = perf_counter()
    if not 0 < rcond < 1 or not 0 < tolerance < 1:
        raise ValueError("rcond and tolerance must lie in (0,1)")
    if not np.isfinite(damping) or damping < 0:
        raise ValueError("damping must be finite and nonnegative")
    if h.n != hamiltonian_mpo.n or any(j.n != h.n for j in jumps_mpo):
        raise ValueError("Mismatched chain sizes")
    if work_bond is None:
        if h.n > 8:
            raise ValueError("Specify work_bond explicitly for N>8")
        work_bond = 4**(h.n//2)
    algebra = Algebra(work_bond, compression_tolerance, strict)
    tangent = TangentCoordinates(h)
    tangent_setup_seconds = perf_counter()-started
    jac = Jacobian(h, work_bond, algebra=algebra)
    rho = algebra.product(jac.x, jac.x)
    g = _generator(rho, hamiltonian_mpo, jumps_mpo, algebra)
    generator_norm = tn.norm(g)
    setup_seconds = perf_counter()-started
    ncoord = tangent.dimension
    p = np.ones(ncoord) if preconditioner is None else np.asarray(preconditioner, float)
    if p.shape != (ncoord,) or not np.isfinite(p).all() or np.any(p <= 0):
        raise ValueError("preconditioner must be a finite positive real-coordinate diagonal")
    if maxiter is None:
        maxiter = min(max(40, 4*ncoord), 600)
    if int(check_every) != check_every or check_every < 1:
        raise ValueError("check_every must be a positive integer")
    if int(maxiter) != maxiter or maxiter < 1:
        raise ValueError("maxiter must be a positive integer")
    y = np.zeros(ncoord) if initial is None else np.asarray(initial, float)/p
    if y.shape != (ncoord,) or not np.isfinite(y).all():
        raise ValueError("initial must contain finite real tangent coordinates")
    # An augmented MPO + coordinate codomain implements damping without
    # normal equations. The second block is empty when damping is zero.
    def apply(z):
        c = p*z
        return jac.apply(tangent.decode(c)), damping*c if damping else np.empty(0)
    def adjoint(value):
        out = tangent.encode(jac.adjoint(value[0]))
        if damping:
            out += damping*value[1]
        return p*out
    def norm(value):
        return float(np.hypot(tn.norm(value[0]), np.linalg.norm(value[1])))
    def scale(value, c):
        return tn.scale(value[0], c), c*value[1]
    def subtract(a, b, c=1.):
        return algebra.add(a[0], b[0], beta=-c), a[1]-c*b[1]
    target = (g, np.zeros(ncoord) if damping else np.empty(0))
    rhs_normal = tangent.encode(jac.adjoint(g))
    normal_scale = max(float(np.linalg.norm(rhs_normal)), np.finfo(float).tiny)
    u = target if initial is None else subtract(target, apply(y))
    beta = norm(u)
    v = adjoint(scale(u, 1/beta)) if beta else np.zeros(ncoord)
    u = scale(u, 1/beta) if beta else u
    alpha = float(np.linalg.norm(v))
    v = v/alpha if alpha else v
    w, rhobar, phibar = v.copy(), alpha, beta
    anorm2, ddnorm = alpha**2, 0.
    records, converged, reason = [], False, 'iteration_limit'
    force_norm, normal_norm, relative_normal = generator_norm, normal_scale, 1.
    if alpha == 0 or beta == 0:
        converged, reason = True, 'zero_initial_normal_residual'
    for iteration in range(1, 0 if converged else int(maxiter)+1):
        u = subtract(apply(v), u, alpha)
        beta = norm(u)
        if beta:
            u = scale(u, 1/beta)
        next_v = adjoint(u) - beta*v if beta else np.zeros(ncoord)
        alpha = float(np.linalg.norm(next_v))
        v = next_v/alpha if alpha else next_v
        anorm2 += beta**2 + alpha**2
        radius = float(np.hypot(rhobar, beta))
        if radius == 0:
            reason = 'bidiagonal_breakdown'
            break
        cosine, sine = rhobar/radius, beta/radius
        theta, rhobar = sine*alpha, -cosine*alpha
        phi, phibar = cosine*phibar, sine*phibar
        direction = w/radius
        y = y + phi*direction
        ddnorm += float(direction @ direction)
        w = v-theta*direction
        # The recurrence can be inaccurate under compression: independently
        # check when it predicts convergence and at fixed intervals, and
        # always reevaluate on exit. This avoids doubling every iteration's
        # expensive Jacobian applications.
        estimated_normal = alpha*abs(sine*phi)
        check = (iteration % check_every == 0 or iteration == maxiter
                 or estimated_normal <= tolerance*normal_scale
                 or alpha == 0 or beta == 0)
        record = {'iteration': iteration,
                  'normal_relative_estimate': estimated_normal/normal_scale,
                  'condition_estimate': float(np.sqrt(anorm2*ddnorm)),
                  'residual_recomputed': bool(check)}
        if check:
            c = p*y
            residual = algebra.add(jac.apply(tangent.decode(c)), g, beta=-1.)
            force_norm = tn.norm(residual)
            normal = tangent.encode(jac.adjoint(residual)) + damping**2*c
            normal_norm = float(np.linalg.norm(normal))
            relative_normal = normal_norm/normal_scale
            record.update(force_residual=force_norm, normal_relative=relative_normal)
        records.append(record)
        if not np.isfinite(y).all() or (check and not np.isfinite(relative_normal)):
            reason = 'nonfinite_iteration'
            break
        if check and relative_normal <= tolerance:
            converged, reason = True, 'stationarity_tolerance'
            break
    coefficients = p*y
    velocity = tangent.decode(coefficients)
    # Final residual is reevaluated for zero-step and breakdown cases too.
    applied = jac.apply(velocity)
    residual = algebra.add(applied, g, beta=-1.)
    force_norm = tn.norm(residual)
    normal = tangent.encode(jac.adjoint(residual)) + damping**2*coefficients
    normal_norm = float(np.linalg.norm(normal))
    relative_normal = normal_norm/normal_scale
    converged = bool(np.isfinite(relative_normal) and relative_normal <= tolerance)
    density_trace = float(mpo.trace(rho).real)
    trace_speed = float(mpo.trace(applied).real)
    normalization_speed = trace_speed/(4*density_trace) if density_trace > 0 else 0.
    normalized_residual = algebra.add(residual, rho, beta=-4*normalization_speed)
    info = {
        'backend': 'mpo_output_real_lsqr', 'tangent': 'one_site_fixed_bond',
        'real_tangent_dimension': ncoord, 'iterations': len(records),
        'converged': converged, 'least_squares_stationary': converged,
        'stopping_reason': reason, 'ls_tolerance': tolerance,
        'ls_rcond': None, 'rcond_compatibility_argument': rcond,
        'ls_normal_residual': normal_norm, 'ls_normal_relative': relative_normal,
        'stationarity_normalization': 'norm(B* J* G)',
        'ls_residual': force_norm, 'force_residual': force_norm,
        'relative_force_residual': force_norm/max(generator_norm, np.finfo(float).tiny),
        'physical_residual_small': bool(force_norm <= tolerance*max(generator_norm, np.finfo(float).tiny)),
        'generator_norm': generator_norm, 'velocity_norm': float(np.linalg.norm(coefficients)),
        'trace_velocity': trace_speed,
        'normalization_tangent_correction_norm': float(abs(normalization_speed)*tn.norm(h)),
        'normalized_force_residual': tn.norm(normalized_residual),
        'condition_estimate': float(np.sqrt(anorm2*ddnorm)),
        'condition_estimate_kind': 'LSQR bidiagonal estimate, not full Jacobian condition',
        'damping': damping, 'preconditioned': preconditioner is not None,
        'warm_started': initial is not None,
        'minimum_norm_convention': 'Euclidean at zero start without coordinate scaling',
        'factor_bond': h.bond_dimension(), 'root_product_bond': ts.root_mpo(h).bond_dimension(),
        'work_bond': int(work_bond),
        'product_local_discard_sum': float(sum(algebra.discards)),
        'product_local_discard_max': float(max(algebra.discards, default=0.)),
        'max_relative_local_discard': float(max(algebra.relative_discards, default=0.)),
        'compression_safe': bool(max(algebra.relative_discards, default=0.) <= compression_tolerance),
        'discard_interpretation': 'local allowances only; no propagated global certificate',
        'jacobian_applications': jac.applications, 'adjoint_applications': jac.adjoint_applications,
        'mpo_products': algebra.products, 'mpo_roundings': algebra.roundings,
        'setup_seconds': setup_seconds, 'tangent_setup_seconds': tangent_setup_seconds,
        'product_seconds': algebra.product_seconds, 'round_seconds': algebra.round_seconds,
        'solve_seconds': perf_counter()-started-setup_seconds,
        'seconds': perf_counter()-started, 'history': records,
        'coefficients': coefficients.tolist(),
    }
    if strict and not converged:
        raise SolverError('MPO LSQR did not meet its stationarity tolerance', info)
    return velocity, info
