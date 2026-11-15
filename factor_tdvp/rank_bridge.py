"""Dense, small-system support audits and finite CPTP factor bridges.

This module is deliberately *not* a scalable MPO support detector. It adapts
the innovation test and factor-SVD startup in automatic_rank_opening, adding
the extra square root needed for rho=(H H^dagger)^2. All rank thresholds,
channel roundoff corrections and removed numerical singular values are
reported. A bridge whose support remains unresolved is rejected explicitly.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm


def herm(a):
    return (a + a.conj().T) / 2


class RankOpeningError(RuntimeError):
    """A finite bridge could not establish a safe smooth-solver handoff."""

    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


def lindblad_rhs(rho, hamiltonian, jumps):
    out = -1j * (hamiltonian @ rho - rho @ hamiltonian)
    for jump in jumps:
        number = jump.conj().T @ jump
        out += jump @ rho @ jump.conj().T - (number @ rho + rho @ number) / 2
    return herm(out)


def lindblad_superoperator(hamiltonian, jumps):
    """Column-major vectorization: vec(A rho B)=(B^T tensor A)vec(rho)."""
    d = len(hamiltonian)
    identity = np.eye(d)
    generator = -1j * (np.kron(identity, hamiltonian)
                       - np.kron(hamiltonian.T, identity))
    for jump in jumps:
        number = jump.conj().T @ jump
        generator += (np.kron(jump.conj(), jump)
                      - (np.kron(identity, number) + np.kron(number.T, identity)) / 2)
    return generator


def _commuting_involution_kraus(hamiltonian, jumps, step):
    """Exact product channels for commuting Hermitian involution jumps.

    This avoids Choi-eigenvalue roundoff creating fake sectors in the
    correlated dephasing controls. Return None for other generators.
    """
    d = len(hamiltonian)
    identity = np.eye(d, dtype=complex)
    rates = []
    for jump in jumps:
        rate = float(np.trace(jump.conj().T @ jump).real / d)
        if rate == 0:
            rates.append(0.)
            continue
        scale = max(np.linalg.norm(jump), 1e-300)
        if np.linalg.norm(jump-jump.conj().T) > 1e-13 * scale:
            return None
        if np.linalg.norm(jump@jump-rate*identity) > 1e-13*rate*np.sqrt(d):
            return None
        if np.linalg.norm(hamiltonian@jump-jump@hamiltonian) > 1e-13*scale*max(1.,np.linalg.norm(hamiltonian)):
            return None
        rates.append(rate)
    for i, left in enumerate(jumps):
        for right in jumps[i+1:]:
            if np.linalg.norm(left@right-right@left) > 1e-13*max(1e-300,np.linalg.norm(left)*np.linalg.norm(right)):
                return None
    kraus = [expm(-1j*step*hamiltonian)]
    for jump, rate in zip(jumps, rates):
        if rate == 0:
            continue
        probability = -np.expm1(-2*rate*step)/2
        kraus = [np.sqrt(1-probability)*k for k in kraus] + [np.sqrt(probability/rate)*jump@k for k in kraus]
    return kraus


def exact_channel_kraus(hamiltonian, jumps, step, choi_rtol=None):
    """Kraus representation of exp(step*L), with explicit roundoff repair.

    For general generators a dense d^2 by d^2 Choi eigensolve is used.
    Eigensolver-scale Choi modes are removed, then completeness is restored
    by right multiplication by (sum K^dagger K)^(-1/2). The resulting map is
    CPTP; its measured superoperator discrepancy from exp(step*L) is returned.
    No claim of an exact floating-point channel or interval certificate is made.
    """
    if step < 0:
        raise ValueError("A physical channel bridge requires a nonnegative step.")
    hamiltonian = np.asarray(hamiltonian, dtype=complex)
    jumps = [np.asarray(jump, dtype=complex) for jump in jumps]
    d = len(hamiltonian)
    identity = np.eye(d, dtype=complex)
    if np.linalg.norm(hamiltonian-hamiltonian.conj().T) > 1e-12*max(1.,np.linalg.norm(hamiltonian)):
        raise ValueError("Hamiltonian must be Hermitian.")
    analytic = _commuting_involution_kraus(hamiltonian, jumps, step)
    if analytic is not None:
        completeness = sum((k.conj().T@k for k in analytic), np.zeros((d,d),complex))
        return analytic, dict(channel_backend="exact_commuting_involutions",
                             channel_completeness_error=float(np.linalg.norm(completeness-identity)),
                             channel_superoperator_error=0., channel_discretization_error=0.,
                             choi_discarded_norm=0., choi_negative_norm=0.,
                             kraus_count=len(analytic))
    channel = expm(step*lindblad_superoperator(hamiltonian, jumps))
    # C_(a,i),(b,j) = E_(a,b),(i,j), with each compound index column-major.
    choi = channel.reshape((d,d,d,d), order="F").transpose(0,2,1,3).reshape((d*d,d*d),order="F")
    values, vectors = np.linalg.eigh(herm(choi))
    threshold = ((32*np.finfo(float).eps*d*d) if choi_rtol is None else choi_rtol)*max(float(values[-1]),1.)
    if values[0] < -10*threshold:
        raise RankOpeningError("Exponential channel has non-roundoff negative Choi spectrum.",
                               dict(min_choi_eigenvalue=float(values[0]), choi_threshold=threshold))
    keep = values > threshold
    kraus = [(vectors[:,i]*np.sqrt(values[i])).reshape((d,d),order="F") for i in np.flatnonzero(keep)]
    completeness = sum((k.conj().T@k for k in kraus), np.zeros((d,d),complex))
    cval, cvec = np.linalg.eigh(herm(completeness))
    if cval[0] <= 0:
        raise RankOpeningError("Choi repair lost invertible completeness.", dict(min_completeness=float(cval[0])))
    correction = (cvec*(1/np.sqrt(cval)))@cvec.conj().T
    kraus = [k@correction for k in kraus]
    represented = sum((np.kron(k.conj(),k) for k in kraus),np.zeros_like(channel))
    completeness = sum((k.conj().T@k for k in kraus),np.zeros((d,d),complex))
    return kraus, dict(channel_backend="dense_exponential_choi",
                       channel_completeness_error=float(np.linalg.norm(completeness-identity)),
                       channel_superoperator_error=float(np.linalg.norm(represented-channel)),
                       channel_discretization_error=0.,
                       choi_discarded_norm=float(np.linalg.norm(values[~keep])),
                       choi_negative_norm=float(np.linalg.norm(values[values<0])),
                       choi_threshold=float(threshold), kraus_count=len(kraus))


def bridge_from_kraus(factor, kraus, amplitude_rtol=None):
    """Finite channel step from Z=[K_mu H H^dagger], using factor SVD.

    H_seed=U sqrt(Sigma) U^dagger, so H_seed H_seed^dagger=sqrt(Z Z^dagger).
    The cutoff is applied to Z amplitudes, not density eigenvalues. Only
    roundoff-scale SVD values are removed by default, and all removed mass
    and factor singular values are recorded. Caller must audit support.
    """
    factor = np.asarray(factor,complex)
    d = len(factor)
    if not kraus:
        raise ValueError("At least one Kraus operator is required.")
    root = factor@factor.conj().T
    stacked = np.column_stack([k@root for k in kraus])
    vectors, singular, _ = np.linalg.svd(stacked,full_matrices=False)
    threshold = ((32*np.finfo(float).eps*max(stacked.shape)) if amplitude_rtol is None else amplitude_rtol)*max(float(singular[0]),1e-300)
    keep = singular > threshold
    kept = np.where(keep,singular,0.)
    seed = herm((vectors*np.sqrt(kept))@vectors.conj().T)
    rebuilt_root = seed@seed.conj().T
    channel_density = stacked@stacked.conj().T
    completeness = sum((k.conj().T@k for k in kraus),np.zeros((d,d),complex))
    return seed, dict(bridge_factor_rank=int(sum(keep)),
                     bridge_factor_singular_values=np.sqrt(kept).tolist(),
                     bridge_removed_factor_singular_values=np.sqrt(singular[~keep]).tolist(),
                     bridge_amplitude_threshold=float(threshold),
                     bridge_discarded_root_frobenius=float(np.linalg.norm(singular[~keep])),
                     bridge_discarded_density_trace=float(sum(singular[~keep]**2)),
                     bridge_density_reconstruction_error=float(np.linalg.norm(rebuilt_root@rebuilt_root-channel_density)),
                     bridge_trace=float(np.trace(rebuilt_root@rebuilt_root).real),
                     bridge_kraus_completeness_error=float(np.linalg.norm(completeness-np.eye(d))))


def _orthogonal_append(basis, vector, tolerance):
    residual = vector.astype(complex).copy()
    for _ in range(2):
        for item in basis:
            residual -= item*np.vdot(item,residual)
    norm = np.linalg.norm(residual)
    if norm <= tolerance:
        return False
    basis.append(residual/norm)
    return True


def _reachable_rank(support, hamiltonian, jumps, tolerance=1e-10):
    """Numerical span of time-ordered jump words in the no-jump frame.

    Generate the ad_A-invariant operator span of jumps, A=-iH-sum L†L/2,
    then close the input support under these operators. Unlike closure under
    H itself this leaves a rotating pure state under unitary evolution rank 1.
    Finite precision and thresholded spans make this a diagnostic, not an
    exact symbolic rank proof. It is conservative about unresolved sectors.
    """
    d = len(hamiltonian)
    if support.shape[1] == d or not jumps:
        return support.shape[1]
    number = sum((j.conj().T@j for j in jumps),np.zeros((d,d),complex))
    drift = -1j*hamiltonian-number/2
    operators = []
    for jump in jumps:
        norm = np.linalg.norm(jump)
        if norm:
            _orthogonal_append(operators,(jump/norm).reshape(-1),tolerance)
    index = 0
    drift_scale = max(np.linalg.norm(drift,2),1e-300)
    while index < len(operators) and len(operators)<d*d:
        operator = operators[index].reshape((d,d))
        commutator = (drift@operator-operator@drift)/drift_scale
        _orthogonal_append(operators,commutator.reshape(-1),tolerance)
        index += 1
    vectors = [support[:,i].copy() for i in range(support.shape[1])]
    index = 0
    while index < len(vectors) and len(vectors)<d:
        vector = vectors[index]
        for operator in operators:
            _orthogonal_append(vectors,operator.reshape((d,d))@vector,tolerance)
            if len(vectors) == d:
                break
        index += 1
    return len(vectors)


def audit_support(factor, hamiltonian, jumps, step=0.01, factor_rtol=1e-9,
                  leakage_tol=1e-12, probe=True):
    """Audit instantaneous leakage, delayed opening and full-rank stiffness.

    Rank is measured on H singular values, whose fourth powers are density
    populations. A finite-channel probe compares ranks, not outside-old-
    support weight, and therefore does not mistake coherent rotation for
    rank opening. The jump-word diagnostic catches unresolved delayed modes.
    """
    factor = np.asarray(factor,complex)
    hamiltonian = np.asarray(hamiltonian,complex)
    d = len(factor)
    vectors, singular, _ = np.linalg.svd(factor,full_matrices=True)
    keep = singular > factor_rtol*max(float(singular[0]),1e-300)
    support = vectors[:,keep]
    null = vectors[:,~keep]
    root = factor@factor.conj().T
    density = root@root
    innovations = [null.conj().T@jump@root for jump in jumps]
    leakage = float(sum(np.linalg.norm(item)**2 for item in innovations))
    rate = float(sum(np.linalg.norm(jump@root)**2 for jump in jumps))
    first = leakage > leakage_tol*max(1.,rate)
    rank = int(sum(keep))
    reachable = _reachable_rank(support,hamiltonian,jumps)
    probe_rank = rank
    probe_tail = 0.
    probe_channel_error = 0.
    if probe and rank<d and reachable>rank and step>0:
        kraus, channel_info = exact_channel_kraus(hamiltonian,jumps,step)
        seed, _ = bridge_from_kraus(factor,kraus)
        probe_singular = np.linalg.svd(seed,compute_uv=False)
        probe_rank = int(sum(probe_singular>factor_rtol*max(probe_singular[0],1e-300)))
        probe_tail = float(np.linalg.norm(probe_singular[rank:]**2))
        probe_channel_error = channel_info["channel_superoperator_error"]
    delayed = rank<d and not first and (probe_rank>rank or reachable>rank)
    rhs = lindblad_rhs(density,hamiltonian,jumps)
    diagonal = np.real(np.diag(vectors.conj().T@rhs@vectors))
    exposure = float(np.max(step*np.abs(diagonal[keep])/np.maximum(singular[keep]**4,1e-300),initial=0.))
    if first:
        mode = "first_order_opening"
    elif delayed:
        mode = "delayed_opening" if probe_rank>rank else "delayed_opening_unresolved"
    elif rank==d and exposure>.25:
        mode = "full_rank_ill_conditioned"
    else:
        mode = "smooth_full_rank" if rank==d else "smooth_rank_preserving"
    return dict(mode=mode,rank=rank,dimension=d,factor_rtol=factor_rtol,
                minimum_factor_singular_value=float(singular[-1]),
                discarded_support_mass=float(sum(singular[~keep]**4)),
                leakage_rate=leakage,first_order_opening=bool(first),
                delayed_opening=bool(delayed),reachable_rank=reachable,
                finite_probe_rank=probe_rank,finite_probe_root_tail=probe_tail,
                probe_channel_superoperator_error=probe_channel_error,
                near_boundary_exposure=exposure,
                safe_handoff=bool(not first and not delayed))


def bridge_factor(factor, hamiltonian, jumps, step, factor_rtol=1e-9,
                  leakage_tol=1e-12, require_safe=True):
    """Exact finite physical channel followed by a checked factor-SVD seed.

    This is a small-system dense event handler. It advances physical time
    by ``step`` and must never be mistaken for a zero-time retraction.
    Missing multi-jump/delayed sectors after Choi/SVD roundoff cause an
    explicit guarded failure, not a singular smooth step.
    """
    before = audit_support(factor,hamiltonian,jumps,step,factor_rtol,leakage_tol,probe=False)
    kraus, channel_info = exact_channel_kraus(hamiltonian,jumps,step)
    seed, seed_info = bridge_from_kraus(factor,kraus)
    after = audit_support(seed,hamiltonian,jumps,step,factor_rtol,leakage_tol,probe=False)
    info = dict(**channel_info,**seed_info,step=float(step),before=before,after=after,
                safe_handoff=after["safe_handoff"],
                support_rank_before=before["rank"],support_rank_after=after["rank"])
    if require_safe and not info["safe_handoff"]:
        raise RankOpeningError("Finite bridge left unresolved opening; increase/compose the physical bridge or raise accuracy.",info)
    return seed, info
