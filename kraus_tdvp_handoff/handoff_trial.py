"""Compare buffered Kraus evolution and an audited two-site CP-TDVP handoff."""
import tdvp_core as tn  # Set the BLAS thread count before importing numpy.
import argparse
import math
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import expm_multiply

import buffered_kraus_cp as kraus
import large_mpo_tdvp as observables
import mpo_cp_force as mpo
import test_automatic_rank_opening as models


def matrix(state):
    return tn.to_matrix(state.dense(), state.n)


def exact_density(initial, model, elapsed):
    hamiltonian = sparse.csr_matrix(model.h)
    identity = sparse.eye(hamiltonian.shape[0], format='csr')
    generator = -1j * (sparse.kron(identity, hamiltonian) -
                       sparse.kron(hamiltonian.T, identity))
    for jump in map(sparse.csr_matrix, model.jumps):
        number = jump.conj().T @ jump
        generator += sparse.kron(jump.conj(), jump) - (
            sparse.kron(identity, number) + sparse.kron(number.T, identity)
        ) / 2
    q = matrix(initial)
    rho = q @ q.conj().T
    evolved = expm_multiply(elapsed * generator.tocsr(), rho.flatten(order='F'))
    rho = evolved.reshape(q.shape, order='F')
    assert abs(np.trace(rho) - 1) < 1e-11
    assert np.linalg.norm(rho - rho.conj().T) < 1e-11
    values, vectors = np.linalg.eigh((rho + rho.conj().T) / 2)
    assert values[0] > -1e-13
    return rho, (vectors * np.sqrt(np.maximum(values, 0))) @ vectors.conj().T


def kraus_step(state, step, buffer, chi):
    rates = .3 + .12 * np.cos(np.arange(state.n))**2
    half, _ = kraus.hamiltonian_step(state, step / 2, chi)
    channels = [kraus.kraus('loss', rate, step) for rate in rates]
    half, _ = kraus.noise_sweep(half, channels, buffer, chi)
    out, _ = kraus.hamiltonian_step(half, step / 2, chi)
    return kraus.trace_optimize(out, buffer, chi, sweeps=2)[0]


def local_gap(first, second):
    return max(observables.trace_distance(
        observables.doubled_marginal(first, site, 2),
        observables.doubled_marginal(second, site, 2)
    ) for site in range(first.n - 1))


def stable_kraus(state, step, chi, tolerance, stats, depth=0):
    previous = kraus_step(state, step, 1, chi)
    stats['kraus_trials'] += 1
    for buffer in range(2, min(3, state.n-1) + 1):
        candidate = kraus_step(state, step, buffer, chi)
        stats['kraus_trials'] += 1
        gap = local_gap(candidate, previous)
        if gap <= tolerance:
            stats['max_buffer_gap'] = max(gap, stats['max_buffer_gap'])
            return candidate
        previous = candidate
    if depth >= 8:
        raise RuntimeError('Kraus reference exceeded its subdivision limit')
    stats['kraus_subdivisions'] += 1
    half = stable_kraus(state, step/2, chi, tolerance, stats, depth+1)
    return stable_kraus(half, step/2, chi, tolerance, stats, depth+1)


class DenseForce:
    """Small-chain control for the cost and conditioning of the MPO force."""
    def __init__(self, model, chi):
        self.model = model
        self.chi = chi

    def __call__(self, t, state):
        q = matrix(state)
        q = (q + q.conj().T)/2
        values, basis = np.linalg.eigh(q)
        if values[0] <= 1e-13:
            raise RuntimeError('Dense force outside strictly positive support')
        rho = q @ q
        noise = np.zeros_like(rho)
        for jump in self.model.jumps:
            number = jump.conj().T @ jump
            noise += jump @ rho @ jump.conj().T - (number @ rho + rho @ number)/2
        rhs = basis.conj().T @ noise @ basis
        derivative = basis @ (rhs / (values[:,None] + values[None,:])) @ basis.conj().T
        derivative -= 1j * (self.model.h @ q - q @ self.model.h)
        return tn.from_dense(tn.to_interleaved(derivative, state.n), state.n, self.chi), {}


def tdvp_step(state, t, step, chi, provider):
    # The sqrt(t) clock regularizes the post-startup midpoint quadrature.
    u = np.sqrt(t)
    du = step / (np.sqrt(t + step) + u)
    def force(v, current):
        derivative, info = provider(v*v, current)
        residual = info.get('true_residual', 0.)
        if not np.isfinite(residual) or residual > 1e-5:
            raise RuntimeError(f'Sylvester relative residual failed the check ({residual:.3g})')
        return tn.scale(derivative, 2*v), info
    candidate, _ = tn.midpoint_step(state, u, du, force, chi,
                                     observables.retraction(chi), mode='two')
    return candidate


def run(n, chi, mode, steps, tolerance, audit_interval, elapsed, buffer, force_chi,
        tdvp_substeps=1):
    start = time.perf_counter()
    initial = mpo.product_root(n, 1.)
    state = initial
    stats = dict(kraus_trials=0, kraus_subdivisions=0, max_buffer_gap=0.,
                 tdvp_accepted=0, tdvp_rejected=0, solver_rejected=0, audits=0)
    rows = []
    force = None
    if mode == 'mpo':
        rates = .3 + .12 * np.cos(np.arange(n))**2
        force = mpo.MPOForce(n, rates, force_chi, 1e-6, maxiter=60)
    elif mode == 'dense':
        force = DenseForce(models.model(n), force_chi)
    audit_due = True
    step = elapsed / steps
    for index in range(steps):
        detail = dict(step=index+1, time=(index+1)*step)
        if mode == 'fixed':
            state = kraus_step(state, step, buffer, chi)
            stats['kraus_trials'] += 1
            detail['action'] = 'fixed_kraus'
        elif mode == 'adaptive' or index == 0:
            state = stable_kraus(state, step, chi, tolerance, stats)
            detail['action'] = 'kraus'
        else:
            require_check = audit_due or (index-1) % audit_interval == 0
            try:
                trial = state
                microstep = step / tdvp_substeps
                for substep in range(tdvp_substeps):
                    trial = tdvp_step(trial, index*step + substep*microstep, microstep, chi, force)
            except (RuntimeError, FloatingPointError, np.linalg.LinAlgError) as error:
                detail['solver_message'] = str(error)
                trial = None
                stats['solver_rejected'] += 1
            if require_check or trial is None:
                reference = stable_kraus(state, step, chi, tolerance, stats)
                stats['audits'] += 1
                gap = local_gap(reference, trial) if trial is not None else math.inf
                detail['local_trial_gap'] = gap
                if not np.isfinite(gap) or gap > tolerance:
                    state = reference
                    if mode == 'mpo':
                        force.previous_noise = None
                    stats['tdvp_rejected'] += 1
                    audit_due = True
                    detail['action'] = 'fallback'
                    rows.append(detail)
                    continue
            state = trial
            stats['tdvp_accepted'] += 1
            audit_due = False
            detail['action'] = 'tdvp'
        rows.append(detail)
    seconds = time.perf_counter() - start
    # Exact global information is accessed only after the controller has finished.
    rho, exact = exact_density(initial, models.model(n), elapsed)
    q = matrix(state)
    left, singular, _ = np.linalg.svd(q)
    own_root = (left * singular) @ left.conj().T
    summary = dict(mode=mode, n=n, chi=chi, force_chi=force_chi, steps=steps,
                   elapsed=elapsed, buffer=buffer, tolerance=tolerance, cadence=audit_interval,
                   tdvp_substeps=tdvp_substeps, seconds=seconds, global_cp_error=np.linalg.norm(q-exact),
                   branch_error=np.linalg.norm(q-own_root),
                   physical_trace_distance=observables.trace_distance(q@q.conj().T, rho), **stats)
    return summary, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['mpo','dense','adaptive','fixed'], required=True)
    parser.add_argument('--n', type=int, default=4)
    parser.add_argument('--chi', type=int, default=16)
    parser.add_argument('--force-chi', type=int, default=16)
    parser.add_argument('--time', type=float, default=.1)
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--buffer', type=int, default=2)
    parser.add_argument('--tol', type=float, default=1e-4)
    parser.add_argument('--cadence', type=int, default=4)
    parser.add_argument('--rep', type=int, default=0)
    parser.add_argument('--tdvp-substeps',type=int,default=1)
    parser.add_argument('--out', type=Path, default=Path(__file__).parent/'handoff_results')
    args = parser.parse_args()
    if min(args.steps, args.chi, args.force_chi, args.cadence, args.tdvp_substeps) < 1:
        parser.error('Step counts, cadence and bond dimensions must be positive.')
    if args.n < 3 or args.time <= 0 or args.tol <= 0 or args.buffer < 0:
        parser.error('The handoff benchmark requires at least three sites, positive time and tolerance, and a nonnegative buffer.')
    args.out.mkdir(parents=True, exist_ok=True)
    summary, rows = run(args.n,args.chi,args.mode,args.steps,args.tol,args.cadence,
                        args.time,args.buffer,args.force_chi,args.tdvp_substeps)
    tag = f'{args.mode}_n{args.n}_c{args.chi}_f{args.force_chi}_T{args.time}_s{args.steps}_b{args.buffer}_tol{args.tol}_a{args.cadence}_r{args.rep}'
    if args.tdvp_substeps != 1:
        tag += f'_sub{args.tdvp_substeps}'
    observables.save(args.out/f'{tag}.csv',[summary])
    observables.save(args.out/f'{tag}_history.csv',rows)
    print(summary, flush=True)


if __name__ == '__main__':
    main()
