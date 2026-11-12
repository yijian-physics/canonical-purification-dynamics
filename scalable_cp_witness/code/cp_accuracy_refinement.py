"""Refine a TDVP endpoint against an independently propagated physical MPDO.

The dense backend is a small-chain control. The Chebyshev backend operates on
spin MPOs and requires a verified positive spectral enclosure. Oracle roots
are confined to benchmark scoring and initialization of the pure-state bridge.
"""
import argparse
import copy
import csv
from pathlib import Path

import guarded_cp_tdvp as g
import mpo_cp_force as mpo
import tdvp_core as tn
import numpy as np
from critical_uniform_tdvp import CurrentForce, channel_root, critical_ground, polar_retractor
from guarded_force_trials import root_mpo
from guarded_trajectories import correlated_input


HERE = Path(__file__).resolve().parent
INPUTS = HERE.parent / 'inputs' if (HERE.parent / 'inputs').is_dir() else HERE
OUT = HERE.parent if INPUTS != HERE else HERE.parent / 'outputs' / 'cp_accuracy_refinement'


def write_rows(name, rows):
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / (name + '.csv')).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def checkpoint(name, directory='guarded_final'):
    path = INPUTS / directory / name
    with path.with_name(path.name + '.csv').open() as handle:
        elapsed = float(next(csv.DictReader(handle))['time'])
    with np.load(path.with_name(path.name + '_last_accepted.npz')) as data:
        tensors = [data[f'a{j}'] for j in range(len(data.files))]
    state = tn.MPS(len(tensors), max_bond=max(a.shape[-1] for a in tensors))
    state.tensors, state.center = tensors, state.n - 1
    return elapsed, state


def matrix(state):
    return np.real_if_close(tn.to_matrix(state.dense(), state.n))


def rounded(matrix_in, n, bond):
    state = root_mpo(matrix_in, n, bond)
    state.tensors[-1] /= np.linalg.norm(state.tensors[-1])
    return state


def critical_data(n):
    psi, _, signs = critical_ground(n)
    distances = np.sum(signs[:, None, :] != signs[None, :, :], axis=-1)
    pure = np.outer(psi, psi.conj())
    return psi, signs, lambda t: pure * np.exp(-t * distances)


def dense_correct(state, density, state_bond, update_bond, passes=5, tolerance=0.):
    """Try regularized Newton updates and keep the best certified endpoint."""
    candidate = matrix(state)
    bound = g.endpoint_dense(candidate, density)['cp_bound']
    history = []
    for _ in range(passes):
        if bound <= tolerance:
            break
        values, basis = np.linalg.eigh((candidate + candidate.conj().T) / 2)
        values = np.maximum(values, 0.)
        positive = (basis * values) @ basis.conj().T
        residual = basis.conj().T @ density @ basis - np.diag(values**2)
        # The shifted solve also has a defined value on a rounded null space.
        scale = max(values[-1], np.finfo(float).tiny)
        floor = max(0., 1e-12 * scale - values[0])
        accepted = False
        for shift in sorted(set((floor, max(floor, .01*bound), max(floor, .1*bound)))):
            delta = residual / (values[:, None] + values[None, :] + 2*shift)
            if np.linalg.norm(delta) > 2*bound:
                continue
            update = basis @ delta @ basis.conj().T
            if update_bond < 4**(state.n//2):
                update = matrix(root_mpo(update, state.n, update_bond))
            trial = rounded(positive + (update + update.conj().T)/2, state.n, state_bond)
            trial_matrix = matrix(trial)
            next_bound = g.endpoint_dense(trial_matrix, density)['cp_bound']
            if next_bound < bound * (1 - 1e-7):
                state, candidate, bound = trial, trial_matrix, next_bound
                history.append(bound)
                accepted = True
                break
        if not accepted:
            break
    return state, bound, history


def dense_sweep(name, filename, directory='guarded_final'):
    t, initial = checkpoint(name, directory)
    psi, signs, physical = critical_data(initial.n)
    density = physical(t)
    reference = channel_root(psi[:, None], signs, t)
    paired = tn.to_interleaved(reference, initial.n)
    spectra = [np.linalg.svd(paired.reshape(4**j, -1), compute_uv=False)
               for j in range(1, initial.n)]
    rows = []
    for state_bond, update_bond in ((16,256),(32,256),(64,8),(64,16),(64,32),
                                    (64,64),(64,256),(128,256),(256,256)):
        initial_state = rounded(matrix(initial), initial.n, state_bond)
        start = matrix(initial_state)
        start_bound = g.endpoint_dense(start, density)['cp_bound']
        final, bound, steps = dense_correct(initial_state, density, state_bond, update_bond)
        error = float(np.linalg.norm(matrix(final)-reference))
        assert error <= bound + 2e-10
        witness = lower_bond_witness(final, bound, 64)
        assert witness <= max(np.linalg.norm(s[64:]) for s in spectra) + 2e-10
        rows.append(dict(time=t, state_bond=state_bond, update_bond=update_bond,
                         start_error=np.linalg.norm(start-reference), start_bound=start_bound,
                         passes=len(steps), cp_error=error, cp_bound=bound,
                         schmidt_lower=max(np.linalg.norm(s[state_bond:]) for s in spectra),
                         lower_for_bond64_from_this_state=witness,
                         exact_root_ttsvd_error=np.linalg.norm(
                             matrix(rounded(reference, initial.n, state_bond))-reference)))
        write_rows(filename, rows)
        print(filename, state_bond, update_bond, error, bound, flush=True)


def chebyshev_solve(herm, source, lower, upper, bond, iterations, frame=None):
    """Approximately solve H D + D H = source without forming an inverse."""
    if not (0 < lower <= upper) or iterations < 1:
        raise ValueError('Positive spectral enclosure and at least one iteration required')
    alg = g.MPOAlgebra(bond)
    defect, _ = tn.compress(source, bond)
    low, high = 2*lower, 2*upper
    if frame is not None:
        # P(X)=W X W*. P A_H is similar to a self-adjoint positive operator.
        # Its interval follows by comparison with the product W^{-2} root.
        cond = np.prod([np.linalg.cond(w) for w in frame.matrices])
        low = 2*frame.c
        high = (1+frame.defect)*(cond + 1/cond)
        defect = g.whiten(defect, frame.matrices)
    center, radius = (high+low)/2, (high-low)/2
    direction = alg.scale(defect, 1/center)
    update = direction
    if radius <= 1e-14*center:
        return update
    previous = radius/center
    for _ in range(1, iterations):
        left = alg.product(herm, update)
        right = alg.product(update, herm)
        applied = alg.combine(left, right, 1., 1.)
        if frame is not None:
            applied = g.whiten(applied, frame.matrices)
        residual = alg.combine(defect, applied, 1., -1.)
        coefficient = 1/(2*center/radius - previous)
        direction = alg.combine(residual, direction, 2*coefficient/radius, coefficient*previous)
        update = alg.combine(update, direction, 1., 1.)
        previous = coefficient
    return update


def hermitian_round(state, bond, normalize=False):
    """Round Re(Pauli coefficients) using real tensors to preserve Hermiticity."""
    pauli = np.stack([mpo.I, mpo.X, np.array([[0,-1j],[1j,0]]), mpo.Z]).reshape(4,4)/np.sqrt(2)
    real_state = tn.MPS(state.n, max_bond=2*state.max_bond)
    real_state.tensors = []
    for j, a in enumerate(state.tensors):
        coef = np.einsum('ms,asb->amb', pauli.conj(), a)
        x, y = coef.real, coef.imag
        if state.n == 1:
            block = x
        elif j == 0:
            block = np.concatenate([x,y], axis=2)
        elif j == state.n-1:
            block = np.concatenate([x,-y], axis=0)
        else:
            block = np.concatenate([np.concatenate([x,y],axis=2),
                                    np.concatenate([-y,x],axis=2)], axis=0)
        real_state.tensors.append(block)
    real_state.center = state.n-1
    result, _ = tn.compress(real_state, bond, normalize=normalize)
    result.tensors = [np.einsum('ms,amb->asb', pauli, a) for a in result.tensors]
    return result


def correlated_data(n):
    initial, prep_error = correlated_input(n, .12, .35)
    raw = mpo.multiply(initial, mpo.adjoint(initial))
    density0, info = tn.compress(raw, 16)
    trace_error = (2.**(n/2)*info['discarded_bound'] + prep_error*(2+prep_error))
    anchor = (initial, .44**(n/2)-prep_error, .56**(n/2)+prep_error)
    if n >= 32:
        w = np.diag(np.array([.56,.44])**(-.25))
        h = np.array([[1.,1.],[1.,-1.]])/np.sqrt(2)
        hh, ww = np.kron(h,h), np.kron(w,w)
        gate = hh @ np.diag([1.,1.,1.,np.exp(.35j)]) @ hh
        scales = np.linalg.svd(ww @ gate @ np.linalg.inv(ww), compute_uv=False)
        allowance = prep_error*np.linalg.norm(w,2)**(2*n)

    def physical(t):
        density = copy.deepcopy(density0)
        scale = np.array([1., np.exp(-t), np.exp(-t), 1.])
        density.tensors = [a*scale[None, :, None] for a in density.tensors]
        return density

    def frame_for(state):
        herm = hermitian_round(state, state.bond_dimension())
        if n >= 32:
            frame = g.ProductFrame.from_anchor(herm, initial, [w]*n,
                                scales[-1]**(2*(n-1))-allowance,
                                scales[0]**(2*(n-1))+allowance)
        else:
            frame = g.ProductFrame(herm, anchor=anchor)
        return frame

    def audit(state, density):
        frame = frame_for(state)
        bound = frame.endpoint(density, trace_error)['cp_bound'] + tn.distance(frame.root, state)
        return frame, bound
    return initial, physical, audit, frame_for


def mpo_trial(frame, density, state_bond, work_bond, iterations, precondition=False):
    square = g.exact_product(frame.root, frame.root, frame.max_elements)
    defect = tn.add(density, square, beta=-1.)
    update = chebyshev_solve(frame.root, defect, frame.lower, frame.upper, work_bond, iterations,
                             frame if precondition else None)
    return hermitian_round(tn.add(frame.root, update), state_bond, normalize=True)


def mpo_refine(n, state_bond, work_bond, iterations, passes=3, tolerance=0.):
    names = {16: 'correlated_n16_chi6_s5_tol0.002', 32: 'correlated_n32_chi6_s2_tol0.002_prep'}
    elapsed, state = checkpoint(names[n])
    _, physical, audit, _ = correlated_data(n)
    density = physical(elapsed)
    frame, bound = audit(state, density)
    rows = []
    def record(index, trial_bound, status):
        rows.append(dict(n=n, time=elapsed, state_bond=state_bond, work_bond=work_bond,
                         iterations=iterations, correction=index, trial_bound=trial_bound,
                         kept_bound=bound, status=status))
        file = f'mpo_n{n}_state{state_bond}_work{work_bond}_k{iterations}'
        write_rows(file, rows)
        print(file, index, trial_bound, status, flush=True)
    record(0, bound, 'input')
    for step in range(1, passes+1):
        if bound <= tolerance:
            break
        try:
            trial = mpo_trial(frame, density, state_bond, work_bond, iterations)
            trial_frame, trial_bound = audit(trial, density)
        except g.Rejected as failure:
            record(step, '', failure.reason)
            break
        if trial_bound >= bound:
            record(step, trial_bound, 'no_improvement')
            break
        state, frame, bound = trial, trial_frame, trial_bound
        record(step, trial_bound, 'improved')
    return state, bound


def lower_bond_witness(state, bound, smaller_bond):
    """Lower bound on the error attainable at smaller_bond, without the exact CP."""
    canonical = copy.deepcopy(state)
    canonical.center = canonical.n-1
    canonical.move_center(0)
    tail = 0.
    for j in range(canonical.n-1):
        a = canonical.tensors[j]
        singular = np.linalg.svd(a.reshape(-1, a.shape[-1]), compute_uv=False)
        tail = max(tail, np.linalg.norm(singular[smaller_bond:]))
        canonical.move_center(j+1)
    return max(0., tail - bound)


def refine_mpo_to_tolerance(state, density, audit, tolerance, levels, passes, precondition):
    """No exact root is used. Hold the best verified candidate across cap changes."""
    frame, bound = audit(state, density)
    history = []
    for bond, work_bond, iterations in levels:
        if bound <= tolerance:
            break
        for _ in range(passes):
            try:
                trial = mpo_trial(frame, density, bond, work_bond, iterations, precondition)
                trial_frame, next_bound = audit(trial, density)
            except g.Rejected as failure:
                history.append((bond,work_bond,iterations,failure.reason,bound))
                break
            if not np.isfinite(next_bound) or next_bound >= bound:
                history.append((bond,work_bond,iterations,'no_improvement',bound))
                break
            state, frame, bound = trial, trial_frame, next_bound
            history.append((bond,work_bond,iterations,'improved',bound))
            if bound <= tolerance:
                break
    return state, bound, history


def mpo_trajectory(n, stop, intervals, tolerance, max_bond, passes, precondition=True):
    state, physical, audit, frame_for = correlated_data(n)
    name = f'mpo_trajectory_n{n}_s{intervals}_chi{max_bond}_tol{tolerance:g}'
    if precondition:
        name += '_frame'
    records = []
    trials = []
    def record(step, proposed_time, accepted_time, status, prediction, bound, bond, attempts):
        error = ''
        if status == 'accepted':
            np.savez_compressed(OUT / (name+'_last_accepted.npz'), time=accepted_time,
                                cp_bound=bound, **{f'a{j}':a for j,a in enumerate(state.tensors)})
            if n <= 8:
                values, basis = np.linalg.eigh(matrix(physical(accepted_time)))
                reference = (basis*np.sqrt(np.maximum(values,0))) @ basis.conj().T
                error = np.linalg.norm(matrix(state)-reference)
                assert error <= bound + 2e-11
        records.append(dict(step=step, proposed_time=proposed_time, accepted_time=accepted_time,
                            status=status, predictor_bound=prediction, cp_bound=bound,
                            measured_cp_error=error, stored_bond=bond, correction_trials=attempts))
        write_rows(name, records)
        print(name, step, status, accepted_time, bound, flush=True)
    OUT.mkdir(parents=True,exist_ok=True)
    try:
        _, initial_bound = audit(state, physical(0.))
    except g.Rejected as failure:
        record(0,0.,'',failure.reason,'','',state.bond_dimension(),0)
        return failure.reason
    initial_status = 'accepted' if initial_bound <= tolerance else 'initial_tolerance'
    record(0, 0., 0. if initial_status=='accepted' else '', initial_status, initial_bound,
           initial_bound, state.bond_dimension(), 0)
    if initial_status != 'accepted':
        return initial_status
    for step in range(1, intervals+1):
        t0, t1 = stop*(step-1)/intervals, stop*step/intervals
        density = physical(t1)
        pred_bond = min(max_bond, max(6, state.bond_dimension()))
        def force(t, current):
            frame = frame_for(current)
            square = g.exact_product(frame.root, frame.root, frame.max_elements)
            square, _ = tn.compress(square, 2*pred_bond)
            rhs = mpo.onsite_dissipation(square, np.full(n,.5), 'dephasing')
            force_mpo = chebyshev_solve(frame.root, rhs, frame.lower, frame.upper,
                                        pred_bond, 4, frame if precondition else None)
            return force_mpo, {}
        def retract(candidate):
            return hermitian_round(candidate, pred_bond, normalize=True), {}
        try:
            prediction, _ = tn.midpoint_step(state, t0, t1-t0, force, pred_bond,
                                              retract=retract, mode='two')
            _, prediction_bound = audit(prediction, density)
        except g.Rejected as failure:
            # A failed predictor can still use the previous accepted state as a warm start.
            prediction = state
            _, prediction_bound = audit(prediction, density)
            trials.append(dict(step=step, state_bond=pred_bond, work_bond=pred_bond,
                               iterations=4, result='predictor_'+failure.reason, bound=prediction_bound))
        levels = [(b,w,k) for b,w,k in [(6,8,12),(6,12,24),(8,16,24),(12,20,32)]
                  if pred_bond <= b <= max_bond]
        best, bound, history = refine_mpo_to_tolerance(prediction, density, audit,
                                                        tolerance, levels, passes, precondition)
        for b,w,k,result,value in history:
            trials.append(dict(step=step, state_bond=b, work_bond=w, iterations=k,
                               result=result, bound=value))
        if trials:
            write_rows(name+'_trials',trials)
        if bound > tolerance:
            record(step,t1,t0,'budget_exhausted',prediction_bound,
                   records[-1]['cp_bound'],state.bond_dimension(),len(history))
            return 'budget_exhausted'
        state = best
        record(step,t1,t1,'accepted',prediction_bound,bound,state.bond_dimension(),len(history))
    return 'accepted'


def trajectory(n, bond, intervals, correct, predict=True, force_bond=0):
    psi, signs, physical = critical_data(n)
    start, stop = np.sqrt(.01), np.sqrt(.3)
    state = rounded(channel_root(psi[:, None], signs, start**2), n, bond)
    exact_force, retract = CurrentForce(n, signs, 'sqrt'), polar_retractor(n, bond)
    def force(t, current):
        vector, info = exact_force(t, current)
        if force_bond:
            vector, _ = tn.compress(vector, force_bond)
        return vector, info
    grid = np.linspace(start, stop, intervals+1)
    rows = []
    for index, (u,v) in enumerate(zip(grid, grid[1:]), 1):
        trial = state
        if predict:
            trial, _ = tn.midpoint_step(state, u, v-u, force, bond, retract=retract, mode='two')
        density = physical(v*v)
        before = g.endpoint_dense(matrix(trial), density)['cp_bound']
        if correct:
            state, bound, steps = dense_correct(trial, density, bond, 4**(n//2), passes=4)
        else:
            state, bound, steps = trial, before, []
        error = np.linalg.norm(matrix(state) - channel_root(psi[:, None], signs, v*v))
        assert error <= bound + 2e-10
        rows.append(dict(n=n, bond=bond, force_bond=force_bond, intervals=intervals,
                         corrected=correct, predictor=predict, step=index,
                         time=v*v, prediction_bound=before, corrections=len(steps),
                         cp_bound=bound, cp_error=error))
        file = f'trajectory_n{n}_chi{bond}_s{intervals}_correct{int(correct)}'
        if not predict:
            file += '_no_predictor'
        if force_bond:
            file += f'_force{force_bond}'
        write_rows(file, rows)
        print(file, index, error, bound, flush=True)


def resume_critical(tolerance=1e-5):
    """Resume the earlier bond-16 sign-force run using the same provisional force."""
    elapsed, state = checkpoint('critical_n8_chi64_s2_tol0.005_trialbond16_T0.012',
                                'guarded_final2')
    n = state.n
    psi, signs, physical = critical_data(n)
    hamming = np.sum(signs[:,None,:] != signs[None,:,:],axis=-1)
    def force(u, current):
        q, _ = g.regularized_stage(matrix(current), .02*.005/np.sqrt(2**n), .1*.005)
        values = np.linalg.eigvalsh(q)
        rhs = -2*u*hamming*(q@q)
        def audit(f):
            x = matrix(f)
            return np.linalg.norm(q@x+x@q-rhs)/(2*values[0])
        try:
            return g.sylvester_sign(root_mpo(q,n),root_mpo(rhs,n),g.MPOAlgebra(16),
                                    values[0],values[-1],audit,1e-4,max_iterations=24)
        except g.Rejected as failure:
            if failure.reason not in ('iteration_limit','force_tolerance') or failure.candidate is None:
                raise
            return failure.candidate, dict(provisional=True)
    rows = []
    for endpoint in (elapsed, .012):
        trial = state
        if endpoint != elapsed:
            u = np.sqrt(elapsed)
            trial, _ = tn.midpoint_step(state, u, np.sqrt(endpoint)-u, force, state.bond_dimension(),
                                        retract=polar_retractor(n,state.bond_dimension()), mode='two')
        density = physical(endpoint)
        best = trial
        initial_bound = g.endpoint_dense(matrix(trial),density)['cp_bound']
        bound = initial_bound
        for b,w in [(64,32),(64,64),(64,128),(128,128),(128,256)]:
            best, bound, steps = dense_correct(best, density, b, w, passes=5, tolerance=tolerance)
            rows.append(dict(time=endpoint, prediction_bound=initial_bound, state_bond=b,
                             update_bond=w, passes=len(steps), cp_bound=bound,
                             measured_error=np.linalg.norm(matrix(best)-channel_root(psi[:,None],signs,endpoint)),
                             meets_tolerance=bound<=tolerance))
            assert rows[-1]['measured_error'] <= bound+2e-10
            if bound <= tolerance:
                break
        write_rows('critical_resume',rows)
        print(rows[-1],flush=True)
        if bound > tolerance:
            return False
        state = best
    return True


def self_check():
    rng = np.random.default_rng(8734)
    a = rng.normal(size=(16,16)) + 1j*rng.normal(size=(16,16))
    state = root_mpo(a,4)
    herm = hermitian_round(state,16)
    assert np.linalg.norm(matrix(herm)-(a+a.conj().T)/2) < 1e-12
    small = matrix(hermitian_round(state,3))
    assert np.linalg.norm(small-small.conj().T) < 1e-12
    initial, physical, audit, _ = correlated_data(4)
    density = physical(.1)
    frame, before = audit(initial,density)
    trial = mpo_trial(frame,density,16,16,12,True)
    _, after = audit(trial,density)
    values,basis = np.linalg.eigh(matrix(density))
    reference = (basis*np.sqrt(values))@basis.conj().T
    assert np.linalg.norm(matrix(trial)-reference) <= after + 1e-12
    assert after < before
    print('Passed complex Hermitian rounding, real-tensor truncation, and independent CP audit.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('test', choices=['late', 'early', 'mpo', 'trajectory', 'mpo-trajectory','resume','check'])
    parser.add_argument('--n', type=int, default=8)
    parser.add_argument('--bond', type=int, default=64)
    parser.add_argument('--work-bond', type=int, default=12)
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--intervals', type=int, default=8)
    parser.add_argument('--time', type=float, default=.1)
    parser.add_argument('--tolerance', type=float, default=1e-6)
    parser.add_argument('--passes', type=int, default=3)
    parser.add_argument('--control', action='store_true')
    parser.add_argument('--no-predictor', action='store_true')
    parser.add_argument('--force-bond', type=int, default=0)
    parser.add_argument('--unpreconditioned', action='store_true')
    parser.add_argument('--out', type=Path, help='Output directory for CSV files and checkpoints')
    args = parser.parse_args()
    if args.out is not None:
        OUT = args.out
    if args.test == 'check':
        self_check()
    elif args.test == 'resume':
        raise SystemExit(0 if resume_critical() else 2)
    elif args.test == 'mpo':
        mpo_refine(args.n, args.bond, args.work_bond, args.iterations, args.passes)
    elif args.test == 'mpo-trajectory':
        if args.bond not in (6,8,12) or args.tolerance <= 0 or args.intervals < 1 or args.time <= 0:
            parser.error('Use a bond cap of 6, 8 or 12, with positive tolerance, time and intervals')
        status = mpo_trajectory(args.n, args.time, args.intervals, args.tolerance, args.bond,
                                args.passes, not args.unpreconditioned)
        raise SystemExit(0 if status == 'accepted' else 2)
    elif args.test == 'trajectory':
        trajectory(args.n, args.bond, args.intervals, not args.control, not args.no_predictor, args.force_bond)
    elif args.test == 'late':
        dense_sweep('critical_n8_chi64_s40_tol0.002', 'critical_late')
    else:
        dense_sweep('critical_n8_chi64_s2_tol0.005_trialbond16_T0.012', 'critical_early', 'guarded_final2')
