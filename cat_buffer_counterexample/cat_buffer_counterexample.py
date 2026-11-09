"""Test a proper-buffer polar update from an exact canonical cat projector."""
from pathlib import Path
import csv
import tdvp_core as tn
import buffered_kraus_cp as solver
import mpo_cp_force as mpo
import numpy as np


def cat_amplitude(n, plus, minus):
    """Exact MPO of plus P_+ + minus P_- using four product strings."""
    coefficients = np.array([plus + minus, plus - minus, plus - minus, plus + minus]) / 2
    middle = np.zeros((4, 4, 4), complex)
    idx = np.arange(4)
    middle[idx, idx, idx] = 1
    state = tn.MPS(n, max_bond=4)
    state.tensors = [np.diag(coefficients).reshape(1, 4, 4)]
    state.tensors.extend(middle.copy() for _ in range(n - 2))
    state.tensors.append(np.eye(4).reshape(4, 4, 1))
    return tn.compress(state, 4)[0]


def evolve(state, total_p, repetitions, buffer):
    step_p = -np.expm1(np.log1p(-2 * total_p) / repetitions) / 2
    channel = [np.sqrt(1-step_p)*mpo.I, np.sqrt(step_p)*mpo.Z]
    max_rounding = 0.
    first_loss = None
    for _ in range(repetitions):
        state, info = solver.local_step(state, state.n//2, channel, buffer, chi=4)
        if first_loss is None:
            first_loss = info['kraus_loss']
        max_rounding = max(max_rounding, info['rounding_bound'])
    state, records = solver.trace_optimize(state, buffer, 4, sweeps=2)
    return state, first_loss, max_rounding, sum(r['gauge_gates'] for r in records)


def main():
    rows = []
    p = .1
    for n, buffer, steps in [(6,0,1), (6,1,1), (6,2,1), (6,3,1),
                              (16,2,1), (64,2,1), (64,2,32)]:
        plus = cat_amplitude(n, 1, 0)
        minus = cat_amplitude(n, 0, 1)
        q, loss, rounding, gates = evolve(plus, p, steps, buffer)
        other, _, _, _ = evolve(minus, p, steps, buffer)
        target = cat_amplitude(n, np.sqrt(1-p), np.sqrt(p))
        site = n//2
        whole = site-buffer <= 0 and site+buffer >= n-1
        weights = np.array([1-p,p])
        prediction = target if whole else cat_amplitude(n, *(weights/np.linalg.norm(weights)))
        density = mpo.multiply(q, mpo.adjoint(q))
        observed_weight = tn.overlap(minus, density).real
        observed_affinity = tn.overlap(q, other).real
        expected_affinity = 2*np.sqrt(p*(1-p)) if whole else 2*p*(1-p)/(weights@weights)
        formula_error = tn.distance(q, prediction)
        assert formula_error < 1e-11, (n, buffer, steps, formula_error, observed_weight, gates, loss, rounding)
        assert abs(observed_weight - (p if whole else p*p/(weights@weights))) < 1e-11
        assert abs(observed_affinity - expected_affinity) < 1e-11
        assert gates == 0, 'A positive cat amplitude is stationary under trace rotations.'
        rows.append(dict(n=n, buffer=buffer, full_window=whole, steps=steps, p=p,
                         minus_weight=observed_weight, exact_minus_weight=p,
                         cp_error=tn.distance(q, target), affinity=observed_affinity,
                         exact_affinity=2*np.sqrt(p*(1-p)), first_norm_loss=loss,
                         max_spatial_rounding=rounding, gauge_gates=gates))
    channel = [np.sqrt(1-p)*mpo.I, np.sqrt(p)*mpo.Z]
    _, guarded = solver.automatic_local_step(plus, n//2, channel, cp_tolerance=1e-4,
                                             max_buffer=2, chi=4)
    assert not guarded['accepted']
    out = Path(__file__).parent/'cat_buffer_counterexample.csv'
    with out.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)
    print('Loss-sensitive guard accepted:', guarded['accepted'])


if __name__ == '__main__':
    main()
