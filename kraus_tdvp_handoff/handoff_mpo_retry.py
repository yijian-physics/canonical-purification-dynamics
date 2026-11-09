"""Attempt a longer, uninterrupted MPO solve at the six-spin handoff point."""
import handoff_trial as trial
from pathlib import Path
from time import perf_counter
import numpy as np


def main():
    state = trial.mpo.product_root(6, 1.)
    for _ in range(5):
        state = trial.kraus_step(state, .01, 2, 16)
    root = trial.mpo.hermitize(state,16)
    out = Path(__file__).parent/'handoff_results'
    rows = []
    for chi in (24, 32):
        started = perf_counter()
        rho = trial.tn.compress(trial.mpo.multiply(root, root),chi)[0]
        rhs = trial.tn.compress(trial.mpo.onsite_dissipation(rho,.3+.12*np.cos(np.arange(6))**2),chi)[0]
        _, info = trial.mpo.pcg_force(root,rhs,chi=chi,tolerance=1e-6,maxiter=180,restart_every=0)
        row = dict(force_chi=chi,physical_time=.05,restart_every=0,maxiter=180,
                   iterations=info['iterations'],true_residual=info['true_residual'],
                   passed=info['true_residual']<=1e-5,curvature_failure=info['curvature_failure'],
                   seconds=perf_counter()-started)
        rows.append(row)
        trial.observables.save(out/'mpo_force_retry.csv',rows)
        print(row,flush=True)


if __name__=='__main__':
    main()
