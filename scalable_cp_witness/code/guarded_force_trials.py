import argparse
from pathlib import Path
from time import perf_counter
import guarded_cp_tdvp as g
import tdvp_core as tn
import numpy as np
from critical_uniform_tdvp import critical_ground, channel_root
from long_markov_defect import save


def table(out, name, rows):
    out.mkdir(parents=True, exist_ok=True)
    save(out/name, rows)


def root_mpo(matrix, n, bond=None):
    return tn.from_dense(tn.to_interleaved(matrix, n), n, bond)


def force_trials(out, bonds=(0,8,16), times=(.01,.1,.3), filename='critical_forces.csv'):
    rows = []
    n = 8
    psi, _, signs = critical_ground(n)
    hamming = np.sum(signs[:,None,:] != signs[None,:,:],axis=-1)
    for elapsed in times:
        q = channel_root(psi[:,None], signs, elapsed)
        rhs = -hamming*(q@q)
        oracle = g.SpectralAudit(q, rhs)
        exact_norm = np.linalg.norm(oracle.transformed_rhs/oracle.denominator)
        for bond in bonds:
            engine = g.MPOAlgebra(bond) if bond else g.DenseAlgebra()
            root = root_mpo(q,n) if bond else q
            source = root_mpo(rhs,n) if bond else rhs
            best_error = np.inf
            smallest_residual = np.inf
            def audit(force):
                nonlocal best_error, smallest_residual
                candidate = tn.to_matrix(force.dense(),n) if bond else force
                residual = np.linalg.norm(q@candidate+candidate@q-rhs)
                smallest_residual = min(smallest_residual, residual/np.linalg.norm(rhs))
                best_error = min(best_error, oracle.error(candidate)/exact_norm)
                return residual/(2*oracle.values[0])
            row = dict(n=n, time=elapsed, bond=bond, method='zip_MPO' if bond else 'dense_products')
            started = perf_counter()
            try:
                _, info = g.sylvester_sign(root, source, engine, oracle.values[0],
                    oracle.values[-1], audit, tolerance=1e-4*exact_norm)
                row.update(status='accepted', **info)
            except g.Rejected as failure:
                row.update(status=failure.reason, **failure.measurements)
            row.update(best_relative_force_error=best_error, best_relative_residual=smallest_residual,
                       seconds=perf_counter()-started)
            rows.append(row)
            table(out, filename, rows)
            print(row, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path(__file__).parent/'guarded_results')
    parser.add_argument('--bonds',type=int,nargs='+',default=[0,8,16])
    parser.add_argument('--times',type=float,nargs='+',default=[.01,.1,.3])
    parser.add_argument('--filename',default='critical_forces.csv')
    args=parser.parse_args()
    force_trials(args.out,args.bonds,args.times,args.filename)
