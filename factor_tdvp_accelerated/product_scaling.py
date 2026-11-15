"""MPO-only force scaling on an analytically solvable product path.

This validates a force, not a time-evolution trajectory. Product correlations
are absent; it cannot establish generic interacting-chain scalability.
"""
import argparse
import json
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import resource
import sys
from pathlib import Path
from time import perf_counter
import numpy as np
from factor_tdvp import tangent_solver as ts, factor_tdvp as fd
from factor_tdvp_accelerated import matrix_free_solver as mf


def product(local, n):
    result=ts.tn.MPS(n,max_bond=1)
    result.tensors=[local.reshape(1,4,1).copy() for _ in range(n)]
    result.center=n-1
    return result


def run(n):
    tick=perf_counter(); p=.12
    # Local eigenvalues (1 +/- p)/2, p-dot = -p for Z/sqrt(2).
    h=fd.positive_power((fd.I+p*fd.PAULI_X)/2,.25)
    vals,vec=np.linalg.eigh(fd.PAULI_X)
    k=(vec*((-.5*p*vals)*.25*((1+p*vals)/2)**(-.75)))@vec.conj().T
    state=product(h,n); ham=ts.tn.scale(product(fd.I,n),0.)
    jumps=[]
    exact=ts.tn.scale(state,0.)
    for site in range(n):
        jump=product(fd.I,n);jump.tensors[site]=(fd.Z/np.sqrt(2)).reshape(1,4,1);jumps.append(jump)
        term=product(h,n);term.tensors[site]=k.reshape(1,4,1)
        exact=ts.tn.compress(ts.tn.add(exact,term),2,normalize=False)[0]
    setup=perf_counter()-tick
    started=perf_counter()
    velocity,info=mf.factor_velocity(state,ham,jumps,work_bond=8,tolerance=1e-8,maxiter=120)
    elapsed=perf_counter()-started
    relative=ts.tn.distance(velocity,exact)/ts.tn.norm(exact)
    return dict(n=n,status='completed',scope='analytic product-path force only',
        setup_seconds=setup,force_seconds=elapsed,relative_velocity_error=relative,
        independent_global_check='Frobenius distance to exact product-rule factor derivative via MPO contractions',
        peak_process_memory_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if sys.platform=='darwin' else 1024),
        diagnostics=info)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--n',type=int,required=True);p.add_argument('--out',required=True)
    a=p.parse_args()
    try: row=run(a.n)
    except Exception as exc: row=dict(n=a.n,status='failed',error=f'{type(exc).__name__}: {exc}',diagnostics=getattr(exc,'diagnostics',{}))
    Path(a.out).write_text(json.dumps(row,indent=2)+'\n');print(json.dumps({k:v for k,v in row.items() if k!='diagnostics'}))
