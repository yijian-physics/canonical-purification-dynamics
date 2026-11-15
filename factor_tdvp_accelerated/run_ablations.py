"""Small dense oracles for conditioning, gauge penalties and variational metrics.

This is deliberately NOT the production matrix-free solver. All full matrices
are confined to two-spin diagnostics; no claim of scalable preconditioning.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from scipy.sparse.linalg import lsqr
from factor_tdvp import tangent_solver as ts, factor_tdvp as fd, rank_bridge as rb


def real(a):
    return np.r_[a.real.ravel(), a.imag.ravel()]


def run(out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    conditioning = []
    for population in (0.2, 0.02, 0.002, 0.00002):
        h = np.diag(np.array([1-population, population])**.25).astype(complex)
        basis = [np.eye(4, dtype=complex)[i].reshape(2,2)*phase
                 for i in range(4) for phase in (1, 1j)]
        a = np.column_stack([real(ts.jacobian_dense(h,k)) for k in basis])
        singular = np.linalg.svd(a, compute_uv=False)
        s = singular[singular > singular[0]*1e-13]
        # A generic Hermitian forcing probes all horizontal modes.
        g = np.array([[.13, .21+.17j], [.21-.17j, -.13]])
        b = real(g)
        direct = lsqr(a,b,atol=1e-12,btol=1e-12,iter_lim=100)
        scale = 1/np.maximum(np.linalg.norm(a,axis=0),s[-1])
        pc = lsqr(a*scale,b,atol=1e-12,btol=1e-12,iter_lim=100)
        conditioning.append(dict(population=population, condition=float(s[0]/s[-1]),
            predicted=float(((1-population)/population)**.75), iterations=direct[2],
            column_scaled_iterations=pc[2], residual=float(np.linalg.norm(a@direct[0]-b)),
            column_scaled_residual=float(np.linalg.norm(a@(scale*pc[0])-b)),
            scope='dense qubit oracle; column scaling is not a scalable preconditioner'))
    rng = np.random.default_rng(7281)
    initial = np.eye(4) + .15*(rng.normal(size=(4,4))+1j*rng.normal(size=(4,4)))
    state, _ = ts.normalize_factor(ts.matrix_to_mpo(initial,2,chi=2))
    h = ts.matrix(state); x = h@h.conj().T; rho = x@x
    ham = np.kron(fd.Z,fd.Z)+.23*fd.embed(fd.PAULI_X,0,2)
    g = rb.lindblad_rhs(x@x, ham, [np.sqrt(.2)*fd.embed(fd.SM,1,2)])
    basis = [phase*ts.matrix(b) for b in ts.tangent_basis(state) for phase in (1,1j)]
    a = np.column_stack([real(ts.jacobian_dense(h,k)) for k in basis]); target=real(g)
    d = np.column_stack([real(k@h.conj().T+h@k.conj().T) for k in basis])
    c = np.column_stack([real(h.conj().T@k-k.conj().T@h) for k in basis])
    values,vectors=np.linalg.eigh(x); ge=vectors.conj().T@g@vectors
    f=vectors@(ge/(values[:,None]+values[None,:]))@vectors.conj().T
    rows=[]
    for method in ('density','horizontal_penalty','ridge','root_metric'):
        if method=='density': design,b=a,target
        elif method=='horizontal_penalty': design,b=np.vstack([a,c]),np.r_[target,np.zeros(c.shape[0])]
        elif method=='ridge': design,b=np.vstack([a,.01*np.eye(a.shape[1])]),np.r_[target,np.zeros(a.shape[1])]
        else: design,b=d,real(f)
        coeff=np.linalg.lstsq(design,b,rcond=1e-11)[0]
        rows.append(dict(method=method,density_residual=float(np.linalg.norm(a@coeff-target)),
            root_velocity_residual=float(np.linalg.norm(d@coeff-real(f))),
            horizontal_defect=float(np.linalg.norm(c@coeff)),velocity_norm=float(np.linalg.norm(coeff)),
            density_stationarity=float(np.linalg.norm(a.T@(a@coeff-target))),
            scope='restricted bond-two two-spin tangent; modified objectives explicitly labelled'))
    cert=[]
    for eps in (1e-2,1e-4,1e-6):
        perturb=np.diag([1.,-.3,.2,-.1]); candidate=h+eps*perturb
        candidate/=np.trace((candidate@candidate.conj().T)@(candidate@candidate.conj().T)).real**.25
        root=candidate@candidate.conj().T; error=root@root-rho; target=fd.positive_power(rho,.5)
        tick=perf_counter(); metric=np.linalg.inv(rho); ref_cost=perf_counter()-tick
        tick=perf_counter(); weighted=np.sqrt(max(0,np.trace(error.conj().T@error@metric).real)); weighted_cost=perf_counter()-tick
        tick=perf_counter(); eta=np.linalg.norm(root-candidate@candidate.conj().T); witness_cost=perf_counter()-tick
        cert.append(dict(perturbation=eps,global_cp_error=float(np.linalg.norm(root-target)),
            inverse_weighted_bound=float(weighted),witness_defect=float(eta),
            inverse_reference_seconds=ref_cost,weighted_contraction_seconds=weighted_cost,
            redundant_witness_check_seconds=witness_cost,
            interpretation='ordinary floating-point dense reference; excludes witness fitting, not a rigorous roundoff bound'))
    payload=dict(conditioning=conditioning,variational_ablations=rows,certification=cert)
    (out/'ablations.json').write_text(json.dumps(payload,indent=2)+'\n')
    return payload

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default='factor_tdvp_accelerated/results')
    print(json.dumps(run(p.parse_args().out),indent=2))
