"""Half-Petz versus the exact-target optimum over local doubled-site unitaries.

The unitary oracle uses the exact target CP. It measures attainable error.
The half-Petz update receives only the initial MPS and physical channel.
"""
from pathlib import Path
import argparse
import copy
import csv
import time
import half_petz_evolution as hp
import tdvp_core as tn
import cluster_dephasing as base
import numpy as np


def save(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def prepare(n, phase=.37):
    state = base.MPS(n,max_bond=4)
    for site in range(n-1):
        diagonal = [np.exp(1j*phase*((a//2)*(b//2)-(a%2)*(b%2)))
                    for a in range(4) for b in range(4)]
        state.apply_gate(site,2,np.diag(diagonal))
    return state


def exact_target(initial, site, probability, pauli):
    mean = initial.product_expectation({site:np.kron(pauli,base.I)}).real
    weights = np.sqrt([1-probability,probability])
    gram = np.diag(weights**2).astype(complex)
    gram[0,1] = gram[1,0] = weights[0]*weights[1]*mean
    values,basis = np.linalg.eigh(gram)
    inverse_root = (basis/np.sqrt(values))@basis.conj().T
    operators = [base.I,pauli]
    # sqrt(ZZ†) = Z (Z†Z)^(-1/2) Z†, with two Kraus-factor columns.
    gate = sum(weights[a]*weights[b]*inverse_root[a,b]*np.kron(operators[a],operators[b].T)
               for a in range(2) for b in range(2))
    target = copy.deepcopy(initial)
    target.apply_gate(site,1,gate)
    assert abs(target.norm()-1)<1e-11
    return target,values


def block(state, start, width):
    tensor = state.tensors[start]
    for next_tensor in state.tensors[start+1:start+width]:
        tensor = np.tensordot(tensor,next_tensor,axes=(-1,0))
    return tensor.reshape(tensor.shape[0],4**width,tensor.shape[-1])


def procrustes(a_rows,b_rows,exterior):
    joint,scale,_ = np.linalg.svd(np.column_stack([a_rows,b_rows]),full_matrices=False)
    joint = joint[:,scale>1e-13*scale[0]]
    reduced_a,reduced_b = joint.conj().T@a_rows,joint.conj().T@b_rows
    cross = reduced_b@exterior.T@reduced_a.conj().T
    u,s,vh = np.linalg.svd(cross,full_matrices=False)
    return joint,u@vh,s


def apply_rotation(initial,start,width,basis,rotation):
    output = copy.deepcopy(initial)
    output.move_center(start)
    tensor = block(output,start,width)
    rows = tensor.transpose(1,0,2).reshape(4**width,-1)
    rotated = rows+basis@((rotation-np.eye(len(rotation)))@(basis.conj().T@rows))
    updated = rotated.reshape(4**width,tensor.shape[0],tensor.shape[2]).transpose(1,0,2)
    rounding = hp.split_block(output,start,width,
                              updated.reshape([tensor.shape[0]]+[4]*width+[tensor.shape[2]]),64)
    return output,rounding


def best_unitary(initial, target, start, width):
    """Solve global Procrustes on a compact joint support, with identity outside."""
    first,second = copy.deepcopy(initial),copy.deepcopy(target)
    first.move_center(start)
    second.move_center(start)
    left = np.ones((1,1),complex)
    for a,b in zip(first.tensors[:start],second.tensors[:start]):
        left = np.einsum('ab,asr,bsu->ru',left,a.conj(),b,optimize=True)
    right = np.ones((1,1),complex)
    for a,b in zip(first.tensors[start+width:][::-1],second.tensors[start+width:][::-1]):
        right = np.einsum('asr,bsu,ru->ab',a.conj(),b,right,optimize=True)
    a,b = block(first,start,width),block(second,start,width)
    a_rows = a.transpose(1,0,2).reshape(4**width,-1)
    b_rows = b.transpose(1,0,2).reshape(4**width,-1)
    exterior = np.einsum('ab,uv->aubv',left,right).reshape(a_rows.shape[1],b_rows.shape[1])
    joint,rotation,s = procrustes(a_rows,b_rows,exterior)
    output,rounding = apply_rotation(initial,start,width,joint,rotation)
    error = tn.distance(output,target)
    # Stable distance above, cancellation-prone overlap identity only as a check.
    assert abs(error**2-(2-2*np.sum(s)))<1e-10
    assert abs(output.norm()-1)<1e-11
    return output,dict(error=error,rounding=rounding,gate_rank=len(rotation))


def local_data_unitary(initial,site,channel,start,width,factor,r_start,r_width):
    before_basis,before_scale,_ = hp.factor_spectrum(factor,1e-13)
    local_kraus = [hp.embed(k,site-r_start,r_width) for k in channel]
    after_factor = np.column_stack([k@factor for k in local_kraus])
    after_basis,after_scale,_ = hp.factor_spectrum(after_factor,1e-13)
    before = (before_basis*before_scale)@before_basis.conj().T
    after = (after_basis*after_scale)@after_basis.conj().T
    offset = start-r_start
    def unfold(matrix):
        vector = base.to_interleaved(matrix,r_width)
        return vector.reshape(4**offset,4**width,-1).transpose(1,0,2).reshape(4**width,-1)
    a,b = unfold(before),unfold(after)
    basis,rotation,_ = procrustes(a,b,np.eye(a.shape[1]))
    local_residual = np.linalg.norm(a+basis@((rotation-np.eye(len(rotation)))@(basis.conj().T@a))-b)
    output,rounding = apply_rotation(initial,start,width,basis,rotation)
    return output,dict(residual=local_residual,rounding=rounding,fit_width=r_width)


def entropy_from_values(values):
    values = np.maximum(values,0)
    values = values[values>1e-15]
    return float(-np.sum(values*np.log(values)))


def information_loss(before, after, global_values):
    return (entropy_from_values(np.linalg.eigvalsh(before))-
            entropy_from_values(np.linalg.eigvalsh(after))+
            entropy_from_values(global_values))


def dense_checks(initial,target,site,p,pauli,start,width,unitary_error,half_output):
    n = initial.n
    q0 = tn.to_matrix(initial.dense(),n)
    rho = q0@q0.conj().T
    global_pauli = hp.embed(pauli,site,n)
    factor = np.column_stack([np.sqrt(1-p)*q0,np.sqrt(p)*global_pauli@q0])
    u,s,_ = np.linalg.svd(factor,full_matrices=False)
    keep = s>1e-12*s[0]
    dense_root = (u[:,keep]*s[keep])@u[:,keep].conj().T
    expected = tn.to_matrix(target.dense(),n)
    reference_error = np.linalg.norm(dense_root-expected)
    assert reference_error<1e-11
    a,b = [base.to_interleaved(q,n).reshape(4**start,4**width,-1).transpose(1,0,2).reshape(4**width,-1)
           for q in (q0,dense_root)]
    # A dense cross matrix is manageable for three-site gates.
    if width == 3:
        u,_,vh = np.linalg.svd(b@a.conj().T,full_matrices=False)
        dense_error = np.linalg.norm((u@vh)@a-b)
        assert abs(dense_error-unitary_error)<1e-10
    actual = tn.to_matrix(half_output.dense(),n)
    normalized = actual/np.linalg.norm(actual)
    vectors,singular,_ = np.linalg.svd(normalized,full_matrices=False)
    own = (vectors*singular)@vectors.conj().T
    physical_difference = normalized@normalized.conj().T-dense_root@dense_root
    # Explicit recovery for the un-tilted channel on the three central spins.
    bits = (np.arange(8)[:,None] >> np.arange(2,-1,-1)) & 1
    dressing = np.diag(np.exp(.37j*(bits[:,0]*bits[:,1]+bits[:,1]*bits[:,2])))
    plus = np.ones(2)/np.sqrt(2)
    sigma = factor@factor.conj().T
    recovered = np.zeros_like(sigma)
    for outcome in range(2):
        reset = np.outer(plus,np.eye(2)[outcome])
        recovery = dressing@hp.embed(reset,1,3)@dressing.conj().T
        global_recovery = np.kron(np.eye(2**(site-1)),
                                 np.kron(recovery,np.eye(2**(n-site-2))))
        recovered += global_recovery@sigma@global_recovery.conj().T
    recovery_error = np.sum(abs(np.linalg.eigvalsh(recovered-rho)))
    return dict(reference_dense_error=reference_error,
                half_dense_cp_error=np.linalg.norm(actual-dense_root),
                half_normalized_branch_error=np.linalg.norm(normalized-own),
                half_physical_trace_distance=np.sum(abs(np.linalg.eigvalsh(physical_difference)))/2,
                explicit_recovery_trace_error=recovery_error)


def run(n,angles,probability,out):
    initial = prepare(n)
    site = n//2
    rows = []
    fit_cache = {}
    for theta in angles:
        pauli = np.cos(theta)*base.Z+np.sin(theta)*base.X
        channel = [np.sqrt(1-probability)*base.I,np.sqrt(probability)*pauli]
        target,global_values = exact_target(initial,site,probability,pauli)
        for buffer in (1,2):
            started = time.perf_counter()
            solver = hp.HalfPetzEvolution(initial,buffer=buffer,chi=64,cutoff=1e-13)
            setup = time.perf_counter()-started
            start,width = solver.site_window[site]
            before = solver.windows[(start,width)].copy()
            started = time.perf_counter()
            diagnostic = solver.update(site,channel)
            half_seconds = time.perf_counter()-started
            half_error = tn.distance(solver.state,target)
            length = solver.state.norm()
            normalized_error = tn.distance(tn.scale(solver.state,1/length),target)
            started = time.perf_counter()
            unitary,info = best_unitary(initial,target,start,width)
            unitary_seconds = time.perf_counter()-started
            r_start = max(0,start-1)
            r_stop = min(n,start+width+1)
            key = (r_start,r_stop-r_start)
            fit_setup = 0.
            if key not in fit_cache:
                started = time.perf_counter()
                factor = hp.physical_factor(copy.deepcopy(initial),*key)
                basis,scale,_ = hp.factor_spectrum(factor,1e-13)
                fit_cache[key] = basis*scale
                fit_setup = time.perf_counter()-started
            started = time.perf_counter()
            local_output,local_info = local_data_unitary(initial,site,channel,start,width,
                                                       fit_cache[key],*key)
            local_seconds = time.perf_counter()-started
            local_error = tn.distance(local_output,target)
            assert local_error+1e-10>=info['error']
            loss = information_loss(before,solver.windows[(start,width)],global_values)
            assert loss>-1e-10
            row = dict(n=n,site=site,p=probability,theta=theta,width=width,
                       half_cp_error=half_error,half_normalized_cp_error=normalized_error,
                       half_norm=length,unitary_cp_error=info['error'],
                       information_loss=max(0.,loss),recovery_trace_bound=4*probability*abs(np.sin(theta/2)),
                       half_setup_seconds=setup,half_update_seconds=half_seconds,
                       oracle_unitary_seconds=unitary_seconds,unitary_support_rank=info['gate_rank'],
                       local_unitary_cp_error=local_error,local_fit_residual=local_info['residual'],
                       local_fit_width=local_info['fit_width'],local_fit_setup_seconds=fit_setup,
                       local_fit_update_seconds=local_seconds,
                       half_rounding=diagnostic.truncation_bound,unitary_rounding=info['rounding'])
            if n<=8:
                row.update(dense_checks(initial,target,site,probability,pauli,start,width,
                                        info['error'],solver.state))
                assert row['explicit_recovery_trace_error']<=row['recovery_trace_bound']+1e-10
            if theta == 0 or buffer == 2:
                assert half_error<1e-9 and info['error']<1e-9, row
            rows.append(row)
            save(out/f'n{n}.csv',rows)
            print(row,flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n',type=int,nargs='+',default=[6,8])
    parser.add_argument('--angles',type=float,nargs='+',default=[0.,.05,.1,.2])
    parser.add_argument('--p',type=float,default=.1)
    parser.add_argument('--out',type=Path,default=Path(__file__).parent/'tilted_half_petz_results')
    args = parser.parse_args()
    if min(args.n)<6 or not 0<args.p<1:
        parser.error('Require n>=6 and 0<p<1.')
    args.out.mkdir(parents=True,exist_ok=True)
    rows = [row for n in args.n for row in run(n,args.angles,args.p,args.out)]
    save(args.out/'summary.csv',rows)


if __name__=='__main__':
    main()
