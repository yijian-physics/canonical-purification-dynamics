"""Current-state, center-tensor TDVP for one Pauli-dephasing defect.

The trajectory stays in span{|psi>,P|psi>}. The force is recovered from the
current center tensor in an orthonormal four-operator basis.
"""
from pathlib import Path
import argparse
import copy
import csv
import time as time_module
import tdvp_core as tn
import numpy as np
import half_petz_evolution as hp
import critical_ising_dmrg as ground


def save(path,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def ghz_ket(n):
    first=np.zeros((1,2,2))
    first[0,0,0]=first[0,1,1]=1/np.sqrt(2)
    middle=np.zeros((2,2,2))
    middle[0,0,0]=middle[1,1,1]=1
    last=np.zeros((2,2,1))
    last[0,0,0]=last[1,1,0]=1
    return [first]+[middle.copy() for _ in range(n-2)]+[last]


def positive(matrix):
    u,s,_=np.linalg.svd(matrix,full_matrices=False)
    result=(u*s)@u.conj().T
    return result/np.linalg.norm(result)


class DefectTDVP:
    def __init__(self,initial,site,pauli,rate=1.):
        self.initial=copy.deepcopy(initial)
        self.initial.move_center(site)
        self.site,self.pauli,self.rate=site,pauli,rate
        center=self.initial.tensors[site]
        acted=np.einsum('st,atr->asr',np.kron(pauli,np.eye(2)),center,optimize=True)
        mean=float(np.vdot(center,acted).real)
        variance=1-mean**2
        if variance<1e-12:
            raise ValueError('Use a non-dark Pauli defect.')
        self.mean,self.variance=mean,variance
        self.pauli_reduced=np.array([[mean,np.sqrt(variance)],[np.sqrt(variance),-mean]])
        ops=[np.eye(2),(pauli-mean*np.eye(2))/np.sqrt(variance)]
        self.shape=center.shape
        self.gates=[np.kron(a,b.conj()) for a in ops for b in ops]
        self.basis=np.stack([np.einsum('st,atr->asr',gate,center).reshape(-1)
                             for gate in self.gates])
        assert np.linalg.norm(self.basis.conj()@self.basis.T-np.eye(4))<1e-8
        self.rank_flags=0
        self.max_retraction=0.
        self.max_sylvester_residual=0.

    def coefficients(self,tensor):
        return (self.basis.conj()@tensor.reshape(-1)).reshape(2,2)

    def tensor(self,coefficients):
        return (coefficients.reshape(-1)@self.basis).reshape(self.shape)

    def retract(self,tensor):
        coefficients=self.coefficients(tensor)
        canonical=positive(coefficients)
        result=self.tensor(canonical)
        self.max_retraction=max(self.max_retraction,float(np.linalg.norm(result-tensor)))
        return result

    def derivative(self,coordinate,tensor,clock):
        q=self.coefficients(tensor)
        q=(q+q.conj().T)/2
        values,vectors=np.linalg.eigh(q)
        rho=q@q
        rhs=self.rate/2*(self.pauli_reduced@rho@self.pauli_reduced-rho)
        transformed=vectors.conj().T@rhs@vectors
        kernel=values<1e-12
        if coordinate==0 and np.any(kernel):
            leakage=transformed[np.ix_(kernel,kernel)]
            if np.linalg.norm(leakage)>1e-14:
                self.rank_flags+=1
                if clock!='sqrt':
                    raise RuntimeError('A physical-time derivative diverges at this rank event.')
                vals,vecs=np.linalg.eigh((leakage+leakage.conj().T)/2)
                opening=(vecs*np.sqrt(np.maximum(vals,0)))@vecs.conj().T
                force=vectors[:,kernel]@opening@vectors[:,kernel].conj().T
                return self.tensor(force)
        denominator=values[:,None]+values[None,:]
        if np.min(denominator)<1e-14:
            raise RuntimeError('Additional rank opening requires a finite bridge.')
        derivative=vectors@(transformed/denominator)@vectors.conj().T
        residual=np.linalg.norm(q@derivative+derivative@q-rhs)/max(np.linalg.norm(rhs),1e-300)
        self.max_sylvester_residual=max(self.max_sylvester_residual,float(residual))
        if clock=='sqrt':
            derivative*=2*coordinate
        result=self.tensor(derivative)
        return result-tensor*np.vdot(tensor,result).real

    def state(self,tensor):
        result=copy.deepcopy(self.initial)
        result.tensors[self.site]=tensor
        return result

    def exact(self,time):
        q0=np.diag([1.,0.])
        probability=-np.expm1(-self.rate*time)/2
        rho=(1-probability)*q0+probability*self.pauli_reduced@q0@self.pauli_reduced
        values,vectors=np.linalg.eigh(rho)
        return self.tensor((vectors*np.sqrt(np.maximum(values,0)))@vectors.conj().T)

    def evolve(self,time,steps,clock='sqrt'):
        tensor=self.initial.tensors[self.site].copy()
        endpoint=np.sqrt(time) if clock=='sqrt' else time
        step=endpoint/steps
        first=0
        if clock=='physical':
            # Finite channel bridge on the current two-dimensional density.
            q=self.coefficients(tensor)
            p=-np.expm1(-self.rate*step)/2
            rho=(1-p)*q@q+p*self.pauli_reduced@q@q@self.pauli_reduced
            val,vec=np.linalg.eigh(rho)
            tensor=self.tensor((vec*np.sqrt(np.maximum(val,0)))@vec.conj().T)
            self.rank_flags+=1
            first=1
        for index in range(first,steps):
            coordinate=index*step
            k1=self.derivative(coordinate,tensor,clock)
            middle=self.retract(tensor+step*k1/2)
            k2=self.derivative(coordinate+step/2,middle,clock)
            tensor=self.retract(tensor+step*k2)
        return tensor


def ket_factor(ket,start,width):
    state=tn.MPS(len(ket),max_bond=max(a.shape[-1] for a in ket))
    state.tensors=[a.copy() for a in ket]
    state.center=0
    state.move_center(start)
    tensor=state.tensors[start]
    for a in state.tensors[start+1:start+width]:
        tensor=np.tensordot(tensor,a,axes=(-1,0))
    return tensor.reshape(tensor.shape[0],2**width,tensor.shape[-1]).transpose(1,0,2).reshape(2**width,-1)


def entropy(factor):
    singular=np.linalg.svd(factor,compute_uv=False)
    values=singular**2
    values=values[values>1e-15]
    return float(-np.sum(values*np.log(values)))


def paired_order(width):
    return np.arange(4**width).reshape([2]*(2*width)).transpose(
        [v for j in range(width) for v in (j,j+width)]).reshape(-1)


def buffered_errors(ket,provider,time,width):
    n=len(ket)
    start=max(0,min(n-width,provider.site-width//2))
    position=provider.site-start
    raw=ket_factor(ket,start,width)
    basis,scale,_=hp.factor_spectrum(raw,1e-13)
    factor=basis*scale
    probability=-np.expm1(-provider.rate*time)/2
    kraus=[np.sqrt(1-probability)*np.eye(2**width),
           np.sqrt(probability)*hp.embed(provider.pauli,position,width)]
    after=np.column_stack([k@factor for k in kraus])
    new_basis,new_scale,_=hp.factor_spectrum(after,1e-13)
    left_ops=[new_basis@((new_basis.conj().T@k@basis)*scale[None,:]/new_scale[:,None])@basis.conj().T
              for k in kraus]
    order=paired_order(width)
    forward=sum(np.kron(left,k.conj()) for left,k in zip(left_ops,kraus))
    forward=forward[np.ix_(order,order)]
    target_coeff=provider.coefficients(provider.exact(time))
    target_gate=sum(target_coeff.reshape(-1)[j]*provider.gates[j] for j in range(4))
    target_gate=np.kron(np.eye(4**position),
                        np.kron(target_gate,np.eye(4**(width-position-1))))
    pair_factor=np.kron(factor,factor.conj())[order]
    source_density=pair_factor@pair_factor.conj().T
    actual=forward@pair_factor
    desired=target_gate@pair_factor
    half_error=np.linalg.norm(actual-desired)
    actual_norm=np.linalg.norm(actual)
    half_normalized=np.linalg.norm(actual/actual_norm-desired)
    cross=target_gate@source_density
    fidelity=float(np.sum(np.linalg.svd(cross,compute_uv=False)))
    unitary_error=np.sqrt(max(0.,2-2*fidelity))
    b_factor=factor.reshape(2**position,2,2**(width-position-1),-1)
    b_factor=b_factor.transpose(0,2,1,3).reshape(2**(width-1),-1)
    s_factor=ket_factor(ket,provider.site,1)
    global_rho=target_coeff@target_coeff
    global_values=np.linalg.eigvalsh(global_rho)
    global_entropy=-np.sum(global_values[global_values>1e-15]*np.log(global_values[global_values>1e-15]))
    before_cmi=entropy(factor)+entropy(s_factor)-entropy(b_factor)
    after_cmi=entropy(after)+entropy(s_factor)-entropy(b_factor)-global_entropy
    return dict(width=width,half_cp_error=half_error,half_normalized_cp_error=half_normalized,
                half_norm=actual_norm,best_unitary_error=unitary_error,
                initial_cmi=before_cmi,final_cmi=after_cmi,information_loss=before_cmi-after_cmi)


def run_case(family,n,ket_chi,steps,clock,time,out):
    if family=='ghz':
        ket=ghz_ket(n)
    else:
        loaded=np.load(out/f'ising_n{n}_ket{ket_chi}.npz')
        ket=[loaded[f'a{j}'] for j in range(n)]
    initial=ground.cp_projector(ket)
    started=time_module.perf_counter()
    provider=DefectTDVP(initial,n//2,ground.Z)
    setup=time_module.perf_counter()-started
    started=time_module.perf_counter()
    tensor=provider.evolve(time,steps,clock)
    elapsed=time_module.perf_counter()-started
    target=provider.exact(time)
    error=np.linalg.norm(tensor-target)
    state=provider.state(tensor)
    exact=provider.state(target)
    mps_error=tn.distance(state,exact)
    assert abs(error-mps_error)<1e-9
    row=dict(family=family,n=n,ket_chi=ket_chi,steps=steps,clock=clock,time=time,
             cp_error=error,mps_cp_error=mps_error,rank_flags=provider.rank_flags,
             max_sylvester_residual=provider.max_sylvester_residual,
             max_retraction=provider.max_retraction,setup_seconds=setup,
             tdvp_seconds=elapsed,bond=state.bond_dimension())
    if n<=8:
        force=provider.state(provider.derivative(np.sqrt(time),tensor,'sqrt'))
        projected=tn.tangent_project(state,force)
        row['full_tangent_projection_residual']=tn.distance(force,projected)/tn.norm(force)
        assert row['full_tangent_projection_residual']<1e-10
    if steps==80 and clock=='sqrt':
        buffers=[]
        for width in (1,3,5):
            if width<=n:
                buffers.append(dict(family=family,n=n,ket_chi=ket_chi,time=time,
                                    **buffered_errors(ket,provider,time,width)))
        save(out/f'{family}_n{n}_ket{ket_chi}_buffers.csv',buffers)
    tag=f'{family}_n{n}_ket{ket_chi}_{clock}_s{steps}'
    save(out/(tag+'.csv'),[row])
    print(row,flush=True)
    return row


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family',choices=['ghz','ising'],default='ghz')
    parser.add_argument('--n',type=int,nargs='+',default=[8,64,256])
    parser.add_argument('--ket-chi',type=int,default=8)
    parser.add_argument('--steps',type=int,nargs='+',default=[20,40,80])
    parser.add_argument('--clocks',nargs='+',default=['sqrt','physical'])
    parser.add_argument('--time',type=float,default=.3)
    parser.add_argument('--out',type=Path,default=Path(__file__).parent/'long_markov_results')
    args=parser.parse_args()
    args.out.mkdir(exist_ok=True)
    for n in args.n:
        for steps in args.steps:
            for clock in args.clocks:
                run_case(args.family,n,args.ket_chi,steps,clock,args.time,args.out)


if __name__=='__main__':
    main()
