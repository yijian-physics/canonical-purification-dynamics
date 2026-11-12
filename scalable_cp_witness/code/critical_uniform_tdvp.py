"""Small-system stress test: critical Ising state, dephasing at every site.

This uses a generic *dense current-state* Sylvester force and full MPS tensor
tangent projection. Dense polar retraction is a diagnostic, not a scalable
algorithm. The exact trajectory is only used for the final error measurement.
"""
import argparse
from pathlib import Path
import time
import tdvp_core as tn
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh
import half_petz_evolution as hp
from long_markov_defect import save


def critical_ground(n):
    labels=np.arange(2**n)
    signs=1-2*((labels[:,None] >> np.arange(n-1,-1,-1)) & 1)
    diagonal=-np.sum(signs[:,:-1]*signs[:,1:],axis=1)
    rows=[labels]; columns=[labels]; data=[diagonal]
    for j in range(n):
        rows.append(labels)
        columns.append(labels^(1 << j))
        data.append(-np.ones(len(labels)))
    h=coo_matrix((np.concatenate(data),(np.concatenate(rows),np.concatenate(columns))),
                 shape=(len(labels),len(labels))).tocsr()
    values,vectors=eigsh(h,k=1,which='SA',tol=1e-13,
                         v0=np.ones(len(labels))/np.sqrt(len(labels)))
    return vectors[:,0],float(values[0]),signs


def channel_root(factor,signs,elapsed):
    """Exact commuting channel layer, evaluated at root rather than density scale."""
    p=-np.expm1(-elapsed)/2
    for j in range(signs.shape[1]):
        factor=np.column_stack([np.sqrt(1-p)*factor,np.sqrt(p)*signs[:,j,None]*factor])
        u,s,_=np.linalg.svd(factor,full_matrices=False)
        factor=u*s
    u,s,_=np.linalg.svd(factor,full_matrices=False)
    return (u*s)@u.conj().T


class CurrentForce:
    def __init__(self,n,signs,clock):
        self.n,self.clock=n,clock
        self.hamming=np.sum(signs[:,None,:]!=signs[None,:,:],axis=-1)

    def __call__(self,coordinate,state):
        amplitude=tn.to_matrix(state.dense(),self.n)
        u,values,_=np.linalg.svd(amplitude,full_matrices=False)
        root=(u*values)@u.conj().T
        rhs=-self.hamming*(amplitude@amplitude.conj().T)
        transformed=u.conj().T@rhs@u
        denominator=values[:,None]+values[None,:]
        keep=denominator>1e-13*values[0]
        null_forcing=np.linalg.norm(transformed[~keep])
        if null_forcing>1e-8:
            raise RuntimeError(f'Unresolved rank opening, kernel forcing {null_forcing:.3g}')
        solution=np.zeros_like(transformed)
        solution[keep]=transformed[keep]/denominator[keep]
        force=u@solution@u.conj().T
        residual=np.linalg.norm(root@force+force@root-rhs)/max(np.linalg.norm(rhs),1e-300)
        if self.clock=='sqrt':
            force*=2*coordinate
        return tn.from_dense(tn.to_interleaved(force,self.n),self.n),dict(
            sylvester_residual=float(residual),null_forcing=float(null_forcing),
            min_root=float(values[-1]),branch_defect=float(np.linalg.norm(amplitude-root)))


def polar_retractor(n,chi):
    def retract(state):
        result,info=tn.compress(state,chi,normalize=True)
        amplitude=tn.to_matrix(result.dense(),n)
        u,s,_=np.linalg.svd(amplitude,full_matrices=False)
        root=(u*s)@u.conj().T
        info['polar_correction']=float(np.linalg.norm(amplitude-root))
        result,second=tn.compress(tn.from_dense(tn.to_interleaved(root,n),n),chi,normalize=True)
        info['discarded_bound']+=second['discarded_bound']
        return result,info
    return retract


def run_tdvp(n,chi,steps,clock,total,seed,out):
    started=time.perf_counter()
    psi,energy,signs=critical_ground(n)
    # Automatic initial rank flag: P_kernel L(rho) P_kernel = 1/2 sum |v_j><v_j|.
    vectors=signs*psi[:,None]
    vectors-=psi[:,None]*(psi.conj()@vectors)[None,:]
    opening_singular=np.linalg.svd(vectors/np.sqrt(2),compute_uv=False)
    opening_rank=int(np.sum(opening_singular>1e-12))
    initial_root=channel_root(psi[:,None],signs,seed)
    state,compressed=tn.compress(tn.from_dense(tn.to_interleaved(initial_root,n),n),chi,normalize=True)
    startup_error=np.linalg.norm(tn.to_matrix(state.dense(),n)-initial_root)
    provider=CurrentForce(n,signs,clock)
    retract=polar_retractor(n,chi)
    start=np.sqrt(seed) if clock=='sqrt' else seed
    stop=np.sqrt(total) if clock=='sqrt' else total
    step=(stop-start)/steps
    max_sylvester=max_branch=max_polar=max_discard=0.
    min_root=1.
    for index in range(steps):
        state,info=tn.midpoint_step(state,start+index*step,step,provider,chi,
                                    retract=retract,mode='two')
        for key in ('force1','force2'):
            max_sylvester=max(max_sylvester,info[key]['sylvester_residual'])
            max_branch=max(max_branch,info[key]['branch_defect'])
            min_root=min(min_root,info[key]['min_root'])
        for key in ('mid_retraction','end_retraction'):
            max_polar=max(max_polar,info[key]['polar_correction'])
            max_discard=max(max_discard,info[key]['discarded_bound'])
        if steps>=40 and (index+1)%20==0:
            print(f'progress n={n} chi={chi} {clock} {index+1}/{steps}',flush=True)
    target=channel_root(psi[:,None],signs,total)
    amplitude=tn.to_matrix(state.dense(),n)
    proxy,_=tn.compress(tn.from_dense(tn.to_interleaved(target,n),n),chi,normalize=True)
    row=dict(n=n,chi=chi,steps=steps,clock=clock,time=total,seed_time=seed,
             opening_rank=opening_rank,cp_error=float(np.linalg.norm(amplitude-target)),
             startup_compression_error=float(startup_error),
             target_compression_error=float(np.linalg.norm(tn.to_matrix(proxy.dense(),n)-target)),
             max_sylvester_residual=max_sylvester,max_branch_defect=max_branch,
             max_polar_correction=max_polar,max_truncation_bound=max_discard,
             min_root=min_root,ground_energy=energy,seconds=time.perf_counter()-started)
    seed_tag='' if seed==.01 else f'_seed{seed:g}'
    save(out/f'uniform_n{n}_chi{chi}_{clock}_s{steps}{seed_tag}.csv',[row])
    print(row,flush=True)
    return row


def run_buffer(n,chi,sweeps,buffer,total,out):
    started=time.perf_counter()
    psi,_,signs=critical_ground(n)
    initial=tn.from_dense(tn.to_interleaved(np.outer(psi,psi.conj()),n),n)
    driver=hp.HalfPetzEvolution(initial,buffer=buffer,chi=chi)
    p=-np.expm1(-total/sweeps)/2
    operators=[np.sqrt(1-p)*np.eye(2),np.sqrt(p)*np.diag([1.,-1.])]
    max_discard=0.
    for sweep in range(sweeps):
        for j in range(n):
            diagnostics=driver.update(j,operators)
            max_discard=max(max_discard,diagnostics.truncation_bound)
    amplitude=tn.to_matrix(driver.state.dense(),n)
    target=channel_root(psi[:,None],signs,total)
    length=np.linalg.norm(amplitude)
    row=dict(n=n,chi=chi,sweeps=sweeps,buffer=buffer,time=total,
             cp_error=float(np.linalg.norm(amplitude-target)),
             normalized_cp_error=float(np.linalg.norm(amplitude/length-target)),
             norm=float(length),max_truncation_bound=max_discard,
             seconds=time.perf_counter()-started)
    save(out/f'uniform_half_n{n}_chi{chi}_b{buffer}_s{sweeps}.csv',[row])
    print(row,flush=True)
    return row


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n',type=int,default=6)
    parser.add_argument('--chi',type=int,nargs='+',default=[16,64])
    parser.add_argument('--steps',type=int,nargs='+',default=[20,40,80])
    parser.add_argument('--clocks',nargs='+',default=['sqrt'])
    parser.add_argument('--time',type=float,default=.3)
    parser.add_argument('--seed',type=float,default=.01)
    parser.add_argument('--buffers',type=int,nargs='*',default=[])
    parser.add_argument('--out',type=Path,default=Path(__file__).parent/'long_markov_results')
    args=parser.parse_args()
    args.out.mkdir(exist_ok=True)
    for chi in args.chi:
        for steps in args.steps:
            if not args.buffers:
                for clock in args.clocks:
                    run_tdvp(args.n,chi,steps,clock,args.time,args.seed,args.out)
            for buffer in args.buffers:
                run_buffer(args.n,chi,steps,buffer,args.time,args.out)


if __name__=='__main__':
    main()
