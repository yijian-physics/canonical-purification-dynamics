"""Two-site finite DMRG for the open critical spin chain H=-sum ZZ-sum X."""
from pathlib import Path
import argparse
import csv
import time
import tdvp_core as tn
import numpy as np
from scipy.sparse.linalg import LinearOperator,eigsh

I=np.eye(2)
X=np.array([[0.,1.],[1.,0.]])
Z=np.diag([1.,-1.])


def mpo(n):
    tensors=[]
    for j in range(n):
        w=np.zeros((3,2,2,3))
        w[0,:,:,0]=I
        w[2,:,:,2]=I
        w[0,:,:,1]=-Z
        w[1,:,:,2]=Z
        w[0,:,:,2]=-X
        if j==0:w=w[:1]
        if j==n-1:w=w[:,:,:,-1:]
        tensors.append(w)
    return tensors


def advance_left(env,a,w):
    return np.einsum('awb,air,wijv,bjs->rvs',env,a.conj(),w,a,optimize=True)


def advance_right(env,a,w):
    return np.einsum('air,wijv,bjs,rvs->awb',a.conj(),w,a,env,optimize=True)


def energy(tensors,operators):
    env=np.ones((1,1,1))
    for a,w in zip(tensors,operators):
        env=advance_left(env,a,w)
    return float(env.item().real)


def exact_energy(n):
    k=np.arange(1,n+1)
    return float(-np.sum(2*np.sin((2*k-1)*np.pi/(4*n+2))))


def solve_pair(theta,left,right,w1,w2):
    pair=np.einsum('wijv,vklz->wikjlz',w1,w2,optimize=True)
    shape=theta.shape
    def apply(vector):
        tensor=vector.reshape(shape)
        return np.einsum('awb,wikjlz,czd,bjld->aikc',left,pair,right,tensor,optimize=True).reshape(-1)
    operator=LinearOperator((theta.size,theta.size),matvec=apply,dtype=np.float64)
    if theta.size<=4:
        dense=np.column_stack([apply(v) for v in np.eye(theta.size)])
        values,vectors=np.linalg.eigh(dense)
        return values[0],vectors[:,0].reshape(shape)
    values,vectors=eigsh(operator,k=1,which='SA',v0=theta.reshape(-1),
                        tol=1e-10,maxiter=300,ncv=min(24,theta.size))
    return values[0],vectors[:,0].reshape(shape)


def run(n,chi,sweeps=6,initial=None):
    started=time.perf_counter()
    operators=mpo(n)
    tensors=([a.copy() for a in initial] if initial is not None else
             [np.ones((1,2,1))/np.sqrt(2) for _ in range(n)])
    records=[]
    for sweep in range(sweeps):
        right=[None]*(n+1)
        right[n]=np.ones((1,1,1))
        for j in range(n-1,-1,-1):
            right[j]=advance_right(right[j+1],tensors[j],operators[j])
        left=[np.ones((1,1,1))]
        discarded=0.
        for j in range(n-1):
            theta=np.tensordot(tensors[j],tensors[j+1],axes=(-1,0))
            _,theta=solve_pair(theta,left[j],right[j+2],operators[j],operators[j+1])
            u,s,vh=np.linalg.svd(theta.reshape(theta.shape[0]*2,-1),full_matrices=False)
            rank=min(chi,len(s))
            discarded=max(discarded,float(np.sum(s[rank:]**2)))
            tensors[j]=u[:,:rank].reshape(theta.shape[0],2,rank)
            tensors[j+1]=(s[:rank,None]*vh[:rank]).reshape(rank,2,theta.shape[-1])
            left.append(advance_left(left[j],tensors[j],operators[j]))
        env=np.ones((1,1,1))
        for j in range(n-2,-1,-1):
            theta=np.tensordot(tensors[j],tensors[j+1],axes=(-1,0))
            _,theta=solve_pair(theta,left[j],env,operators[j],operators[j+1])
            u,s,vh=np.linalg.svd(theta.reshape(theta.shape[0]*2,-1),full_matrices=False)
            rank=min(chi,len(s))
            discarded=max(discarded,float(np.sum(s[rank:]**2)))
            tensors[j]=(u[:,:rank]*s[:rank]).reshape(theta.shape[0],2,rank)
            tensors[j+1]=vh[:rank].reshape(rank,2,theta.shape[-1])
            env=advance_right(env,tensors[j+1],operators[j+1])
        tensors[0]/=np.linalg.norm(tensors[0])
        measured=energy(tensors,operators)
        row=dict(n=n,chi=chi,sweep=sweep+1,energy=measured,
                 exact_energy=exact_energy(n),energy_error=measured-exact_energy(n),
                 max_discarded=discarded,seconds=time.perf_counter()-started)
        records.append(row)
        print(row,flush=True)
        if sweep>=2 and abs(records[-1]['energy']-records[-2]['energy'])<1e-10:
            break
    return tensors,records


def cp_projector(ket):
    state=tn.MPS(len(ket),max_bond=max(a.shape[-1] for a in ket)**2)
    state.tensors=[np.einsum('asb,ctd->acstbd',a,a.conj(),optimize=True).reshape(
        a.shape[0]**2,4,a.shape[-1]**2) for a in ket]
    state.center=0
    return state


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n',type=int,nargs='+',default=[8,16,32,64])
    parser.add_argument('--chi',type=int,default=8)
    parser.add_argument('--sweeps',type=int,default=6)
    parser.add_argument('--warm-chi',type=int)
    parser.add_argument('--out',type=Path,default=Path(__file__).parent/'long_markov_results')
    args=parser.parse_args()
    args.out.mkdir(exist_ok=True)
    for n in args.n:
        initial=None
        if args.warm_chi:
            saved=np.load(args.out/f'ising_n{n}_ket{args.warm_chi}.npz')
            initial=[saved[f'a{j}'] for j in range(n)]
        ket,records=run(n,args.chi,args.sweeps,initial)
        np.savez_compressed(args.out/f'ising_n{n}_ket{args.chi}.npz',
                            **{f'a{j}':a for j,a in enumerate(ket)})
        with (args.out/f'ising_n{n}_ket{args.chi}.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=records[0])
            writer.writeheader()
            writer.writerows(records)


if __name__=='__main__':
    main()
