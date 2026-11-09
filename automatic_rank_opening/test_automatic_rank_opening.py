import automatic_rank_opening as auto
import mpo_cp_force as mpo
import large_mpo_tdvp as tables
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm_multiply
from pathlib import Path
import argparse
import time


def kron(items):
    out=np.ones((1,1),complex)
    for a in items:out=np.kron(out,a)
    return out


def embed(a,j,n):return kron([a if j==k else mpo.I for k in range(n)])


def model(n,kind='loss',hamiltonian=True):
    h=np.zeros((2**n,2**n),complex)
    if hamiltonian:
        h=sum(.43*embed(mpo.X,j,n)+.19*embed(mpo.Z,j,n) for j in range(n))
        h+=sum(.7*embed(mpo.Z,j,n)@embed(mpo.Z,j+1,n) for j in range(n-1))
    op={'loss':mpo.SM,'phase':mpo.Z,'bit':mpo.X}[kind]
    jumps=[np.sqrt(.3+.12*np.cos(j)**2)*embed(op,j,n) for j in range(n)]
    return auto.Model(h,jumps)


def exact_root(root,m,elapsed):
    d=len(root);ident=np.eye(d);l=-1j*(np.kron(ident,m.h)-np.kron(m.h.T,ident))
    for jump in m.jumps:
        s=jump.conj().T@jump;l+=np.kron(jump.conj(),jump)-.5*(np.kron(ident,s)+np.kron(s.T,ident))
    rho=expm_multiply(elapsed*csr_matrix(l),(root@root).reshape(-1,order='F')).reshape((d,d),order='F')
    val,vec=np.linalg.eigh(auto.herm(rho));return (vec*np.sqrt(np.maximum(val,0)))@vec.conj().T


def main():
    p=argparse.ArgumentParser();p.add_argument('--outdir',type=Path,default=Path(__file__).parent/'automatic_rank_results');args=p.parse_args();out=args.outdir;out.mkdir(parents=True,exist_ok=True)
    p0=np.diag([1.,0.]).astype(complex);plus=(mpo.I+mpo.X)/2
    ghz=np.zeros(16,complex);ghz[0]=ghz[-1]=1/np.sqrt(2);ghz=np.outer(ghz,ghz.conj())
    mixed=np.zeros((16,16),complex);mixed[0,0]=mixed[-1,-1]=1/np.sqrt(2)
    near=np.diag(np.sqrt([1-1e-8,1e-8])).astype(complex)
    cases=[('pure_plus_loss',kron([plus]*4),model(4),'first_order_opening'),
           ('GHZ_phase_full_local_rank',ghz,model(4,'phase',False),'first_order_opening'),
           ('mixed_rank2_bit',mixed,model(4,'bit',False),'first_order_opening'),
           ('mixed_rank2_phase',mixed,model(4,'phase',False),'smooth_within_budget'),
           ('dark_loss',p0,auto.Model(np.zeros((2,2)),[mpo.SM]),'smooth_within_budget'),
           ('unitary_support_motion',p0,auto.Model(mpo.X,[]),'smooth_within_budget'),
           ('Hamiltonian_activated_loss',p0,auto.Model(mpo.X,[mpo.SM]),'delayed_opening'),
           ('full_rank',mpo.I/np.sqrt(2),auto.Model(mpo.X,[mpo.SM]),'smooth'),
           ('near_boundary_full_rank',near,auto.Model(np.zeros((2,2)),[mpo.X]),'near_boundary')]
    rows=[]
    for name,root,m,expected in cases:
        info=auto.classify(root,m,.02,1e-6)
        assert info['mode']==expected,(name,info,expected)
        row=dict(case=name,**info);rows.append(row);print(row,flush=True)
    tables.save(out/'detection.csv',rows)
    rows=[]
    for name,root,m,_ in cases:
        if name not in ('pure_plus_loss','Hamiltonian_activated_loss','near_boundary_full_rank'):continue
        for tol in (1e-3,1e-4):
            start=time.perf_counter();result,history=auto.evolve(root,m,.03,tol,max_step=.01)
            reference=exact_root(root,m,.03);error=np.linalg.norm(result-reference)
            row=dict(case=name,tolerance=tol,cp_error=error,steps=len(history),
                     bridge_steps=sum(r['method']=='positive_bridge' for r in history),
                     smooth_steps=sum(r['method']=='sqrt_time_midpoint' for r in history),
                     estimated_sum=history[-1]['estimated_sum'],min_root_eigenvalue=min(np.linalg.eigvalsh(result)),
                     seconds=time.perf_counter()-start)
            rows.append(row);tables.save(out/'evolution.csv',rows);tables.save(out/f'{name}_{tol:g}_history.csv',history);print(row,flush=True)
    rows=[]
    for name,root,m,_ in cases:
        if name not in ('pure_plus_loss','Hamiltonian_activated_loss'):continue
        for h in (.01,.0025,.000625):
            info=auto.detect(root,m,h);tail,_=auto.finite_probe(root,m,h,info['rank'])
            rows.append(dict(case=name,step=h,finite_probe_tail=tail,leakage_rate=info['leakage_rate']))
    tables.save(out/'opening_scaling.csv',rows)


if __name__=='__main__':main()
