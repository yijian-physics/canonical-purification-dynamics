"""Finite-density interacting CP-TDVP with MPO-only stage forces."""
import mpo_cp_force as mpo
import tdvp_core as tn
import argparse
import copy
import csv
import time
from pathlib import Path
import numpy as np
from scipy.linalg import expm


def save(path,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)


def retraction(chi,gauge=False):
    def retract(state):
        out,info=tn.compress(state,chi,tolerance=1e-10,normalize=True)
        before=out
        out=mpo.hermitize(out,chi,normalize=True)
        info['hermitian_correction']=tn.distance(before,out)
        if gauge:
            out,g=mpo.local_trace_gauge(out,chi);info.update(g)
        return out,info
    return retract


def cp_physical_marginal(state,start,width):
    state=copy.deepcopy(state);state.move_center(start)
    block=state.tensors[start]
    for j in range(start+1,start+width):block=np.tensordot(block,state.tensors[j],axes=(-1,0))
    dl,dr=block.shape[0],block.shape[-1]
    order=[0]+[1+2*j for j in range(width)]+[2+2*j for j in range(width)]+[2*width+1]
    a=block.reshape([dl]+[2,2]*width+[dr]).transpose(order).reshape(dl,2**width,2**width,dr)
    return np.einsum('apkr,aqkr->pq',a,a.conj(),optimize=True)


def doubled_marginal(state,start,width):
    state=copy.deepcopy(state);state.move_center(start);block=state.tensors[start]
    for j in range(start+1,start+width):block=np.tensordot(block,state.tensors[j],axes=(-1,0))
    a=block.reshape(block.shape[0],4**width,block.shape[-1]).transpose(1,0,2).reshape(4**width,-1)
    return a@a.conj().T


def density_marginal(rho,start,width):
    left=np.ones(1,complex);right=np.ones(1,complex)
    for a in rho.tensors[:start]:left=left@(a[:,0,:]+a[:,3,:])
    for a in rho.tensors[start+width:][::-1]:right=(a[:,0,:]+a[:,3,:])@right
    block=rho.tensors[start]
    for a in rho.tensors[start+1:start+width]:block=np.tensordot(block,a,axes=(-1,0))
    vector=np.einsum('a,apb,b->p',left,block.reshape(block.shape[0],4**width,block.shape[-1]),right,optimize=True)
    return tn.to_matrix(vector,width)


def density_reference(n,elapsed,dt,polarization=.2,chi=32):
    rho=tn.MPS(n,max_bond=chi);local=(mpo.I+polarization*mpo.X)/2
    rho.tensors=[local.reshape(1,4,1).copy() for _ in range(n)];rho.center=n-1;rho.move_center(0)
    steps=round(elapsed/dt);dt=elapsed/steps;rates=.3+.12*np.cos(np.arange(n))**2
    h=.43*mpo.X+.19*mpo.Z;number=mpo.SM.conj().T@mpo.SM
    onsite=[]
    for rate in rates:
        liouv=-1j*(np.kron(h,mpo.I)-np.kron(mpo.I,h.T))+rate*(np.kron(mpo.SM,mpo.SM.conj())-.5*np.kron(number,mpo.I)-.5*np.kron(mpo.I,number.T))
        onsite.append(expm(dt/2*liouv))
    unitary=expm(-1j*.7*dt*np.kron(mpo.Z,mpo.Z));gate=np.kron(unitary,unitary.conj())
    order=np.arange(16).reshape(2,2,2,2).transpose(0,2,1,3).reshape(-1);gate=gate[np.ix_(order,order)]
    for _ in range(steps):
        for j in range(n):rho.apply_gate(j,1,onsite[j])
        for parity in (0,1):
            for j in range(parity,n-1,2):rho.apply_gate(j,2,gate)
        for j in range(n):rho.apply_gate(j,1,onsite[j])
        rho.tensors[rho.center]/=mpo.trace(rho)
    return rho


def trace_distance(a,b):return float(np.sum(abs(np.linalg.eigvalsh((a-b+a.conj().T-b.conj().T)/2)))/2)


def run(n=8,chi=8,force_chi=16,dt=.025,elapsed=.1,tolerance=1e-6,maxiter=80,polarization=.2):
    started=time.perf_counter();state=mpo.product_root(n,polarization);rates=.3+.12*np.cos(np.arange(n))**2
    provider=mpo.MPOForce(n,rates,force_chi,tolerance,maxiter)
    steps=round(elapsed/dt);dt=elapsed/steps;max_hermitian=0.;gauge_count=0
    for step in range(steps):
        state,info=tn.midpoint_step(state,step*dt,dt,provider,chi,retraction(chi),mode='two')
        state,gauge=mpo.local_trace_gauge(state,chi);gauge_count+=gauge['gauge_gates']
        max_hermitian=max(max_hermitian,info['mid_retraction']['hermitian_correction'],info['end_retraction']['hermitian_correction'])
        print('step',n,chi,step+1,'force residual',provider.records[-1]['true_residual'],'iterations',provider.records[-1]['iterations'],flush=True)
    reference=density_reference(n,elapsed,.0025,polarization,chi=max(32,2*chi))
    probe=n//2-1;actual=cp_physical_marginal(state,probe,2);target=density_marginal(reference,probe,2)
    row=dict(n=n,chi=chi,force_chi=force_chi,dt=dt,time=elapsed,polarization=polarization,
             physical_two_site_error=trace_distance(actual,target),hermiticity=tn.distance(state,mpo.adjoint(state)),
             max_hermitian_retraction=max_hermitian,max_sylvester_residual=max(r['true_residual'] for r in provider.records),
             max_cg_iterations=max(r['iterations'] for r in provider.records),mean_cg_iterations=np.mean([r['iterations'] for r in provider.records]),
             curvature_failures=sum(r['curvature_failure'] for r in provider.records),gauge_gates=gauge_count,seconds=time.perf_counter()-started)
    if n<=8:
        amplitude=tn.to_matrix(state.dense(),n);density=tn.to_matrix(reference.dense(),n)
        values,vectors=np.linalg.eigh((density+density.conj().T)/2);root=(vectors*np.sqrt(np.maximum(values,0)))@vectors.conj().T
        u,s,_=np.linalg.svd(amplitude);own=(u*s)@u.conj().T
        row.update(global_cp_error=np.linalg.norm(amplitude-root),branch_error=np.linalg.norm(amplitude-own),min_root_eigenvalue=np.linalg.eigvalsh((amplitude+amplitude.conj().T)/2)[0])
    return row,state,reference,provider.records


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--n',type=int,default=8);p.add_argument('--chi',type=int,default=8)
    p.add_argument('--force-chi',type=int,default=16);p.add_argument('--dt',type=float,default=.025);p.add_argument('--time',type=float,default=.1)
    p.add_argument('--tol',type=float,default=1e-6);p.add_argument('--maxiter',type=int,default=80);p.add_argument('--polarization',type=float,default=.2)
    p.add_argument('--outdir',type=Path,default=Path(__file__).parent/'large_mpo_results');args=p.parse_args();args.outdir.mkdir(parents=True,exist_ok=True)
    row,state,reference,records=run(args.n,args.chi,args.force_chi,args.dt,args.time,args.tol,args.maxiter,args.polarization)
    tag=f'n{args.n}_c{args.chi}_f{args.force_chi}_dt{args.dt:g}_t{args.time:g}_m{args.polarization:g}'
    save(args.outdir/f'{tag}.csv',[row]);save(args.outdir/f'{tag}_forces.csv',[{k:v for k,v in r.items() if k!='history'} for r in records])
    np.savez_compressed(args.outdir/f'{tag}_state.npz',**{f'a{j}':a for j,a in enumerate(state.tensors)})
    np.savez_compressed(args.outdir/f'{tag}_density.npz',**{f'a{j}':a for j,a in enumerate(reference.tensors)})
    print(row,flush=True)


if __name__=='__main__':main()
