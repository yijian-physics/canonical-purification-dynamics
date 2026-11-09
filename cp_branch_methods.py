"""Cluster-dephasing experiments: local Kraus polar alignment, block gates,
a local Pauli generator search, and density-constrained two-tensor trace fits.
Run beside cluster_dephasing.py and cp_tensor_search.py, with numpy and scipy.
"""
import argparse
import itertools
import json
from functools import lru_cache
from pathlib import Path
import cp_tensor_search as cs  # sets BLAS thread count before NumPy import
import cluster_dephasing as base
import numpy as np
from scipy.optimize import minimize


def kraus_polar(x,n,j,h,start,width):
    """Maximize Re Tr of label-zero amplitude by rotating (reference block,label)."""
    d,b=2**n,2**width
    p=-np.expm1(-h)/2
    z=1-2*((np.arange(d)>>(n-1-j))&1)
    blocks=[np.sqrt(1-p)*x,np.sqrt(p)*z[:,None]*x]
    active=list(range(start,start+width))
    order=[k for k in range(n) if k not in active]+active
    perm=order+[k+n for k in order]
    blocks=[k.reshape([2]*(2*n)).transpose(perm).reshape(d//b,b,d//b,b) for k in blocks]
    m=np.concatenate([np.einsum('iaib->ab',k) for k in blocks],axis=1)
    u,s,vh=np.linalg.svd(m,full_matrices=False)
    w=vh.conj().T@u.conj().T
    assert np.linalg.norm(w.conj().T@w-np.eye(b))<1e-11
    out=sum(np.einsum('iajb,bc->iajc',k,v) for k,v in zip(blocks,np.vsplit(w,2)))
    out=out.reshape([2]*(2*n)).transpose(np.argsort(perm)).reshape(d,d)
    assert abs(np.trace(out).real-s.sum())<1e-11
    return out


def unfold(v,n,start,width):
    return v.reshape(4**start,4**width,-1).transpose(1,0,2).reshape(4**width,-1)


def best_gate_error(v,t,n,start,width):
    a,b=(unfold(y,n,start,width) for y in [v,t])
    u,_,vh=np.linalg.svd(b@a.conj().T)
    return float(np.linalg.norm((u@vh)@a-b))


@lru_cache(None)
def pauli_table(width):
    """Sparse permutations and phases of all Pauli words on the doubled block."""
    words=[''.join(w) for w in itertools.product('IXYZ',repeat=2*width)]
    i=np.arange(4**width)
    perm=[]; phases=[]
    for word in words:
        flip=int(''.join('1' if s in 'XY' else '0' for s in word),2)
        mask=int(''.join('1' if s in 'ZY' else '0' for s in word),2)
        q=i^flip
        phase=np.array([(-1)**(int(a)&mask).bit_count() for a in q])*1j**word.count('Y')
        perm.append(q); phases.append(phase)
    return words,np.array(perm),np.array(phases)


def find_angle(x,y,n,start,width,word=None):
    """Exact one-angle fit to a Kraus-polar target; no analytic generator input."""
    a=unfold(base.to_interleaved(x,n),n,start,width)
    b=unfold(base.to_interleaved(y,n),n,start,width)
    m=a@b.conj().T
    words,perm,phase=pauli_table(width)
    aa=np.vdot(b,a).real
    if word is None:
        bb=(-1j*np.sum(phase*m[perm,np.arange(len(m))],axis=1)).real
        # All orbits share aa; maximizing |bb| is optimal. Prefer sparse ties.
        candidates=np.flatnonzero(abs(bb)>=np.max(abs(bb))-3e-13)
        idx=min(candidates,key=lambda i:(len(words[i])-words[i].count('I'),words[i]))
        beta=bb[idx]
    else:
        idx=words.index(word)
        beta=(-1j*np.sum(phase[idx]*m[perm[idx],np.arange(len(m))])).real
    angle=np.arctan2(beta,aa)
    updated=np.cos(angle)*a-1j*np.sin(angle)*phase[idx,:,None]*a[perm[idx]]
    v=updated.reshape(4**width,4**start,-1).transpose(1,0,2).reshape(-1)
    return words[idx],float(angle),base.to_matrix(v,n)


def diagnostics(y,sigma,reference):
    rho=y@y.conj().T
    return dict(cp_error=float(np.linalg.norm(y-reference)),density_trace_distance=float(
        np.sum(abs(np.linalg.eigvalsh(rho-sigma)))/2),negative_weight=float(np.sum(
        np.minimum(np.linalg.eigvalsh((y+y.conj().T)/2),0)**2)),hermiticity=float(
        np.linalg.norm(y-y.conj().T)/2))


def isolated(n,j):
    support=[]; scale=[]
    for t in [0.,.3]:
        x=base.analytic_root(n,t);sig=cs.noise(x@x.conj().T,n,j,.1);root=cs.positive_root(sig)
        for start,w in [(j,1),(j-1,2),(j,2),(j-1,3),(0,n)]:
            y=kraus_polar(x,n,j,.1,start,w)
            row=dict(t=t,start=start,width=w,leakage=float(max(0,1-np.linalg.norm(y)**2)),
                     **diagnostics(y,sig,root))
            if w<=3:
                row['best_doubled_gate_cp_error']=best_gate_error(base.to_interleaved(x,n),
                    base.to_interleaved(root,n),n,start,w)
            support.append(row)
        # Discover a word on the mixed sample, then hold it fixed across all h and t.
    x=base.analytic_root(n,.3)
    word,_,_=find_angle(x,kraus_polar(x,n,j,.1,j-1,3),n,j-1,3)
    for t in [0.,.3]:
        x=base.analytic_root(n,t)
        for h in [.1,.01,.001,.0001,.00001]:
            y=kraus_polar(x,n,j,h,j-1,3)
            _,angle,v=find_angle(x,y,n,j-1,3,word)
            sig=cs.noise(x@x.conj().T,n,j,h)
            scale.append(dict(t=t,h=h,word=word,angle=angle,angle_over_h=abs(angle)/h,
                         angle_over_sqrt_h=abs(angle)/np.sqrt(h),**diagnostics(v,sig,cs.positive_root(sig))))
    return support,scale


def trajectory(n,sweeps):
    x0=base.analytic_root(n,0.);sig0=x0@x0.conj().T
    rows=[]
    for method in ['kraus_polar','pauli_rotation']:
        x=x0.copy();sig=sig0.copy()
        for step in range(n*sweeps):
            j=step%n;start=max(0,j-1);w=min(n,j+2)-start
            sig=cs.noise(sig,n,j,.1)
            y=kraus_polar(x,n,j,.1,start,w)
            row=dict(method=method,step=step+1,site=j,width=w,leakage=float(max(0,1-np.linalg.norm(y)**2)))
            if method=='pauli_rotation':
                word,angle,y=find_angle(x,y,n,start,w);row.update(word=word,angle=angle)
            y=y/np.linalg.norm(y)
            state=cs.from_vector(base.to_interleaved(y,n),n,chi=4)
            x=base.to_matrix(state.dense(),n)
            row.update(recompression_error=float(np.linalg.norm(x-y)),bond=state.bond_dimension())
            row.update(diagnostics(x,sig,cs.positive_root(sig)))
            rows.append(row)
    return rows


def mps_polar_step(state,j,h,start,width):
    """Root-free MPS update; only the active reference window is materialized."""
    left=np.ones(1,complex);right=left.copy()
    def rescale(v):
        k=np.linalg.norm(v)
        if k==0: raise ValueError('Zero trace environment: trace cannot select a rotation.')
        return v/k
    for a in state.tensors[:start]:
        left=rescale(left@(a[:,0,:]+a[:,3,:]))
    for a in reversed(state.tensors[start+width:]):
        right=rescale((a[:,0,:]+a[:,3,:])@right)
    amp=left
    for a in state.tensors[start:start+width]:
        amp=np.tensordot(amp,a,axes=(-1,0))
    partial=base.to_matrix((amp@right).reshape(-1),width)
    p=-np.expm1(-h)/2; dim=2**width
    k=[np.sqrt(1-p)*np.eye(dim),np.diag(np.sqrt(p)*(1-2*((np.arange(dim)>>(width-1-j+start))&1)))]
    m=np.concatenate([op@partial for op in k],axis=1)
    u,_,vh=np.linalg.svd(m,full_matrices=False)
    ws=np.vsplit(vh.conj().T@u.conj().T,2)
    standard=sum(np.kron(op,w.T) for op,w in zip(k,ws))
    idx=base.to_interleaved(np.arange(dim*dim).reshape(dim,dim),width)
    gate=standard[np.ix_(idx,idx)]
    pos=len(state.discarded_weights)
    state.apply_gate(start,width,gate)
    discarded=sum(state.discarded_weights[pos:])
    norm=state.norm();state.tensors[state.center]/=norm
    return float(max(0.,1-norm**2-discarded)),float(discarded)


def cluster_mps_reference(n,t):
    q=np.exp(-t);sp,sm=np.sqrt((1+q)/2),np.sqrt((1-q)/2)
    v=np.array([sp+sm,sp-sm,sp-sm,sp+sm]).reshape(1,4,1)/2
    state=base.MPS(n)
    state.tensors=[v.copy() for _ in range(n)]
    phase=np.array([(-1)**((a//2)*(b//2)+(a%2)*(b%2)) for a in range(4) for b in range(4)])
    for j in range(n-1):state.apply_gate(j,2,np.diag(phase))
    return state


def mps_benchmark(n,sweeps):
    state=base.prepare_cluster_cp(n);rows=[]
    for sweep in range(sweeps):
        leak=[];trunc=[]
        for j in range(n):
            start=max(0,j-1);width=min(n,j+2)-start
            l,d=mps_polar_step(state,j,.1,start,width);leak.append(l);trunc.append(d)
        ref=cluster_mps_reference(n,.1*(sweep+1))
        env=np.ones((1,1),complex)
        for a,b in zip(ref.tensors,state.tensors):
            env=np.einsum('ab,asr,bst->rt',env,a.conj(),b,optimize=True)
        ov=env.item()/(ref.norm()*state.norm())
        row=dict(n=n,sweep=sweep+1,max_leakage=max(leak),max_discarded=max(trunc),
                 overlap_discrepancy=float(abs(1-ov)),bond=state.bond_dimension())
        if n<=6:
            row['dense_cp_error']=float(np.linalg.norm(state.dense()-ref.dense()))
        rows.append(row)
        assert abs(1-ov)<1e-10 and max(leak)<1e-10 and max(trunc)<1e-15, row
    return rows


def constrained_trace(n,j,t,tols):
    """Optimize the two tensors directly; root-free SLSQP with a residual bound."""
    x=base.analytic_root(n,t)
    state=cs.from_vector(base.to_interleaved(x,n),n,chi=4)
    sig=cs.noise(x@x.conj().T,n,j,.1); root=cs.positive_root(sig)  # diagnostics only
    prob=cs.LocalProblem(state,j,2,sig)
    dl,_,dr=prob.shape
    u,s,vh=np.linalg.svd(prob.initial.reshape(dl*4,4*dr),full_matrices=False)
    k=4
    a=(u[:,:k]*np.sqrt(s[:k])).reshape(dl,4,k)
    b=(np.sqrt(s[:k,None])*vh[:k]).reshape(k,4,dr)
    z=prob.pack(np.r_[a.ravel(),b.ravel()])
    eye=np.eye(2**n)/np.sqrt(2**n)
    class Eval:
        key=None
        def __call__(self,p):
            if self.key is not None and np.array_equal(p,self.key): return self.value
            self.key=p.copy()
            zz=p[:p.size//2]+1j*p[p.size//2:]
            aa=zz[:a.size].reshape(a.shape); bb=zz[a.size:].reshape(b.shape)
            block=(aa.reshape(dl*4,k)@bb.reshape(k,4*dr)).reshape(prob.shape)
            norm=np.linalg.norm(block);block=block/norm
            amp=prob.embed(block); res=amp@amp.conj().T-sig
            relative=np.linalg.norm(res)/np.sqrt(prob.purity)
            def chain(g):
                g=prob.adjoint(g)
                g=((g-block*np.vdot(block,g).real)/norm).reshape(dl*4,4*dr)
                ga=g@bb.reshape(k,4*dr).conj().T
                gb=aa.reshape(dl*4,k).conj().T@g
                return prob.pack(np.r_[ga.ravel(),gb.ravel()])
            self.value=(-np.vdot(eye,amp).real,chain(-eye),relative,
                        chain(2*(res@amp)/(max(relative,1e-15)*prob.purity)),amp)
            return self.value
    ev=Eval()
    # The trace gradient provides a deterministic kick even at a pure stationary density.
    z=z-.02*ev(z)[1]/np.linalg.norm(ev(z)[1])
    rows=[]
    for eps in tols:
        fun=lambda p:ev(p)[:2]
        con=dict(type='ineq',fun=lambda p:eps-ev(p)[2],jac=lambda p:-ev(p)[3])
        opt=minimize(fun,z,jac=True,constraints=[con],method='SLSQP',
                     options=dict(maxiter=250,ftol=2e-12,disp=False))
        z=opt.x; value,_,relative,_,amp=ev(z)
        rows.append(dict(t=t,tolerance=eps,relative_hs=float(relative),feasible=bool(relative<=eps*1.01),
                         normalized_trace=float(-value),success=bool(opt.success),message=str(opt.message),
                         iterations=int(opt.nit),**diagnostics(amp,sig,root)))
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir',type=Path,default=Path(__file__).parent/'branch_results')
    parser.add_argument('--part',choices=['gates','tensors','mps','all'],default='all')
    args=parser.parse_args()
    args.outdir.mkdir(parents=True,exist_ok=True)
    if args.part in ('all','gates'):
        sup,sc=isolated(6,2);traj=trajectory(6,3)
        for name,rows in [('support',sup),('angle_scaling',sc),('trajectory',traj)]:
            cs.write_csv(args.outdir/(name+'.csv'),rows)
        slopes={str(t):float(np.polyfit(np.log([r['h'] for r in sc if r['t']==t][-3:]),
            np.log([abs(r['angle']) for r in sc if r['t']==t][-3:]),1)[0]) for t in [0.,.3]}
        checks=dict(angle_slopes=slopes,largest_trajectory_cp_error=max(r['cp_error'] for r in traj),
                    largest_scaling_cp_error=max(r['cp_error'] for r in sc),example_word=sc[0]['word'])
        print(json.dumps(checks),flush=True)
        assert checks['largest_trajectory_cp_error']<2e-9 and checks['largest_scaling_cp_error']<2e-9
        (args.outdir/'gates.json').write_text(json.dumps(checks,indent=2))
    if args.part in ('all','mps'):
        rows=mps_benchmark(6,3)+mps_benchmark(64,3)
        cs.write_csv(args.outdir/'mps_trajectory.csv',rows)
        print(json.dumps(rows),flush=True)
    if args.part in ('all','tensors'):
        tr=[]
        for t in [0.,.3]:
            tr.extend(constrained_trace(6,2,t,[1e-3,1e-5,1e-7]))
            print(json.dumps(tr[-3:]),flush=True)
            cs.write_csv(args.outdir/'constrained_trace.csv',tr)

if __name__=='__main__': main()
