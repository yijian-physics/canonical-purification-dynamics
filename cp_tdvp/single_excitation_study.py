"""Spin-local loss, coherent transport, and canonical-purification benchmarks.

The exact solver stores the occupied-sector density as V V^dagger and a vacuum
weight. A small Gram matrix determines its positive square root. This is a
controlled low-excitation solver, not a generic many-body MPDO algorithm.
"""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
import argparse
import csv
from pathlib import Path
import time
import numpy as np
from scipy.linalg import expm
from scipy.sparse import diags
from scipy.sparse.linalg import expm_multiply
import local_cp_large as local
base=local.base


def save(path,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)


def gram_root(factor):
    gram=factor.conj().T@factor
    vals,vec=np.linalg.eigh((gram+gram.conj().T)/2)
    inv=np.zeros_like(vals);keep=vals>1e-13*max(vals[-1],1e-300)
    inv[keep]=1/np.sqrt(vals[keep])
    return (vec*inv)@vec.conj().T,float(np.trace(gram).real)


def cp_mps(factor,max_bond=None):
    n,r=factor.shape;k,s=gram_root(factor);d=2*r+2;end=d-1
    assert s<=1+2e-12
    right=np.zeros(d);right[0]=np.sqrt(max(0,1-s));right[end]=1.
    state=base.MPS(n,max_bond=max_bond or d)
    tensors=[]
    for j,row in enumerate(factor):
        a=np.zeros((d,4,d),complex);a[:,0,:]=np.eye(d)
        a[0,2,1:1+r]=row
        a[0,1,1+r:1+2*r]=row.conj()
        a[1:1+r,1,end]=k@row.conj()
        a[1+r:1+2*r,2,end]=row@k
        a[0,3,end]=row@k@row.conj()
        if j==0:a=a[0:1]
        if j==n-1:a=np.einsum('lsr,r->ls',a,right)[:,:,None]
        tensors.append(a)
    state.tensors=tensors;state.center=n-1;state.move_center(0)
    return state


def overlap(a,b):
    env=np.ones((1,1),complex)
    for u,v in zip(a.tensors,b.tensors):env=np.einsum('ab,asr,bsu->ru',env,u.conj(),v,optimize=True)
    return env.item()


def affinity(v,w):
    kv,sv=gram_root(v);kw,sw=gram_root(w);cross=v.conj().T@w
    occupied=float(np.trace(kv@cross@kw@cross.conj().T).real)
    vacuum=np.sqrt(max(0,1-sv)*max(0,1-sw))
    return float(occupied+vacuum),occupied,float(vacuum)


def distances(v,w):
    r=v.shape[1];_,coordinates=np.linalg.qr(np.column_stack([v,w]),mode='reduced')
    a,b=coordinates[:,:r],coordinates[:,r:]
    ka,sa=gram_root(v);kb,sb=gram_root(w)
    root_difference=a@ka@a.conj().T-b@kb@b.conj().T
    cp=np.sqrt(np.linalg.norm(root_difference)**2+(np.sqrt(max(0,1-sa))-np.sqrt(max(0,1-sb)))**2)
    physical=.5*(np.sum(abs(np.linalg.eigvalsh(a@a.conj().T-b@b.conj().T)))+abs(sa-sb))
    return float(cp),float(physical)


def physical_marginal(factor,left,right):
    n=right-left;small=factor[left:right];rho=np.zeros((2**n,2**n),complex)
    indices=[1<<(n-1-j) for j in range(n)]
    rho[np.ix_(indices,indices)]=small@small.conj().T
    rho[0,0]=max(0,1-np.linalg.norm(small)**2)
    return rho


def loss_ops(probability):
    return [np.diag([1.,np.sqrt(1-probability)]),np.array([[0.,np.sqrt(probability)],[0.,0.]])]


def wavepacket(n,center,width,momentum=0.):
    x=np.arange(n)-center
    v=np.exp(-x*x/(4*width*width)+1j*momentum*x)
    return v/np.linalg.norm(v)


def generator_action(v,onsite,hopping,rates):
    out=(-1j*onsite-.5*rates)[:,None]*v
    out[:-1]+=-1j*hopping[:,None]*v[1:]
    out[1:]+=-1j*hopping.conj()[:,None]*v[:-1]
    return out


def cp_certificate(approximate,factor_error):
    """Analytic error bound, with ordinary floating-point evaluation.

    The contractive evolution guarantees the exact factor norm is at most one.
    A rank-one bound is stronger than the general square-root Holder bound.
    """
    b=np.linalg.norm(approximate);delta=factor_error
    a_upper=min(1.,b+delta);survival_error=(a_upper+b)*delta
    if approximate.shape[1]>1:
        return float(min(np.sqrt(2.),np.sqrt(2*survival_error)))
    q=max(0.,1-b*b);q_lower=max(0.,1-a_upper*a_upper)
    denominator=np.sqrt(q)+np.sqrt(q_lower)
    vacuum_error=np.sqrt(survival_error)
    if denominator>0:vacuum_error=min(vacuum_error,survival_error/denominator)
    return float(min(np.sqrt(2.),np.sqrt(2*delta*delta+vacuum_error*vacuum_error)))


def local_integrate(factor,onsite,hopping,rates,elapsed,dt,return_certificate=False):
    if elapsed==0:
        return (factor.copy(),0.) if return_certificate else factor.copy()
    v=factor.copy();steps=int(np.ceil(elapsed/dt));step=elapsed/steps
    onsite_half=np.exp((-1j*onsite-.5*rates)*step/2)
    defect_bound=0.
    for _ in range(steps):
        if return_certificate:old=v.copy()
        v*=onsite_half[:,None]
        for parity,fraction in ((0,.5),(1,1.),(0,.5)):
            sites=np.arange(parity,len(v)-1,2);couplings=hopping[sites]
            magnitude=abs(couplings);phase=np.divide(couplings,magnitude,out=np.ones_like(couplings,dtype=complex),where=magnitude>0)
            cosine=np.cos(magnitude*step*fraction)[:,None]
            sine=np.sin(magnitude*step*fraction)[:,None]
            left=v[sites].copy();right=v[sites+1].copy()
            v[sites]=cosine*left-1j*sine*phase[:,None]*right
            v[sites+1]=cosine*right-1j*sine*phase.conj()[:,None]*left
        v*=onsite_half[:,None]
        if return_certificate:
            # Cubic Hermite interpolant in the Bernstein basis. Its endpoint
            # derivatives equal G v, so residual coefficients r0 and r3 vanish.
            g0=generator_action(old,onsite,hopping,rates)
            g1=generator_action(v,onsite,hopping,rates)
            b1=old+step*g0/3;b2=v-step*g1/3
            middle=3*(b2-b1)/step
            r1=g0/3+2*middle/3-generator_action(b1,onsite,hopping,rates)
            r2=2*middle/3+g1/3-generator_action(b2,onsite,hopping,rates)
            defect_bound+=step*(np.linalg.norm(r1)+np.linalg.norm(r2))/4
    return (v,float(defect_bound)) if return_certificate else v


def reference_integrate(factor,onsite,hopping,rates,elapsed):
    generator=diags([-1j*hopping.conj(),-1j*onsite-.5*rates,-1j*hopping],[-1,0,1],format='csr')
    return expm_multiply(elapsed*generator,factor)


def two_region_root(vector,active,basis_a,basis_e):
    outside=np.array([j for j in range(len(vector)) if j not in set(active)])
    va=basis_a.conj().T@vector[active];ve=basis_e.conj().T@vector[outside]
    da,de=len(va)+1,len(ve)+1
    compressed=np.zeros((da,de),complex);compressed[1:,0]=va;compressed[0,1:]=ve
    flat=compressed.reshape(-1);s=np.vdot(flat,flat).real
    rho=np.outer(flat,flat.conj());rho[0,0]+=max(0,1-s)
    root=local.previous.small_positive_root(rho)
    return root.reshape(da,de,da,de).transpose(0,2,1,3).reshape(da*da,de*de)


def unitary_floor(v,w,left,right):
    active=np.arange(left,right);outside=np.r_[0:left,right:len(v)]
    def basis(indices):
        u,s,_=np.linalg.svd(np.column_stack([v[indices],w[indices]]),full_matrices=False)
        return u[:,s>1e-12*max(s[0],1e-300)]
    a,e=basis(active),basis(outside)
    old=two_region_root(v,active,a,e);new=two_region_root(w,active,a,e)
    best=float(np.linalg.svd(new@old.conj().T,compute_uv=False).sum())
    return np.sqrt(max(0,2-2*best))


def validate(outdir):
    rng=np.random.default_rng(20261104);rows=[]
    for n in (3,5,7):
        for rank in (1,3):
            f=rng.normal(size=(n,rank))+1j*rng.normal(size=(n,rank));f*=np.sqrt(.7)/np.linalg.norm(f)
            rho=physical_marginal(f,0,n);target=local.previous.small_positive_root(rho)
            state=cp_mps(f)
            error=np.linalg.norm(state.dense()-base.to_interleaved(target,n))
            assert error<1e-11
            g=rng.normal(size=(n,rank))+1j*rng.normal(size=(n,rank));g*=np.sqrt(.6)/np.linalg.norm(g)
            exact=float(np.trace(target@local.previous.small_positive_root(physical_marginal(g,0,n))).real)
            computed=affinity(f,g)[0];contracted=overlap(state,cp_mps(g)).real
            assert max(abs(computed-exact),abs(contracted-exact))<1e-11
            rows.append(dict(test='positive root and overlap',n=n,rank=rank,root_error=error,affinity_error=abs(computed-exact),mps_error=abs(contracted-exact)))
    # Independent full spin-Lindblad evolution, with inhomogeneous loss.
    n=4;sm=np.array([[0.,1.],[0.,0.]])
    ops=[base.kron_all([sm if k==j else base.I for k in range(n)]) for j in range(n)]
    ons=np.linspace(-.2,.3,n);hops=np.full(n-1,.8);rates=np.linspace(.1,.7,n)
    h=sum(ons[j]*ops[j].conj().T@ops[j] for j in range(n))
    h+=sum(hops[j]*(ops[j].conj().T@ops[j+1]+ops[j+1].conj().T@ops[j]) for j in range(n-1))
    ident=np.eye(2**n);liouv=-1j*(np.kron(ident,h)-np.kron(h.T,ident))
    for rate,op in zip(rates,ops):
        number=op.conj().T@op
        liouv+=rate*(np.kron(op.conj(),op)-.5*np.kron(ident,number)-.5*np.kron(number.T,ident))
    v=wavepacket(n,1.3,1.,.7)[:,None]
    initial=physical_marginal(v,0,n)
    target=(expm(.7*liouv)@initial.reshape(-1,order='F')).reshape(initial.shape,order='F')
    f=reference_integrate(v,ons,hops,rates,.7)
    error=np.linalg.norm(physical_marginal(f,0,n)-target)
    assert error<1e-11
    rows.append(dict(test='full spin Lindblad',n=n,rank=1,density_error=error))
    # Check the compressed local-unitary optimum against a full spin unfolding.
    n=5;v=wavepacket(n,2.,1.3,.4)[:,None];w=v.copy();w[2]*=np.sqrt(.7)
    old=local.previous.small_positive_root(physical_marginal(v,0,n))
    new=local.previous.small_positive_root(physical_marginal(w,0,n))
    old=local.previous.branch.unfold(base.to_interleaved(old,n),n,1,3)
    new=local.previous.branch.unfold(base.to_interleaved(new,n),n,1,3)
    dense_floor=np.sqrt(max(0,2-2*np.linalg.svd(new@old.conj().T,compute_uv=False).sum()))
    floor_error=abs(unitary_floor(v[:,0],w[:,0],1,4)-dense_floor)
    assert floor_error<1e-10
    rows.append(dict(test='local unitary floor',n=n,rank=1,error=floor_error))
    # Complex hopping and mixed inputs, independently checked with expm_multiply.
    for rank in (1,3):
        n=13;ons=rng.normal(size=n);hops=rng.normal(size=n-1)+1j*rng.normal(size=n-1)
        rates=rng.uniform(.2,1.,size=n)
        v=rng.normal(size=(n,rank))+1j*rng.normal(size=(n,rank));v/=np.linalg.norm(v)
        ref=reference_integrate(v,ons,hops,rates,.7)
        approx,bound=local_integrate(v,ons,hops,rates,.7,.05,True)
        factor_error=np.linalg.norm(approx-ref);cp_error=distances(approx,ref)[0]
        cp_bound=cp_certificate(approx,bound)
        assert factor_error<bound and cp_error<cp_bound
        rows.append(dict(test='complex hopping certificate',n=n,rank=rank,
                         factor_error=factor_error,factor_error_bound=bound,
                         root_error=cp_error,cp_error_bound=cp_bound))
    save(outdir/'validation.csv',rows);print('validation passed',flush=True)


def isolated(outdir):
    rows=[];n=64;j=32;prob=.3
    for kind in ('uniform','packet'):
        v=(np.ones(n)/np.sqrt(n) if kind=='uniform' else wavepacket(n,j,2.))[:,None]
        w=v.copy();w[j]*=np.sqrt(1-prob)
        for radius in (0,1,2,4,8,16):
            start=max(0,j-radius);stop=min(n,j+radius+1)
            floor=unitary_floor(v[:,0],w[:,0],start,stop)
            row=dict(profile=kind,n=n,radius=radius,active_width=stop-start,probability=prob,best_local_unitary_error=floor)
            if radius<=2:
                left=max(0,start-1);right=min(n,stop+1)
                gate=local.fit_gate(physical_marginal(v,left,right),[j-left],start-left,stop-start,loss_ops(prob))
                state=cp_mps(v,max_bond=16);local.apply_compact(state,start,stop-start,gate)
                error=np.sqrt(max(0,overlap(state,state).real+1-2*overlap(state,cp_mps(w)).real))
                row.update(local_fit=gate.residual,achieved_cp_error=error)
                assert error+2e-7>=floor
            rows.append(row)
    save(outdir/'isolated_locality.csv',rows);print('isolated locality done',flush=True)


def buffer_evolution(outdir):
    rows=[]
    configurations=[('uniform',16,1,16),('uniform',64,1,16),('uniform',128,1,16),
                    ('uniform',64,2,4),('uniform',64,2,16),('uniform',128,2,16),
                    ('packet',64,1,16),('packet',64,2,16),('site',64,0,4)]
    for kind,n,radius,chi in configurations:
        started=time.perf_counter();j=n//2
        if kind=='uniform':initial=np.ones(n,complex)/np.sqrt(n)
        elif kind=='packet':initial=wavepacket(n,j,2.,.4)
        else:initial=np.eye(n)[:,j].astype(complex)
        f=initial[:,None];state=cp_mps(f,max_bond=chi);period=2*radius+1
        max_fit=0.;discard=[]
        for step in range(3):
            for color in range(period):
                sites=list(range(color,n,period));gates=[]
                for site in sites:
                    start=max(0,site-radius);stop=min(n,site+radius+1)
                    left=max(0,start-1);right=min(n,stop+1)
                    gate=local.fit_gate(physical_marginal(f,left,right),[site-left],start-left,stop-start,loss_ops(-np.expm1(-.1)))
                    gates.append((start,stop,gate));max_fit=max(max_fit,gate.residual)
                for start,stop,gate in gates:discard.extend(local.apply_compact(state,start,stop-start,gate))
                f[sites]*=np.exp(-.05)
        exact=cp_mps(f);cp_error=np.sqrt(max(0,overlap(state,state).real+1-2*overlap(state,exact).real))
        probe=min(j,n-2)
        local_error=local.trace_distance(local.doubled_marginal(state,probe,2),local.doubled_marginal(exact,probe,2))
        contraction=np.ones(1,complex)
        for tensor in state.tensors:contraction=contraction@tensor[:,0,:]
        row=dict(profile=kind,n=n,radius=radius,chi=chi,time=.3,cp_error=cp_error,
                 doubled_two_site_error=local_error,vacuum_cp_amplitude=abs(contraction.item()),
                 exact_vacuum_cp_amplitude=np.sqrt(1-np.linalg.norm(f)**2),max_local_fit=max_fit,
                 max_discarded=max(discard,default=0.),seconds=time.perf_counter()-started)
        rows.append(row);save(outdir/'buffer_evolution.csv',rows)
        print('buffer',kind,n,radius,chi,cp_error,local_error,flush=True)


def model(n):
    x=np.arange(n)-n//2
    onsite=.23*np.sin(.37*x)+.07*np.cos(.13*x)
    hopping=(.9+.15*np.cos(.29*(x[:-1]+.5))).astype(complex)
    rates=.04+.8*np.exp(-(x-2)**2/(2*6**2))
    return onsite,hopping,rates


def transport(outdir):
    rows=[]
    for n in (64,256,1024):
        initial=wavepacket(n,n//2-5,3.,.6)[:,None];ons,hops,rates=model(n)
        target=reference_integrate(initial,ons,hops,rates,2.)
        for dt in (.2,.1,.05,.025):
            started=time.perf_counter();v,bound=local_integrate(initial,ons,hops,rates,2.,dt,True)
            error,physical=distances(v,target)
            cp_bound=cp_certificate(v,bound)
            assert np.linalg.norm(v-target)<=bound+1e-12
            assert error<=cp_bound+1e-12
            state=cp_mps(v)
            row=dict(n=n,time=2.,dt=dt,cp_error=error,physical_error=physical,
                     factor_error=np.linalg.norm(v-target),factor_error_bound=bound,cp_error_bound=cp_bound,
                     survival=np.linalg.norm(v)**2,bond=state.bond_dimension(),
                     norm_error=abs(overlap(state,state).real-1),seconds=time.perf_counter()-started)
            rows.append(row);save(outdir/'transport.csv',rows)
            print('transport',n,dt,error,flush=True)
    rows=[]
    for n in (64,256):
        ons,hops,rates=model(n)
        a=wavepacket(n,n//2-3,3.,.6);b=wavepacket(n,n//2+3,3.,-.4)
        b-=a*np.vdot(a,b);b/=np.linalg.norm(b)
        for t in (0.,.2,.5,1.,2.,4.):
            va=reference_integrate(a[:,None],ons,hops,rates,t)
            vb=reference_integrate(b[:,None],ons,hops,rates,t)
            exact,occupied,vacuum=affinity(va,vb)
            contracted=overlap(cp_mps(va),cp_mps(vb)).real
            uniform_a=reference_integrate(a[:,None],ons,hops,np.full(n,.3),t)
            uniform_b=reference_integrate(b[:,None],ons,hops,np.full(n,.3),t)
            uniform=affinity(uniform_a,uniform_b)[0]
            assert abs(uniform-(1-np.exp(-.3*t)))<1e-10
            assert abs(contracted-exact)<1e-10
            approx_a,bound_a=local_integrate(a[:,None],ons,hops,rates,t,.05,True)
            approx_b,bound_b=local_integrate(b[:,None],ons,hops,rates,t,.05,True)
            approx_affinity=affinity(approx_a,approx_b)[0]
            error_bound=min(1.,cp_certificate(approx_a,bound_a)+cp_certificate(approx_b,bound_b))
            assert abs(approx_affinity-exact)<=error_bound+1e-12
            rows.append(dict(n=n,time=t,affinity=exact,Holevo_squared=exact**2,
                             occupied_contribution=occupied,vacuum_contribution=vacuum,
                             mps_affinity_error=abs(contracted-exact),uniform_affinity=uniform,
                             split_affinity=approx_affinity,split_affinity_error=abs(approx_affinity-exact),
                             affinity_error_bound=error_bound))
    save(outdir/'affinity.csv',rows)


def mixed_inputs(outdir):
    rows=[]
    for n in (32,128,512):
        ons,hops,rates=model(n);rng=np.random.default_rng(1204)
        for rank in (1,3,8):
            q,_=np.linalg.qr(rng.normal(size=(n,rank))+1j*rng.normal(size=(n,rank)))
            u,_=np.linalg.qr(rng.normal(size=(n,rank))+1j*rng.normal(size=(n,rank)))
            initial=q/np.sqrt(rank);other=u/np.sqrt(rank)
            started=time.perf_counter()
            v=reference_integrate(initial,ons,hops,rates,1.3);w=reference_integrate(other,ons,hops,rates,1.3)
            a,b=cp_mps(v),cp_mps(w);exact=affinity(v,w)[0];contracted=overlap(a,b).real
            assert abs(exact-contracted)<1e-10
            rows.append(dict(n=n,occupied_rank=rank,cp_bond=a.bond_dimension(),
                             affinity=exact,contraction_error=abs(exact-contracted),
                             seconds=time.perf_counter()-started))
    save(outdir/'mixed_inputs.csv',rows);print('mixed inputs done',flush=True)


def pair_cp(probabilities):
    n=len(probabilities)+1;state=base.MPS(n,max_bond=2)
    plus=np.ones(2)/np.sqrt(2);minus=np.array([1.,-1.])/np.sqrt(2)
    vec=[np.kron(plus,plus),np.kron(minus,minus)];tensors=[]
    for j in range(n):
        a=np.zeros((2,4,2))
        for before in (0,1):
            for after in (0,1):
                weight=np.sqrt(probabilities[j] if after else 1-probabilities[j]) if j<n-1 else 1.
                a[before,:,after]=weight*vec[before^after]
        if j==0:a=a[0:1]
        if j==n-1:a=a[:,:,0:1]
        tensors.append(a)
    state.tensors=tensors;state.center=n-1;state.move_center(0)
    return state


def pair_control(outdir):
    rows=[]
    for n in (6,64,256,1024):
        rates=.3+.2*np.sin(np.arange(n-1)*.31)**2
        p=-np.expm1(-2*rates*.7)/2;q=-np.expm1(-2*rates*.8)/2
        a,b=pair_cp(p),pair_cp(q);value=overlap(a,b).real
        exact=float(np.prod(np.sqrt(p*q)+np.sqrt((1-p)*(1-q))))
        row=dict(n=n,cp_bond=a.bond_dimension(),affinity=value,exact_affinity=exact,error=abs(value-exact))
        if n==6:
            psi=np.ones(2**n)/np.sqrt(2**n);rho=np.outer(psi,psi)
            for j,prob in enumerate(p):
                op=base.kron_all([base.Z if i in (j,j+1) else base.I for i in range(n)])
                rho=(1-prob)*rho+prob*op@rho@op
            row['dense_cp_error']=np.linalg.norm(a.dense()-base.to_interleaved(local.previous.small_positive_root(rho),n))
            assert row['dense_cp_error']<1e-11
        assert row['error']<1e-10
        rows.append(row)
    save(outdir/'pair_control.csv',rows);print('pair control done',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--part',choices=['all','validate','isolated','buffer','transport','mixed','pair'],default='all')
    p.add_argument('--outdir',type=Path,default=Path(__file__).parent/'single_excitation_results')
    args=p.parse_args();args.outdir.mkdir(parents=True,exist_ok=True)
    if args.part in ('all','validate'):validate(args.outdir)
    if args.part in ('all','isolated'):isolated(args.outdir)
    if args.part in ('all','buffer'):buffer_evolution(args.outdir)
    if args.part in ('all','transport'):transport(args.outdir)
    if args.part in ('all','mixed'):mixed_inputs(args.outdir)
    if args.part in ('all','pair'):pair_control(args.outdir)


if __name__=='__main__':main()
