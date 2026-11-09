"""Direct local Hamiltonian fits for dephasing of cluster and mixed-field Ising chains.

The solver receives the old amplitude and noisy density only.  The positive root
and the target-aware Procrustes bound are computed separately for scoring.
"""
import argparse
from functools import lru_cache
from pathlib import Path
import cp_branch_methods as branch
import cluster_dephasing as base
import cp_tensor_search as cs
import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.optimize import minimize


def ising_amplitude(n, beta):
    indices = np.arange(2**n)
    signs = np.array([1-2*((indices >> (n-1-i)) & 1) for i in range(n)])
    hamiltonian = np.diag((-np.sum(signs[:-1]*signs[1:],axis=0)-.35*signs.sum(axis=0)).astype(float))
    for j in range(n):
        hamiltonian[indices, indices ^ (1 << (n-1-j))] -= 1.05
    energies, vectors = np.linalg.eigh(hamiltonian)
    if beta == 'ground':
        return np.outer(vectors[:,0],vectors[:,0])
    weights = np.exp(-float(beta)*(energies-energies[0])/2)
    return (vectors*(weights/np.linalg.norm(weights)))@vectors.T


def start_of_window(j,width,n):
    return max(0,min(n-width,j-(width-1)//2))


def dephased_root(x,n,j,h):
    # SVD of a Kraus amplitude avoids an eigenvalue cutoff on cold Gibbs states.
    p=-np.expm1(-h)/2
    z=1-2*((np.arange(2**n)>>(n-1-j))&1)
    k=np.concatenate([np.sqrt(1-p)*x,np.sqrt(p)*z[:,None]*x],axis=1)
    u,s,_=np.linalg.svd(k,full_matrices=False)
    return (u*s)@u.T


def site_kraus(h,channel):
    if channel=='Z':
        p=-np.expm1(-h)/2
        return [np.sqrt(1-p)*np.eye(2),np.sqrt(p)*np.diag([1.,-1.])]
    if channel=='Y_damping':
        # Damping toward |+y>, a nonunital channel with complex Kraus matrices.
        basis=np.array([[1,1],[1j,-1j]])/np.sqrt(2)
        operators=[np.diag([1.,np.exp(-h/2)]),np.array([[0,np.sqrt(-np.expm1(-h))],[0,0]])]
        return [basis@k@basis.conj().T for k in operators]
    raise ValueError(channel)


def apply_channel_root(x,n,j,operators):
    blocks=[]
    for op in operators:
        shaped=x.reshape(2**j,2,2**(n-1-j),-1)
        blocks.append(np.einsum('ab,lbrc->larc',op,shaped).reshape(x.shape))
    u,s,_=np.linalg.svd(np.concatenate(blocks,axis=1),full_matrices=False)
    return (u*s)@u.conj().T


@lru_cache(None)
def local_sectors(width):
    # Orthogonal basis for the symmetric and antisymmetric matrices on the block.
    d = 2**width
    paired = base.to_interleaved(np.arange(d*d).reshape(d,d),width)
    exchange = np.argsort(paired)[(paired%d)*d+paired//d]
    identity = np.eye(d*d)
    plus, minus = [], []
    for i,j in enumerate(exchange):
        if i==j:
            plus.append(identity[i])
        elif i<j:
            plus.append((identity[i]+identity[j])/np.sqrt(2))
            minus.append((identity[i]-identity[j])/np.sqrt(2))
    return [np.array(columns) for columns in (plus,minus)]


class LocalHamiltonian:
    """Cayley coordinates for a real orthogonal local gate commuting with swap.

    This enforces Hermiticity of real input amplitudes. Each real rotation is
    a doubled-space Hamiltonian evolution, with exp(log Q)=Q. We do not take
    its logarithm in the objective. Cayley coordinates cover the identity chart.
    """
    def __init__(self, x, sigma, start, width):
        self.n=int(np.log2(len(x)))
        self.start, self.width=start,width
        self.sigma, self.norm_sigma=sigma,np.linalg.norm(sigma)
        self.basis=local_sectors(width)
        self.vectors=[u@self.unfold(x) for u in self.basis]
        self.indices=[np.triu_indices(len(u),1) for u in self.basis]
        self.sizes=[len(indices[0]) for indices in self.indices]
        self.dimension=sum(self.sizes)
        self.trace_matrix=np.eye(len(x))/np.sqrt(len(x))
        self.cached=None
        self.evaluations=0

    def unfold(self,x):
        return branch.unfold(base.to_interleaved(x,self.n),self.n,self.start,self.width)

    def fold(self,block):
        v=block.reshape(4**self.width,4**self.start,-1).transpose(1,0,2).reshape(-1)
        return base.to_matrix(v,self.n)

    def evaluate(self,params):
        if self.cached is not None and np.array_equal(params,self.cached):
            return self.values
        self.evaluations+=1
        parts=np.split(params,np.cumsum(self.sizes)[:-1])
        outputs, factors=[],[]
        for v,idx,z in zip(self.vectors,self.indices,parts):
            skew=np.zeros((len(v),len(v)))
            skew[idx]=z
            factor=lu_factor(np.eye(len(v))-(skew-skew.T)/2)
            out=2*lu_solve(factor,v)-v
            outputs.append(out); factors.append(factor)
        x=self.fold(sum(u.T@v for u,v in zip(self.basis,outputs)))
        residual=x@x.T-self.sigma
        relative=np.linalg.norm(residual)/self.norm_sigma
        def adjoint(g):
            unfolded=self.unfold(g)
            packed=[]
            for u,v,y,idx,fac in zip(self.basis,self.vectors,outputs,self.indices,factors):
                gradient=lu_solve(fac,u@unfolded,trans=1)@((v+y)/2).T
                packed.append((gradient-gradient.T)[idx])
            return np.concatenate(packed)
        trace=np.sum(self.trace_matrix*x)
        grad_r=adjoint(2*(residual@x)/(self.norm_sigma**2*max(relative,1e-15)))
        self.values=(x,relative,trace,grad_r,adjoint(self.trace_matrix))
        self.cached=params.copy()
        return self.values


def direct_fit(x,sigma,start,width,budget,maxiter=220,check=False):
    model=LocalHamiltonian(x,sigma,start,width)
    z=np.zeros(model.dimension)
    if check:
        rng=np.random.default_rng(1341)
        probe=.012*rng.standard_normal(z.shape)
        v=rng.standard_normal(z.shape);v/=np.linalg.norm(v)
        _,_,_,g,t=model.evaluate(probe)
        _,rp,tp,*_=model.evaluate(probe+1e-6*v)
        _,rm,tm,*_=model.evaluate(probe-1e-6*v)
        assert abs((rp-rm)/2e-6-g@v)<5e-8 and abs((tp-tm)/2e-6-t@v)<5e-8
    # Follow trace-biased solutions to the constraint. Use their directions
    # to precondition a final explicit inequality-constrained search.
    history=[];candidates=[]
    for strength in [1e-2,1e-4,1e-6,0.]:
        def cost(p):
            _,r,t,gr,gt=model.evaluate(p)
            return .5*r*r-strength*t,r*gr-strength*gt
        def record(p):
            history.append(p.copy())
        opt=minimize(cost,z,jac=True,method='L-BFGS-B',callback=record,
                     options=dict(maxiter=maxiter,ftol=1e-16,gtol=1e-11,maxcor=16))
        z=opt.x.copy()
        _,r,t,*_=model.evaluate(z)
        candidates.append((r,t,z.copy()))
        if r<budget*.9: break
    best=min(candidates,key=lambda row:row[0])
    achieved_floor=best[0]
    # If the requested budget was missed, quantify the best residual found.
    # Do not relax it and label the result feasible.
    if achieved_floor <= 1.1*budget:
        z=best[2].copy()
        snapshots=np.array(history[::max(1,len(history)//55)]+[a[2] for a in candidates])
        for refinement in range(3):
            _,_,_,gr,gt=model.evaluate(z)
            rows=np.vstack([snapshots-z,gr,gt])
            norms=np.linalg.norm(rows,axis=1)
            rows=rows[norms>1e-14]/norms[norms>1e-14,None]
            _,s,vh=np.linalg.svd(rows,full_matrices=False)
            directions=vh[:min(32,np.sum(s>s[0]*1e-10))].T
            origin=z.copy()
            def objective(coords):
                _,_,t,_,g=model.evaluate(origin+directions@coords)
                return -t,-directions.T@g
            def constraint(coords):
                return budget-model.evaluate(origin+directions@coords)[1]
            def constraint_grad(coords):
                return -directions.T@model.evaluate(origin+directions@coords)[3]
            fit=minimize(objective,np.zeros(directions.shape[1]),jac=True,method='SLSQP',
                         constraints=[dict(type='ineq',fun=constraint,jac=constraint_grad)],
                         options=dict(maxiter=120,ftol=2e-12))
            z=origin+directions@fit.x
            _,r,t,*_=model.evaluate(z)
            candidates.append((r,t,z.copy()))
            snapshots=np.vstack([snapshots,z])
    feasible=[c for c in candidates if c[0]<=budget*(1+1e-5)]
    selected=max(feasible,key=lambda c:c[1]) if feasible else min(candidates,key=lambda c:c[0])
    y=model.evaluate(selected[2])[0]
    return y,dict(relative_hs=selected[0],normalized_trace=selected[1],feasible=bool(feasible),
                  evaluations=model.evaluations,budget=budget,parameters=model.dimension)


def score(y,sigma):
    return branch.diagnostics(y,sigma,cs.positive_root(sigma))


def polar(x,n,j,h,start,w):
    out=branch.kraus_polar(x,n,j,h,start,w).real
    norm=np.linalg.norm(out)
    return out/norm,max(0.,1-norm**2)


def oracle_floor(x,target,n,start,w):
    a=branch.unfold(base.to_interleaved(x,n),n,start,w)
    b=branch.unfold(base.to_interleaved(target,n),n,start,w)
    # Compute the optimal vector error directly for modest local blocks.
    if w<=3:return branch.best_gate_error(base.to_interleaved(x,n),base.to_interleaved(target,n),n,start,w)
    # For larger windows the spectator side is smaller; compare environment
    # density amplitudes instead of forming an enormous local gate.
    ra=np.linalg.qr(a,mode='r'); rb=np.linalg.qr(b,mode='r')
    u,_,vh=np.linalg.svd(rb@ra.conj().T,full_matrices=False)
    return float(np.linalg.norm((u@vh)@ra-rb))


def isolated(outdir,fast=False):
    rows=[]
    n,j,h=6,2,.1
    for name,x in [('cluster pure',base.analytic_root(n,0).real),
                   ('cluster mixed',base.analytic_root(n,.3).real),
                   ('Ising beta 0.8',ising_amplitude(n,.8)),
                   ('Ising beta 1.6',ising_amplitude(n,1.6)),
                   ('Ising ground',ising_amplitude(n,'ground'))]:
        sigma=cs.noise(x@x.T,n,j,h)
        root=dephased_root(x,n,j,h)
        for w in ([3] if fast else [2,3]):
            start=start_of_window(j,w,n)
            y,info=direct_fit(x,sigma,start,w,1e-5,check=True)
            z,ell=polar(x,n,j,h,start,w)
            row=dict(case=name,n=n,site=j,width=w,h=h,**info,**branch.diagnostics(y,sigma,root),
                     oracle_cp=oracle_floor(x,root,n,start,w),
                     polar_cp=np.linalg.norm(z-root),polar_leak=ell,
                     no_update_cp=np.linalg.norm(x-root))
            rows.append(row);cs.write_csv(outdir/'direct.csv',rows)
            print(name,w,info['feasible'],info['relative_hs'],row['cp_error'],row['oracle_cp'],row['polar_cp'],flush=True)


def scan_windows(outdir):
    rows=[]
    for n in [6,8]:
        j=n//2-1;h=.1
        for beta in [.8,1.6,'ground']:
            x=ising_amplitude(n,beta)
            rho=x@x.T
            sigma=cs.noise(rho,n,j,h);root=dephased_root(x,n,j,h)
            for w in list(range(1,n+1)):
                start=start_of_window(j,w,n)
                y,ell=polar(x,n,j,h,start,w)
                region=range(start,start+w)
                cmi_drop=(base.entropy(sigma)-base.entropy(rho)
                          +base.entropy(base.marginal(rho,n,region))-base.entropy(base.marginal(sigma,n,region)))
                rows.append(dict(n=n,beta=beta,site=j,h=h,start=start,width=w,leakage=ell,
                                 cmi_drop_nats=cmi_drop,no_update_cp=np.linalg.norm(x-root),
                                 **branch.diagnostics(y,sigma,root)))
                if w<=5:
                    floor=oracle_floor(x,root,n,start,w)
                    rows[-1]['best_gate_cp']=floor
                    assert floor <= np.sqrt(max(0,cmi_drop))+1e-6
            assert rows[-1]['cp_error']<1e-8
            cs.write_csv(outdir/'windows.csv',rows)


def local_density(state,start,stop):
    """Physical density on a contiguous region, contracting references and bonds."""
    state.move_center(start)
    block=state.tensors[start]
    for tensor in state.tensors[start+1:stop]:
        block=np.tensordot(block,tensor,axes=(-1,0))
    w=stop-start
    axes=[1+2*k for k in range(w)]+[0]+[2+2*k for k in range(w)]+[2*w+1]
    amplitude=block.reshape([block.shape[0]]+[2]*(2*w)+[block.shape[-1]]).transpose(axes).reshape(2**w,-1)
    return np.real_if_close(amplitude@amplitude.conj().T)


def small_positive_root(rho):
    eig,vec=np.linalg.eigh((rho+rho.conj().T)/2)
    return (vec*np.sqrt(np.where(eig>2e-15*eig[-1],eig,0)))@vec.conj().T


def fit_marginal_gate(rho,local_site,core_start,width,operators):
    """Fit a doubled gate to two physical marginals related by a one-site channel.

    rho covers both the active sites and the surrounding buffer. local_site and
    core_start are relative to that physical marginal. operators are Kraus matrices.
    """
    length=int(np.log2(rho.shape[0]))
    source=small_positive_root(rho)
    target=apply_channel_root(source,length,local_site,operators)
    a=branch.unfold(base.to_interleaved(source,length),length,core_start,width)
    b=branch.unfold(base.to_interleaved(target,length),length,core_start,width)
    # Work on the union of the two column spaces and leave its complement fixed.
    basis,s,_=np.linalg.svd(np.column_stack([a,b]),full_matrices=False)
    basis=basis[:,s>1e-13*s[0]]
    a_small,b_small=basis.conj().T@a,basis.conj().T@b
    cross=b_small@a_small.conj().T+1e-12*np.eye(basis.shape[1])
    u,_,vh=np.linalg.svd(cross)
    rotation=u@vh
    gate=np.eye(4**width)+basis@(rotation-np.eye(len(rotation)))@basis.conj().T
    return gate,float(np.linalg.norm(gate@a-b))


def apply_buffer(state,j,h,w,halo,channel='Z',physical_marginal=None):
    start=start_of_window(j,w,state.n)
    left=max(0,start-halo);right=min(state.n,start+w+halo)
    rho=local_density(state,left,right) if physical_marginal is None else physical_marginal
    gate,local_fit=fit_marginal_gate(rho,j-left,start-left,w,site_kraus(h,channel))
    pos=len(state.discarded_weights)
    state.apply_gate(start,w,gate)
    discarded=sum(state.discarded_weights[pos:])
    norm=state.norm();state.tensors[state.center]/=norm
    return dict(local_fit=local_fit,marginal_width=right-left,discarded=discarded,bond=state.bond_dimension())


def cmi_nats(rho,n,a,b):
    a,b=set(a),set(b);bc=set(range(n))-a
    def entropy(sites):
        return base.entropy(base.marginal(rho,n,sorted(sites))) if sites else 0.
    return max(0.,entropy(a|b)+entropy(bc)-entropy(b)-base.entropy(rho))


def buffer_tests(outdir):
    rows=[]
    for n in [6,8]:
        j=n//2-1;h=.1
        for beta in [.8,1.6,'ground']:
            x=ising_amplitude(n,beta)
            sigma=cs.noise(x@x.T,n,j,h);root=dephased_root(x,n,j,h)
            for w,halo in [(2,1),(3,0),(3,1),(3,2),(5,1)]:
                state=cs.from_vector(base.to_interleaved(x,n),n,chi=4**(n//2))
                row=apply_buffer(state,j,h,w,halo)
                amp=base.to_matrix(state.dense(),n)
                start=start_of_window(j,w,n)
                core=set(range(start,start+w))
                surrounding=set(range(max(0,start-halo),min(n,start+w+halo)))-core
                before=cmi_nats(x@x.T,n,core,surrounding)
                after=cmi_nats(sigma,n,core,surrounding)
                margin=2*np.sqrt((before+after)/2)
                global_error=np.linalg.norm(amp-root)
                assert abs(global_error-row['local_fit'])<=margin+3e-7
                rows.append(dict(n=n,beta=beta,site=j,width=w,halo=halo,cmi_before=before,cmi_after=after,
                                 predicted_upper=row['local_fit']+margin,
                                 **row,**branch.diagnostics(amp,sigma,root)))
            cs.write_csv(outdir/'buffered.csv',rows)
            print('buffer',n,beta,[round(v['cp_error'],7) for v in rows[-5:]],flush=True)


def evolve(outdir):
    rows=[]
    for n,beta in [(6,.8),(8,.8),(8,1.6)]:
        h=.1;num_sweeps=2
        initial=ising_amplitude(n,beta)
        methods=[('polar',3,0,24),('polar',5,0,24),('buffered',3,1,24),('buffered',5,1,24)]
        if n==6:methods.append(('direct',3,0,24))
        if n==8 and beta==.8:methods.extend([('buffered',5,1,8),('buffered',5,1,48)])
        references=[]
        ref=initial
        for step in range(n*num_sweeps):
            ref=dephased_root(ref,n,step%n,h)
            # Independent channel check uses the density, not the previous SVD factors.
            prev=references[-1][1] if references else initial@initial.T
            sig=cs.noise(prev,n,step%n,h)
            assert np.linalg.norm(ref@ref.T-sig)<3e-13
            references.append((ref,sig))
        for name,w,halo,chi in methods:
            state=cs.from_vector(base.to_interleaved(initial,n),n,chi=chi)
            norm=state.norm();state.tensors[state.center]/=norm
            init_err=np.linalg.norm(state.dense()-base.to_interleaved(initial,n))
            for i,(ref,sig) in enumerate(references):
                site=i%n; start=start_of_window(site,w,n)
                if name=='polar':
                    leak,discarded=branch.mps_polar_step(state,site,h,start,w)
                    info=dict(leakage=leak,discarded=discarded)
                elif name=='buffered':
                    info=apply_buffer(state,site,h,w,halo)
                else:
                    old=base.to_matrix(state.dense(),n).real
                    physical=cs.noise(old@old.T,n,site,h)
                    y,info=direct_fit(old,physical,start,w,1e-3,maxiter=180)
                    state=cs.from_vector(base.to_interleaved(y,n),n,chi=chi)
                    info['recompression']=np.linalg.norm(base.to_matrix(state.dense(),n)-y)
                y=base.to_matrix(state.dense(),n)
                rows.append(dict(n=n,beta=beta,method=name,width=w,halo=halo,cap=chi,step=i+1,
                                 step_size=h,initial_error=init_err,bond_used=state.bond_dimension(),
                                 **info,**branch.diagnostics(y,sig,ref)))
            cs.write_csv(outdir/'evolution.csv',rows)
            print('evolution',n,beta,name,w,chi,rows[-1]['cp_error'],flush=True)


def held_out(outdir):
    rows=[]
    n=8;chi=24
    for beta,channel,h in [(.8,'Y_damping',.1),('ground','Z',.1),(.8,'Z',.05),(.8,'Z',.025)]:
        initial=ising_amplitude(n,beta)
        ops=site_kraus(h,channel)
        assert np.linalg.norm(sum(k.conj().T@k for k in ops)-np.eye(2))<1e-13
        for w in ([3,5] if channel=='Y_damping' or beta=='ground' else [3]):
            state=cs.from_vector(base.to_interleaved(initial,n),n,chi=chi)
            norm=state.norm();state.tensors[state.center]/=norm
            ref=initial
            for step in range(n*round(.2/h)):
                j=step%n
                info=apply_buffer(state,j,h,w,1,channel)
                ref=apply_channel_root(ref,n,j,ops)
                amp=base.to_matrix(state.dense(),n)
                rows.append(dict(beta=beta,n=n,cap=chi,channel=channel,step=step+1,width=w,halo=1,h=h,
                                 **info,**branch.diagnostics(amp,ref@ref.conj().T,ref)))
            cs.write_csv(outdir/'held_out.csv',rows)
            print('held out',beta,channel,h,w,rows[-1]['cp_error'],flush=True)


def physical_update(rho,n,j,operators):
    result=np.zeros_like(rho,dtype=np.result_type(rho,*operators))
    for k in operators:
        full=np.kron(np.eye(2**j),np.kron(k,np.eye(2**(n-j-1))))
        result+=full@rho@full.conj().T
    return result


def independent_marginals(outdir):
    """For onsite noise, the small physical marginals evolve autonomously."""
    rows=[]
    n=8;h=.1;chi=24;halo=1
    for beta,channel in [(.8,'Z'),(1.6,'Z'),('ground','Z'),(.8,'Y_damping')]:
        x=ising_amplitude(n,beta);ops=site_kraus(h,channel)
        for w in [3,5]:
            state=cs.from_vector(base.to_interleaved(x,n),n,chi=chi)
            norm=state.norm();state.tensors[state.center]/=norm
            initial_error=np.linalg.norm(state.dense()-base.to_interleaved(x,n))
            cumul=initial_error
            # Initialize local data once from the prepared physical state. From
            # here on the gate builder receives only a cached marginal and Kraus data.
            rho=x@x.T
            regions={}
            for j in range(n):
                start=start_of_window(j,w,n);key=(max(0,start-halo),min(n,start+w+halo))
                if key not in regions: regions[key]=base.marginal(rho,n,range(*key))
            reference=x
            for step in range(2*n):
                j=step%n;start=start_of_window(j,w,n)
                left,right=max(0,start-halo),min(n,start+w+halo)
                info=apply_buffer(state,j,h,w,halo,channel,regions[(left,right)])
                for (lo,hi),marginal in regions.items():
                    if lo<=j<hi: regions[(lo,hi)]=physical_update(marginal,hi-lo,j-lo,ops)
                # Global references and CMIs below are diagnostics only.
                sigma=physical_update(rho,n,j,ops)
                reference=apply_channel_root(reference,n,j,ops)
                assert np.linalg.norm(reference@reference.conj().T-sigma)<3e-13
                assert np.linalg.norm(regions[(left,right)]-base.marginal(sigma,n,range(left,right)))<3e-13
                core=set(range(start,start+w));around=set(range(left,right))-core
                cmi=.5*(cmi_nats(rho,n,core,around)+cmi_nats(sigma,n,core,around))
                # Each SVD split and the final normalization cost at most 2 sqrt(weight)
                # per split. Cauchy-Schwarz bounds the sum given aggregated split weights.
                trunc_bound=2*np.sqrt((w-1)*info['discarded'])
                cumul+=info['local_fit']+2*np.sqrt(cmi)+trunc_bound
                y=base.to_matrix(state.dense(),n)
                row=dict(n=n,beta=beta,channel=channel,step=step+1,h=h,width=w,halo=halo,cap=chi,
                         cmi_flag=cmi,cumulative_bound=cumul,initial_error=initial_error,
                         **info,**branch.diagnostics(y,sigma,reference))
                assert row['cp_error']<=cumul+1e-9
                rows.append(row);rho=sigma
            cs.write_csv(outdir/'independent_marginals.csv',rows)
            print('cached',beta,channel,w,rows[-1]['cp_error'],cumul,flush=True)


def larger_chain(outdir):
    n=10;beta=.8;h=.1;chi=24
    initial=ising_amplitude(n,beta)
    rho=initial@initial.T
    differences=np.bitwise_xor.outer(np.arange(2**n),np.arange(2**n))
    distances=np.array([i.bit_count() for i in range(2**n)])[differences]
    targets=[rho*np.exp(-k*h*distances) for k in (1,2)]
    roots=[small_positive_root(sigma) for sigma in targets]
    rows=[]
    for w,halo in [(3,1),(3,2),(5,1)]:
        state=cs.from_vector(base.to_interleaved(initial,n),n,chi=chi)
        norm=state.norm();state.tensors[state.center]/=norm
        cache={}
        for j in range(n):
            start=start_of_window(j,w,n)
            key=(max(0,start-halo),min(n,start+w+halo))
            if key not in cache: cache[key]=base.marginal(rho,n,range(*key))
        for step in range(2*n):
            j=step%n;start=start_of_window(j,w,n)
            key=(max(0,start-halo),min(n,start+w+halo))
            apply_buffer(state,j,h,w,halo,'Z',cache[key])
            for (lo,hi),m in cache.items():
                if lo<=j<hi:cache[(lo,hi)]=cs.noise(m,hi-lo,j-lo,h)
            if j==n-1:
                idx=step//n;y=base.to_matrix(state.dense(),n)
                rows.append(dict(n=n,beta=beta,cap=chi,width=w,halo=halo,step=step+1,h=h,
                                 **branch.diagnostics(y,targets[idx],roots[idx])))
        cs.write_csv(outdir/'larger_chain.csv',rows)
        print('size',n,w,halo,rows[-1]['cp_error'],flush=True)


def marginal_polar_step(state,j,operators,start,width,physical):
    """Use the local physical root in the polar alignment (Petz Kraus factors)."""
    s=small_positive_root(physical)
    left=np.eye(2**(j-start));right=np.eye(2**(start+width-j-1))
    extended=[np.kron(left,np.kron(k,right)) for k in operators]
    matrix=np.concatenate([k@s for k in extended],axis=1)
    u,_,vh=np.linalg.svd(matrix,full_matrices=False)
    isometry=vh.conj().T@u.conj().T
    ws=np.vsplit(isometry,len(operators))
    gate=sum(np.kron(k,w.T) for k,w in zip(extended,ws))
    perm=base.to_interleaved(np.arange(4**width).reshape(s.shape),width)
    pos=len(state.discarded_weights)
    state.apply_gate(start,width,gate[np.ix_(perm,perm)])
    discarded=sum(state.discarded_weights[pos:])
    norm=state.norm();state.tensors[state.center]/=norm
    return max(0.,1-norm**2-discarded),discarded


def petz_comparison(outdir):
    rows=[]
    n=8;h=.1;chi=24
    for beta,channel in [(.8,'Z'),(1.6,'Z'),('ground','Z'),(.8,'Y_damping')]:
        x=ising_amplitude(n,beta);ops=site_kraus(h,channel)
        for w in [3,5]:
            state=cs.from_vector(base.to_interleaved(x,n),n,chi=chi)
            norm=state.norm();state.tensors[state.center]/=norm
            cache={start:base.marginal(x@x.T,n,range(start,start+w)) for start in range(n-w+1)}
            ref=x
            for step in range(2*n):
                j=step%n;start=start_of_window(j,w,n)
                leak,trunc=marginal_polar_step(state,j,ops,start,w,cache[start])
                for lo,m in cache.items():
                    if lo<=j<lo+w:cache[lo]=physical_update(m,w,j-lo,ops)
                ref=apply_channel_root(ref,n,j,ops);amp=base.to_matrix(state.dense(),n)
                rows.append(dict(n=n,beta=beta,channel=channel,width=w,cap=chi,step=step+1,leakage=leak,
                                 discarded=trunc,**branch.diagnostics(amp,ref@ref.conj().T,ref)))
            cs.write_csv(outdir/'marginal_polar.csv',rows)
            print('marginal polar',beta,channel,w,rows[-1]['cp_error'],flush=True)


def budget_tests(outdir):
    rows=[]
    for name,x in [('cluster pure',base.analytic_root(6,0).real),
                   ('Ising beta 0.8',ising_amplitude(6,.8)),('Ising beta 1.6',ising_amplitude(6,1.6))]:
        for h in ([.1,.01,.001] if name=='cluster pure' else [.1,.025]):
            sigma=cs.noise(x@x.T,6,2,h);root=dephased_root(x,6,2,h)
            for eps in ([1e-7] if name=='cluster pure' else [1e-3,1e-4]):
                y,info=direct_fit(x,sigma,1,3,eps,maxiter=340)
                angle=np.arccos(np.clip(np.vdot(x,y).real,-1,1))
                rows.append(dict(case=name,h=h,width=3,state_angle=angle,
                                 **info,**branch.diagnostics(y,sigma,root)))
        cs.write_csv(outdir/'budgets.csv',rows)
        print('budgets',name,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir',type=Path,default=Path(__file__).parent/'ising_results')
    parser.add_argument('--part',choices=['direct','windows','buffered','evolution','budgets','held_out','cached','size','petz','all'],default='all')
    args=parser.parse_args();args.outdir.mkdir(exist_ok=True,parents=True)
    if args.part in ('direct','all'): isolated(args.outdir)
    if args.part in ('windows','all'): scan_windows(args.outdir)
    if args.part in ('buffered','all'): buffer_tests(args.outdir)
    if args.part in ('evolution','all'): evolve(args.outdir)
    if args.part in ('budgets','all'): budget_tests(args.outdir)
    if args.part in ('held_out','all'): held_out(args.outdir)
    if args.part in ('cached','all'): independent_marginals(args.outdir)
    if args.part in ('size','all'): larger_chain(args.outdir)
    if args.part in ('petz','all'): petz_comparison(args.outdir)


if __name__=='__main__':main()
