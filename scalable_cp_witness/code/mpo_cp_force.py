"""Matrix-free MPO canonical-root force: local contractions and compressed PCG.

No dense whole-chain matrices are used by the force, preparation, or retraction.
The assumed positive branch is monitored, not certified by Hermiticity alone.
"""
import tdvp_core as tn
import copy
import time
import numpy as np

I=np.eye(2,dtype=complex);X=np.array([[0,1],[1,0]],complex);Z=np.diag([1.,-1.]).astype(complex)
SM=np.array([[0,1],[0,0]],complex)


def adjoint(a):
    out=copy.deepcopy(a)
    out.tensors=[x[:,[0,2,1,3],:].conj() for x in out.tensors]
    return out


def hermitize(a,chi,normalize=False):
    return tn.compress(tn.add(a,adjoint(a),alpha=.5,beta=.5),chi,normalize=normalize)[0]


def multiply(a,b):
    out=tn.MPS(a.n,max_bond=a.max_bond*b.max_bond);out.tensors=[]
    for x,y in zip(a.tensors,b.tensors):
        x=x.reshape(x.shape[0],2,2,x.shape[2]);y=y.reshape(y.shape[0],2,2,y.shape[2])
        z=np.einsum('aikb,ckjd->acijbd',x,y,optimize=True)
        out.tensors.append(z.reshape(x.shape[0]*y.shape[0],4,x.shape[-1]*y.shape[-1]))
    out.center=out.n-1;return out


def trace(a):
    env=np.ones(1,complex)
    for x in a.tensors:env=env@(x[:,0,:]+x[:,3,:])
    return env.item()


def multiply_zip(a,b,chi,oversample=2):
    """Zip-up product with an enlarged temporary bond, then canonical rounding.

    The early zip truncation is approximate. The Sylvester residual is checked
    separately against uncompressed products, so its error is not hidden.
    """
    out=tn.MPS(a.n,max_bond=oversample*chi);out.tensors=[];carry=np.ones((1,1,1),complex)
    for j,(x,y) in enumerate(zip(a.tensors,b.tensors)):
        x=x.reshape(x.shape[0],2,2,x.shape[-1]);y=y.reshape(y.shape[0],2,2,y.shape[-1])
        block=np.einsum('lac,apkb,ckqd->lpqbd',carry,x,y,optimize=True)
        if j==a.n-1:out.tensors.append(block.reshape(block.shape[0],4,1));break
        u,s,vh=np.linalg.svd(block.reshape(block.shape[0]*4,-1),full_matrices=False)
        rank=max(1,min(oversample*chi,int(np.sum(s>1e-13*s[0]))))
        out.tensors.append(u[:,:rank].reshape(block.shape[0],4,rank))
        carry=(s[:rank,None]*vh[:rank]).reshape(rank,x.shape[-1],y.shape[-1])
    out.center=a.n-1;return tn.compress(out,chi)[0]


def product_root(n,polarization=.2,axis=X):
    rho=(I+polarization*axis)/2;values,vectors=np.linalg.eigh(rho)
    root=(vectors*np.sqrt(values))@vectors.conj().T
    out=tn.MPS(n,max_bond=1);out.tensors=[root.reshape(1,4,1).copy() for _ in range(n)]
    return out


def hamiltonian(n,jzz=.7,hx=.43,hz=.19):
    out=tn.MPS(n,max_bond=3);out.tensors=[]
    for j in range(n):
        w=np.zeros((3,2,2,3),complex);w[0,:,:,0]=I;w[2,:,:,2]=I
        w[0,:,:,1]=jzz*Z;w[1,:,:,2]=Z;w[0,:,:,2]=hx*X+hz*Z
        if j==0:w=w[0:1]
        if j==n-1:w=w[:,:,:,2:3]
        out.tensors.append(w.reshape(w.shape[0],4,w.shape[-1]))
    out.center=n-1;return out


def onsite_dissipation(rho,rates,kind='loss'):
    out=tn.MPS(rho.n,max_bond=2*rho.max_bond);out.tensors=[]
    for j,(a,rate) in enumerate(zip(rho.tensors,rates)):
        jump=SM if kind=='loss' else Z
        number=jump.conj().T@jump
        local=rate*(np.kron(jump,jump.conj())-.5*np.kron(number,I)-.5*np.kron(I,number.T))
        acted=np.einsum('st,atr->asr',local,a,optimize=True)
        dl,_,dr=a.shape;w=np.zeros((2*dl,4,2*dr),complex)
        w[:dl,:,:dr]=a;w[:dl,:,dr:]=acted;w[dl:,:,dr:]=a
        if j==0:w=w[:dl]
        if j==rho.n-1:w=w[:,:,dr:]
        out.tensors.append(w)
    out.center=out.n-1;return out


def one_site_marginals(amplitude):
    n=amplitude.n;left=[np.ones((1,1),complex)];right=[None]*(n+1);right[n]=np.ones((1,1),complex)
    for a in amplitude.tensors:left.append(np.einsum('ab,asr,bsu->ru',left[-1],a,a.conj(),optimize=True))
    for j in range(n-1,-1,-1):
        a=amplitude.tensors[j];right[j]=np.einsum('asr,bsu,ru->ab',a,a.conj(),right[j+1],optimize=True)
    out=[]
    for j,a in enumerate(amplitude.tensors):
        a=a.reshape(a.shape[0],2,2,a.shape[-1])
        rho=np.einsum('ab,apkr,bqku,ru->pq',left[j],a,a.conj(),right[j+1],optimize=True)
        rho=(rho+rho.conj().T)/2;rho/=np.trace(rho);out.append(rho)
    return out


def product_preconditioner(amplitude,eigenvalue_floor=1e-12):
    local=[]
    for rho in one_site_marginals(amplitude):
        # A 1e-5 floor destroyed the near-pure, square-root-time force even
        # on an exactly factorized input. Keep the actual local spectrum down
        # to numerical resolution; the floor remains a regularization.
        values,vectors=np.linalg.eigh(rho);values=np.maximum(values,eigenvalue_floor)
        inverse=values**(-.25);inverse/=max(inverse)
        local.append((vectors*inverse)@vectors.conj().T)
    return local


def apply_preconditioner(a,local,chi):
    out=copy.deepcopy(a)
    for j,(x,w) in enumerate(zip(a.tensors,local)):
        x=x.reshape(x.shape[0],2,2,x.shape[-1])
        out.tensors[j]=np.einsum('pi,aijb,jq->apqb',w,x,w,optimize=True).reshape(x.shape[0],4,x.shape[-1])
    return tn.compress(out,chi,normalize=False)[0]


def sylvester_apply(x,f,chi):
    # Canonical rounding is deliberate: the faster zip-up approximation broke
    # compressed-CG convergence in a correlated 16-site test.
    return tn.compress(tn.add(multiply(x,f),multiply(f,x)),chi,normalize=False)[0]


def pcg_force(root,rhs,chi=16,tolerance=1e-6,maxiter=100,precondition=True,initial=None,eigenvalue_floor=1e-12,restart_every=12):
    mean=float(trace(root).real)*2.**(-root.n)
    if mean<=0:raise RuntimeError('Nonpositive trace: canonical branch lost.')
    x=tn.scale(root,1/mean);b=tn.scale(rhs,1/mean)
    norm_b=tn.norm(b)
    if norm_b<1e-15:return tn.scale(root,0.),dict(iterations=0,residual=0.,true_residual=0.,curvature_failure=False)
    local=product_preconditioner(root,eigenvalue_floor) if precondition else None
    pre=lambda value:apply_preconditioner(value,local,chi) if local else copy.deepcopy(value)
    solution=tn.scale(root,0.) if initial is None else copy.deepcopy(initial)
    r=tn.compress(tn.add(b,sylvester_apply(x,solution,chi),beta=-1),chi)[0]
    z=pre(r);p=copy.deepcopy(z);rz=tn.overlap(r,z).real;failure=False;history=[]
    initial_residual=tn.norm(r)/norm_b
    # An already solved warm start must not be interpreted as zero curvature.
    for iteration in range(0 if initial_residual<tolerance else maxiter):
        ap=sylvester_apply(x,p,chi);curvature=tn.overlap(p,ap).real
        if curvature<=0:failure=True;break
        alpha=rz/curvature
        solution=tn.compress(tn.add(solution,p,beta=alpha),chi)[0]
        r=tn.compress(tn.add(r,ap,beta=-alpha),chi)[0]
        residual=tn.norm(r)/norm_b;history.append(residual)
        if residual<tolerance:break
        z=pre(r);next_rz=tn.overlap(r,z).real
        p=tn.compress(tn.add(z,p,beta=next_rz/rz),chi)[0];rz=next_rz
        if restart_every and (iteration+1)%restart_every==0:
            r=tn.compress(tn.add(b,sylvester_apply(x,solution,chi),beta=-1),chi)[0]
            z=pre(r);p=copy.deepcopy(z);rz=tn.overlap(r,z).real
    raw=tn.add(tn.add(multiply(x,solution),multiply(solution,x)),b,beta=-1.)
    actual=tn.norm(raw)/norm_b
    return solution,dict(iterations=len(history),residual=history[-1] if history else initial_residual,true_residual=actual,
                         curvature_failure=failure,converged=actual<=tolerance,history=history,mean_root_eigenvalue=mean)


class MPOForce:
    def __init__(self,n,rates,chi=16,tolerance=1e-6,maxiter=100,precondition=True,eigenvalue_floor=1e-12):
        self.h=hamiltonian(n);self.rates=np.asarray(rates);self.chi=chi;self.tolerance=tolerance
        self.maxiter=maxiter;self.precondition=precondition;self.records=[];self.previous_noise=None
        self.eigenvalue_floor=eigenvalue_floor

    def __call__(self,t,state):
        started=time.perf_counter();root=hermitize(state,state.max_bond)
        rho=tn.compress(multiply(root,root),self.chi)[0]
        noise=tn.compress(onsite_dissipation(rho,self.rates),self.chi)[0]
        dissipative,info=pcg_force(root,noise,self.chi,self.tolerance,self.maxiter,self.precondition,self.previous_noise,self.eigenvalue_floor)
        if info['curvature_failure'] or info['true_residual']>5e-3:
            raise RuntimeError(f"Unreliable MPO force: residual={info['true_residual']:.3g}, curvature_failure={info['curvature_failure']}")
        self.previous_noise=copy.deepcopy(dissipative)
        coherent=tn.add(multiply(self.h,root),multiply(root,self.h),alpha=-1j,beta=1j)
        force=hermitize(tn.add(coherent,dissipative),self.chi)
        info.update(seconds=time.perf_counter()-started,force_norm=tn.norm(force),time=t)
        self.records.append(info);return force,info


def local_trace_gauge(state,chi,threshold=1e-9):
    """Ancilla two-site trace maximization; not a global positivity certificate."""
    out=copy.deepcopy(state);n=out.n;right=[None]*(n+1);right[n]=np.ones(1,complex)
    for j in range(n-1,-1,-1):right[j]=(out.tensors[j][:,0,:]+out.tensors[j][:,3,:])@right[j+1]
    left=np.ones(1,complex);gains=[];unitaries=[];applied=0
    for j in range(n-1):
        block=np.tensordot(out.tensors[j],out.tensors[j+1],axes=(-1,0))
        small=np.einsum('a,astb,b->st',left,block,right[j+2],optimize=True)
        c=small.reshape(2,2,2,2).transpose(0,2,1,3).reshape(4,4)
        norm_c=np.linalg.norm(c);c=c/max(norm_c,1e-300)
        u,s,vh=np.linalg.svd(c);rotation=vh.conj().T@u.conj().T
        gain=max(0.,float(sum(s)-np.trace(c).real));angle=np.linalg.norm(rotation-np.eye(4))
        gains.append(gain);unitaries.append(angle)
        # Applying a gate moves the MPS gauge and invalidates cached trace
        # environments. Rebuild and restart after an actual correction.
        if angle>threshold and gain>1e-13:
            raw=np.kron(np.eye(4),rotation.T)
            order=np.arange(16).reshape(2,2,2,2).transpose(0,2,1,3).reshape(-1)
            out.max_bond=chi;out.apply_gate(j,2,raw[np.ix_(order,order)]);applied+=1
            out,_=tn.compress(out,chi,normalize=True)
            # One correction per call keeps the sweep's cached environments valid.
            break
        left=left@(out.tensors[j][:,0,:]+out.tensors[j][:,3,:])
    return out,dict(max_local_trace_gap=max(gains,default=0.),max_ancilla_rotation=max(unitaries,default=0.),gauge_gates=applied)
