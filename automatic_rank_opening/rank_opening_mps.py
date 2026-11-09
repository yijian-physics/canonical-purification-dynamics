"""MPO-only automatic pure-state opening through positive Kraus factors.

This is a controlled finite startup, not a general full-rank MPO root solver.
The small matrix dimension is the number of jump operators plus one.
"""
import tdvp_core as tn
import mpo_cp_force as mpo
import copy
import numpy as np
from scipy.linalg import expm


def extract_pure_ket(root,tolerance=1e-9):
    defect=tn.distance(mpo.multiply(root,root),root)
    if defect>tolerance:raise ValueError('A general mixed MPO needs a support representation; no rank guess from local marginals.')
    ket=copy.deepcopy(root);ket.center=ket.n-1;ket.move_center(0)
    for j in range(ket.n):
        ket.move_center(j);a=ket.tensors[j]
        weight=[np.linalg.norm(a[:,[b,2+b],:])**2 for b in (0,1)];b=int(np.argmax(weight))
        z=np.zeros_like(a);z[:,[0,2],:]=a[:,[b,2+b],:]/np.sqrt(weight[b]);ket.tensors[j]=z
    ket,_=tn.compress(ket,max(root.max_bond,2),normalize=True)
    reconstructed=mpo.multiply(ket,mpo.adjoint(ket))
    return ket,dict(idempotency_defect=defect,pure_reconstruction_error=tn.distance(root,reconstructed))


def act(ket,j,op):
    out=copy.deepcopy(ket);a=out.tensors[j]
    out.tensors[j]=np.einsum('st,atr->asr',np.kron(op,mpo.I),a,optimize=True)
    return out


def detect_pure(root,rates):
    ket,info=extract_pure_ket(root);leaks=[]
    for j,rate in enumerate(rates):
        jump=act(ket,j,np.sqrt(rate)*mpo.SM)
        mean=tn.overlap(ket,jump)
        residual=tn.add(jump,ket,beta=-mean)
        leaks.append(tn.norm(residual)**2)
    info.update(leakage_rate=float(sum(leaks)),first_order_opening=bool(sum(leaks)>1e-12),site_leakages=leaks)
    return ket,info


def no_jump(ket,time,rates,chi,jzz=.7,hx=.43,hz=.19):
    """Lie product of contraction semigroups; no normalization."""
    out=copy.deepcopy(ket);out.max_bond=chi;out.discarded_weights=[]
    for j,rate in enumerate(rates):
        a=-1j*(hx*mpo.X+hz*mpo.Z)-.5*rate*(mpo.SM.conj().T@mpo.SM)
        out.apply_gate(j,1,np.kron(expm(time*a),mpo.I))
    u=expm(-1j*time*jzz*np.kron(mpo.Z,mpo.Z));gate=np.kron(u,np.eye(4))
    order=np.arange(16).reshape([2]*4).transpose(0,2,1,3).reshape(-1)
    for j in range(ket.n-1):out.apply_gate(j,2,gate[np.ix_(order,order)])
    return out,float(sum(np.sqrt(out.discarded_weights)))


def sum_local_action(ket,coefficients,rates):
    out=tn.MPS(ket.n,max_bond=2*ket.max_bond);out.tensors=[]
    for j,(a,c,rate) in enumerate(zip(ket.tensors,coefficients,rates)):
        acted=np.einsum('st,atr->asr',np.kron(c*np.sqrt(rate)*mpo.SM,mpo.I),a,optimize=True)
        dl,_,dr=a.shape;w=np.zeros((2*dl,4,2*dr),complex)
        w[:dl,:,:dr]=a;w[:dl,:,dr:]=acted;w[dl:,:,dr:]=a
        if j==0:w=w[:dl]
        if j==ket.n-1:w=w[:,:,dr:]
        out.tensors.append(w)
    out.center=out.n-1;return out


def channel_bound_constants(rates,jzz=.7,hx=.43,hz=.19):
    # Bounds follow by summing norms of overlapping local commutators.
    n=len(rates);number=mpo.SM.conj().T@mpo.SM
    a=[-1j*(hx*mpo.X+hz*mpo.Z)-.5*g*number for g in rates]
    cab=0.;comm_squares=0.
    for j in range(n-1):
        cab+=abs(jzz)*(np.linalg.norm(a[j]@mpo.Z-mpo.Z@a[j],2)+np.linalg.norm(a[j+1]@mpo.Z-mpo.Z@a[j+1],2))
    for j,g in enumerate(rates):
        l=np.sqrt(g)*mpo.SM
        c=np.linalg.norm(l@a[j]-a[j]@l,2)
        c+=abs(jzz)*((j>0)+(j<n-1))*np.linalg.norm(l@mpo.Z-mpo.Z@l,2)
        comm_squares+=c*c
    return float(sum(rates)),float(cab),float(comm_squares)


def analytic_purification_bound(step,constants):
    lam,cab,comm=constants;z=lam*step
    # Integral bound is stable at z~0: P(Poisson(z)>=2) <= z^2/2.
    omitted=min(1.,z*z/2)
    trajectory=np.sqrt(omitted+step**3*comm/12)
    trotter=np.sqrt((step*step*cab/2)**2+step*lam*(step*step*cab/4)**2)
    return float(trajectory+trotter)


def startup(root,rates,cp_budget=1e-3,max_step=.01,chi=32,ket_chi=16):
    ket,info=detect_pure(root,rates);constants=channel_bound_constants(rates)
    step=max_step
    while np.sqrt(2)*analytic_purification_bound(step,constants)>cp_budget*.5:step/=2
    v0,e0=no_jump(ket,step,rates,ket_chi)
    mid,emid=no_jump(ket,step/2,rates,ket_chi)
    factors=[v0];factor_errors=[e0];jump_action_errors=[]
    for j,rate in enumerate(rates):
        v,e=no_jump(act(mid,j,np.sqrt(rate)*mpo.SM),step/2,rates,ket_chi)
        factors.append(tn.scale(v,np.sqrt(step)))
        factor_errors.append(np.sqrt(step)*(np.sqrt(rate)*emid+e))
        jump_action_errors.append(np.sqrt(step)*e)
    a=tn.norm(v0);b0=tn.scale(v0,1/a);means=np.array([tn.overlap(b0,v) for v in factors[1:]])
    residuals=[tn.add(v,b0,beta=-c) for v,c in zip(factors[1:],means)]
    m=len(rates);gram=np.zeros((m,m),complex)
    for j in range(m):
        for k in range(j,m):gram[j,k]=tn.overlap(residuals[j],residuals[k]);gram[k,j]=gram[j,k].conjugate()
    values,vectors=np.linalg.eigh((gram+gram.conj().T)/2)
    # Deflation keeps the near-one no-jump population out of this small Gram
    # eigensolve. Cutoff is tied to a root-error budget, not a density floor.
    cutoff=(cp_budget*.005)**2/max(m,1)
    keep=values>max(cutoff,1e-13*max(max(values),1e-300));lost=float(sum(np.maximum(values[~keep],0)))
    mu=values[keep];u=vectors[:,keep];r=len(mu)
    coefficients=np.zeros((m+1,r+1),complex);coefficients[0,0]=1/a
    if r:
        coefficients[1:,1:]=u/np.sqrt(mu)
        coefficients[0,1:]=-(means@u)/(a*np.sqrt(mu))
    coords=np.zeros((r+1,m+1),complex);coords[0,0]=a;coords[0,1:]=means
    if r:coords[1:,1:]=np.sqrt(mu)[:,None]*u.conj().T
    rotations,singular,_=np.linalg.svd(coords,full_matrices=False)
    mode_coefficients=coefficients@rotations
    result=None;rounding=0.;mode_rounding=0.
    for k,s in enumerate(singular):
        c=mode_coefficients[:,k]
        action=sum_local_action(mid,np.sqrt(step)*c[1:],rates)
        acted,e=no_jump(action,step/2,rates,ket_chi)
        raw=tn.add(v0,acted,alpha=c[0]);mode,compress=tn.compress(raw,ket_chi)
        delta=e+float(np.dot(abs(c[1:]),jump_action_errors))+compress['discarded_bound']
        mode_rounding+=s*delta*(2*tn.norm(mode)+delta)
        term=tn.scale(mpo.multiply(mode,mpo.adjoint(mode)),s)
        summed=term if result is None else tn.add(result,term)
        result,record=tn.compress(summed,chi);rounding+=record['discarded_bound']
    root_norm=tn.norm(result);result=tn.scale(result,1/root_norm)
    factor_norm=float(np.sqrt(a*a+sum(tn.norm(v)**2 for v in factors[1:])))
    fac_error=float(np.linalg.norm(factor_errors));discard_error=np.sqrt(lost)
    analytic=analytic_purification_bound(step,constants)
    cp_channel_bound=np.sqrt(2)*(analytic+fac_error+discard_error+abs(1-np.linalg.norm(singular)))
    cp_channel_bound+=info['pure_reconstruction_error']
    cp_rounding_bound=2*(rounding+mode_rounding)/max(np.linalg.norm(singular),1e-300)
    info.update(step=step,cp_budget=cp_budget,kraus_rank=len(singular),jump_innovation_rank=r,
                omitted_innovation_mass=lost,channel_cp_bound=cp_channel_bound,
                root_compression_bound=cp_rounding_bound,
                total_bound_excluding_roundoff=cp_channel_bound+cp_rounding_bound,
                trace_before_normalization=factor_norm**2,
                hermiticity=tn.distance(result,mpo.adjoint(result)),bond=result.bond_dimension())
    if info['total_bound_excluding_roundoff']>cp_budget:
        raise RuntimeError('Startup exceeds its budget: increase bonds or reduce max_step.')
    return result,info,factors


def rank_opening_step(root,rates,cp_budget=1e-3,max_step=.01,chi=16,ket_chi=8):
    attempts=[]
    for trial in range(10):
        try:
            state,info,factors=startup(root,rates,cp_budget,max_step,chi,ket_chi)
            info.update(retries=trial,requested_chi=chi,ket_chi=ket_chi,next_action='support_audit_required')
            return state,info,factors
        except RuntimeError as exc:
            attempts.append(str(exc))
            if chi<64:chi*=2;ket_chi=min(32,ket_chi*2)
            else:max_step/=2
    raise RuntimeError(f'Automatic startup could not meet its budget: {attempts[-1]}')


def onsite_transitions(bra,ket,operators):
    """All one-site transition matrix elements in one pair of MPS sweeps."""
    n=ket.n;right=[None]*(n+1);right[n]=np.ones((1,1),complex)
    for j in range(n-1,-1,-1):
        right[j]=np.einsum('asr,bsu,ru->ab',bra.tensors[j].conj(),ket.tensors[j],right[j+1],optimize=True)
    left=np.ones((1,1),complex);values=[]
    for j,(a,b,op) in enumerate(zip(bra.tensors,ket.tensors,operators)):
        values.append(np.einsum('ab,asr,st,btu,ru->',left,a.conj(),np.kron(op,mpo.I),b,right[j+1],optimize=True))
        left=np.einsum('ab,asr,bsu->ru',left,a.conj(),b,optimize=True)
    return np.array(values),left.item()


def detect_factor_support(factors,rates):
    """General support leakage from a supplied MPS factor rho=Z Z^dagger.

    No assumption of purity or equal nonzero eigenvalues. The dense matrix
    dimension is the number of factor columns. The roundoff scale is a
    conditioning diagnostic, not an interval-arithmetic error certificate.
    """
    r=len(factors);n=factors[0].n;gram=np.zeros((r,r),complex);transition=np.zeros((n,r,r),complex)
    ops=[np.sqrt(g)*mpo.SM for g in rates]
    gain=0.
    for a in range(r):
        for b in range(r):
            transition[:,a,b],gram[a,b]=onsite_transitions(factors[a],factors[b],ops)
        numbers=[op.conj().T@op for op in ops]
        values,_=onsite_transitions(factors[a],factors[a],numbers);gain+=float(sum(values).real)
    values,vectors=np.linalg.eigh((gram+gram.conj().T)/2)
    keep=values>1e-12*max(values);condition=float(max(values)/min(values[keep]))
    inverse=(vectors[:,keep]/values[keep])@vectors[:,keep].conj().T
    retained=sum(float(np.trace(m.conj().T@inverse@m).real) for m in transition)
    trace=float(np.trace(gram).real);leakage=(gain-retained)/trace
    uncertainty=100*np.finfo(float).eps*condition*max(1.,gain/trace)
    return dict(support_rank=int(sum(keep)),leakage_rate=leakage,
                gram_condition=condition,roundoff_warning_scale=uncertainty,
                further_opening_detected=bool(leakage>uncertainty),
                numerical_support_mass_dropped=float(sum(np.maximum(values[~keep],0)))/trace)


def automatic_startup(root,rates,cp_budget=1e-3,max_step=.01,chi=16,ket_chi=8,check_handoff=True):
    state,info,factors=rank_opening_step(root,rates,cp_budget,max_step,chi,ket_chi)
    if check_handoff:
        audit=detect_factor_support(factors,rates)
        info.update({f'handoff_{k}':v for k,v in audit.items()})
        info['next_action']='rank_aware_continuation_required' if audit['further_opening_detected'] else 'force_validation_required'
    return state,info,factors
