"""Automatic support-leakage detection and positive rank-opening steps.

Dense reference backend: arbitrary finite-dimensional H and Lindblad jumps.
The MPS backend lives in rank_opening_mps.py. Numerical support thresholds
are reported; they are not claims about exact mathematical rank.
"""
import tdvp_core as tn
import numpy as np
from scipy.linalg import expm
from dataclasses import dataclass


def herm(a):return (a+a.conj().T)/2


@dataclass
class Model:
    h: np.ndarray
    jumps: list

    def rhs(self,rho):
        out=-1j*(self.h@rho-rho@self.h)
        for l in self.jumps:
            s=l.conj().T@l;out+=l@rho@l.conj().T-.5*(s@rho+rho@s)
        return herm(out)


def detect(root,model,step,root_threshold=1e-12,rate_threshold=1e-12):
    x,u=np.linalg.eigh(herm(root));maximum=max(float(max(x)),1e-300)
    if min(x)<-1e-10*maximum:raise ValueError('Amplitude is not on the positive branch.')
    x=np.maximum(x,0);keep=x>root_threshold*maximum;null=u[:,~keep]
    rho=root@root;g=model.rhs(rho)
    # Construct jump innovations directly, avoiding cancellation against the
    # Hamiltonian and anticommutator terms in the kernel block.
    innovations=[null.conj().T@l@root for l in model.jumps]
    b=sum((v@v.conj().T for v in innovations),np.zeros((len(x)-sum(keep),)*2,complex))
    leakage=float(np.trace(b).real);jump_rate=sum(float(np.linalg.norm(l@root)**2) for l in model.jumps)
    first_order=leakage>rate_threshold*max(1.,jump_rate)
    diag=np.real(np.diag(u.conj().T@g@u))
    exposure=float(np.max(step*np.abs(diag[keep])/np.maximum(x[keep]**2,1e-300),initial=0.))
    info=dict(rank=int(sum(keep)),dimension=len(x),discarded_support_mass=float(sum(x[~keep]**2)),
              leakage_rate=leakage,opening_cp_scale=np.sqrt(step*leakage),
              near_boundary_exposure=exposure,first_order_opening=bool(first_order))
    return info


def positive_channel_step(root,model,step):
    """A CPTP finite step, retaining root-scale singular values.

    U(h/2) [ K0 rho K0 + h sum L rho L^dag ] U^dag(h/2),
    K0=sqrt(I-h sum L^dag L). No density eigensolve is used.
    This has first-order density accuracy, not a Strang-accurate dissipator.
    """
    d=len(root);number=sum((l.conj().T@l for l in model.jumps),np.zeros((d,d),complex))
    values,vectors=np.linalg.eigh(herm(number))
    if step*max(values,default=0)>1-1e-12:raise ValueError('CPTP step exceeds its positivity limit.')
    k0=(vectors*np.sqrt(np.maximum(1-step*values,0)))@vectors.conj().T
    u=expm(-1j*step/2*model.h);mid=u@root
    factor=np.column_stack([u@k0@mid]+[np.sqrt(step)*u@l@mid for l in model.jumps])
    basis,s,_=np.linalg.svd(factor,full_matrices=False)
    result=(basis*s)@basis.conj().T
    return herm(result)


def finite_probe(root,model,step,rank):
    rate_bound=sum(np.linalg.norm(l,2)**2 for l in model.jumps)
    pieces=max(1,int(np.ceil(2*step*rate_bound)))
    probe=root
    for _ in range(pieces):probe=positive_channel_step(probe,model,step/pieces)
    s=np.linalg.eigvalsh(probe)[::-1]
    return float(np.linalg.norm(s[rank:])),probe


def classify(root,model,step,cp_budget):
    info=detect(root,model,step)
    if info['first_order_opening']:mode='first_order_opening'
    elif info['rank']<len(root):
        tail,_=finite_probe(root,model,step,info['rank']);info['finite_probe_tail']=tail
        mode='delayed_opening' if tail>cp_budget*.1 else 'smooth_within_budget'
    else:mode='near_boundary' if info['near_boundary_exposure']>.25 else 'smooth'
    info['mode']=mode;return info


def positive_bridge(root,model,max_step,cp_budget,max_trials=40):
    """Accept two positive half steps using a conservative difference estimate.

    The difference is an error estimator, not a rigorous bound; no division
    by three is justified across a rank-changing boundary.
    """
    step=max_step
    for trial in range(max_trials):
        try:
            coarse=positive_channel_step(root,model,step)
            half=positive_channel_step(root,model,step/2)
            fine=positive_channel_step(half,model,step/2)
        except ValueError:
            step/=2;continue
        estimate=float(np.linalg.norm(fine-coarse))
        if estimate<=cp_budget:
            return fine,step,dict(step_estimate=estimate,halvings=trial,
                                  trace_error=abs(np.linalg.norm(fine)**2-1),
                                  min_root_eigenvalue=float(np.linalg.eigvalsh(fine)[0]))
        step/=2
    raise RuntimeError('Could not resolve a positive bridge at the requested budget.')


def smooth_force(root,model):
    x,u=np.linalg.eigh(herm(root));g=u.conj().T@model.rhs(root@root)@u
    den=x[:,None]+x[None,:];keep=den>1e-14*max(x)
    if np.linalg.norm(g[~keep])>1e-9*max(np.linalg.norm(g),1e-30):
        raise RuntimeError('Unresolved kernel forcing; use the positive bridge.')
    f=np.zeros_like(g);f[keep]=g[keep]/den[keep]
    return herm(u@f@u.conj().T)


def retract(a):
    # Reference implementation: exact polar branch, full spatial bond.
    u,s,_=np.linalg.svd(a,full_matrices=False)
    return (u*(s/np.linalg.norm(s)))@u.conj().T


def smooth_midpoint(root,model,time,step):
    # u=sqrt(t), using a full tangent space for this dense control experiment.
    u=np.sqrt(time);du=np.sqrt(time+step)-u
    first=2*u*smooth_force(root,model)
    mid=retract(root+du/2*first)
    second=2*(u+du/2)*smooth_force(mid,model)
    return retract(root+du*second)


def automatic_step(root,model,time,proposed_step,cp_budget):
    status=classify(root,model,proposed_step,cp_budget)
    if status['mode'] in ('first_order_opening','delayed_opening') or (status['mode']=='near_boundary' and time==0):
        out,step,info=positive_bridge(root,model,proposed_step,cp_budget)
        return out,step,dict(**status,**info,method='positive_bridge')
    step=proposed_step
    if status['mode']=='near_boundary':
        # A small but nonzero population is stiffness, not a new rank event.
        # Resolve its relative change before attempting the ordinary force.
        step*=min(1.,.2/status['near_boundary_exposure'])
        status['boundary_limited_step']=step
    for trial in range(40):
        try:
            coarse=smooth_midpoint(root,model,time,step)
            half=smooth_midpoint(root,model,time,step/2)
            fine=smooth_midpoint(half,model,time+step/2,step/2)
        except RuntimeError:
            out,used,info=positive_bridge(root,model,step,cp_budget)
            return out,used,dict(**status,**info,method='positive_bridge',fallback='stage_kernel_forcing')
        estimate=float(np.linalg.norm(fine-coarse))/3
        if estimate<=cp_budget:return fine,step,dict(**status,step_estimate=estimate,halvings=trial,method='sqrt_time_midpoint')
        step/=2
    raise RuntimeError('Smooth step did not converge.')


def evolve(root,model,elapsed,tolerance=1e-4,max_step=.01):
    time=0.;history=[];state=root.copy();spent=0.
    for iteration in range(2000):
        if time>=elapsed-1e-14:return state,history
        proposed=min(max_step,elapsed-time)
        # Reserve a fixed initial budget, then allocate by physical time.
        budget=tolerance*.05 if time==0 else tolerance*.5*proposed/elapsed
        state,used,info=automatic_step(state,model,time,proposed,budget)
        time+=used;spent+=info['step_estimate']
        history.append(dict(time=time,step=used,estimated_sum=spent,**info))
        if info['halvings']>0:max_step=min(max_step,used*2)
        elif info['step_estimate']<budget*.1:max_step=min(.02,max_step*1.5)
    raise RuntimeError('Adaptive loop exceeded its trial budget.')
