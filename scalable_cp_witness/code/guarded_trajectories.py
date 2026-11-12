"""Run guarded tensor-TDVP trajectories. Exact endpoint roots are used for scoring."""
import argparse
import copy
from pathlib import Path
from time import perf_counter
import guarded_cp_tdvp as g
import tdvp_core as tn
import mpo_cp_force as mpo
import numpy as np
from critical_uniform_tdvp import critical_ground, channel_root, polar_retractor
from guarded_force_trials import root_mpo, table


def summarize(out, name, state, diagnostics, extra):
    history = diagnostics.pop('history')
    for k, record in enumerate(history):
        record['trial'] = k
    if history:
        table(out,name+'_trials.csv',history)
    else:
        (out/(name+'_trials.csv')).unlink(missing_ok=True)
    row = dict(**extra, **diagnostics, rejected=sum(not r['accepted'] for r in history))
    table(out,name+'.csv',[row])
    if state is not None:
        np.savez_compressed(out/(name+'_last_accepted.npz'),
                            **{f'a{j}':a for j,a in enumerate(state.tensors)})
    else:
        (out/(name+'_last_accepted.npz')).unlink(missing_ok=True)
    print(row,flush=True)
    return row


def critical(n,chi,intervals,tolerance,total,out,trial_force_bond=0):
    started = perf_counter()
    psi,_,signs = critical_ground(n)
    pure=np.outer(psi,psi.conj())
    distances=np.sum(signs[:,None,:]!=signs[None,:,:],axis=-1)
    physical=lambda t: pure*np.exp(-t*distances)
    seed=.01 if total>.01 else total/4
    # Retain the pure input exactly. Apply the state bond cap after the bridge.
    initial=root_mpo(pure,n)
    counters={'max_sign_iterations':0,'force_budget_misses':0,'max_stage_repair':0.}
    def force(u,state):
        a=tn.to_matrix(state.dense(),n)
        q=(a+a.conj().T)/2
        if trial_force_bond and u>0:
            q,repair=g.regularized_stage(a,.02*tolerance/np.sqrt(len(a)),.1*tolerance)
            counters['max_stage_repair']=max(counters['max_stage_repair'],repair['repair'])
        rhs= -2*u*distances*(q@q)
        vals=np.linalg.eigvalsh(q)
        if vals[0]<=32*np.finfo(float).eps*len(q)*np.linalg.norm(q):
            g.SpectralAudit(q,-distances*(q@q))
            raise g.Rejected('support_unresolved')
        def bound(candidate):
            return np.linalg.norm(q@candidate+candidate@q-rhs)/(2*vals[0])
        if trial_force_bond:
            # The current spectrum, unrounded RHS, and acceptance audit are
            # dense controls here. Only the sign iterations use MPO products.
            engine=g.MPOAlgebra(trial_force_bond)
            try:
                f,info=g.sylvester_sign(root_mpo(q,n),root_mpo(rhs,n),engine,vals[0],vals[-1],
                    lambda a:bound(tn.to_matrix(a.dense(),n)),1e-4,max_iterations=24)
                return f,dict(**info,provisional=True)
            except g.Rejected as failure:
                if failure.reason not in ('force_tolerance','iteration_limit') or failure.candidate is None:
                    raise
                counters['force_budget_misses']+=1
                return failure.candidate,dict(provisional=True)
        f,info=g.sylvester_sign(q,rhs,g.DenseAlgebra(),vals[0],vals[-1],bound,1e-4)
        mps=root_mpo(f,n)
        info['force_error_bound']+=np.linalg.norm(tn.to_matrix(mps.dense(),n)-f)
        if info['force_error_bound']>1.01e-4:
            raise g.Rejected('force_compression', bound=info['force_error_bound'])
        counters['max_sign_iterations']=max(counters['max_sign_iterations'],info['iterations'])
        return mps,info
    def audit(u,state):
        return g.endpoint_dense(tn.to_matrix(state.dense(),n),physical(u*u))
    def bridge(state,u,v):
        amplitude=tn.to_matrix(state.dense(),n)
        return root_mpo(channel_root(amplitude,signs,v*v-u*u),n,chi)
    state,info=g.integrate_guarded(initial,0.,np.sqrt(total),
                    (np.sqrt(total)-np.sqrt(seed))/intervals,chi,force,polar_retractor(n,chi),
                    audit,tolerance,max_halvings=5,bridge=bridge,bridge_width=np.sqrt(seed))
    initial_status=next((r.get('reason') for r in info['history'] if 'reason' in r),'smooth')
    elapsed=info['coordinate']**2
    if state is None:
        return summarize(out,f'critical_n{n}_chi{chi}_s{intervals}_tol{tolerance:g}',None,info,
                  dict(n=n,chi=chi,target=total,initial_status=initial_status))
    actual=np.linalg.norm(tn.to_matrix(state.dense(),n)-channel_root(psi[:,None],signs,elapsed))
    verified=audit(info['coordinate'],state)
    assert actual<=verified['cp_bound']+2e-11
    suffix=f'_trialbond{trial_force_bond}_T{total:g}' if trial_force_bond else ''
    return summarize(out,f'critical_n{n}_chi{chi}_s{intervals}_tol{tolerance:g}{suffix}',state,info,
              dict(n=n,chi=chi,time=elapsed,target=total,initial_status=initial_status,
                   cp_error=actual,
                   seconds=perf_counter()-started,**counters))


def correlated_input(n,polarization,angle):
    initial=mpo.product_root(n,polarization)
    initial.max_bond=4
    gate=np.diag([1.,1.,1.,np.exp(1j*angle)])
    order=[0,1,4,5,2,3,6,7,8,9,12,13,10,11,14,15]
    lift=np.kron(gate,gate.conj())[np.ix_(order,order)]
    for j in range(n-1):
        initial.apply_gate(j,2,lift)
    rotation=np.array([[1.,1.],[1.,-1.]])/np.sqrt(2)
    for j in range(n):
        initial.apply_gate(j,1,np.kron(rotation,rotation))
    prepared,info=tn.compress(initial,4,normalize=True)
    error=sum(np.sqrt(initial.discarded_weights))+info['discarded_bound']+abs(1-info['norm_before'])
    return prepared,error


def correlated(n,chi,intervals,tolerance,total,out,prepared_frame=False):
    started=perf_counter()
    initial,preparation_error=correlated_input(n,.12,.35)
    # The initial unitary preserves the product spectrum. Subtract the measured
    # gate truncation allowance below when using the prepared numerical anchor.
    anchor = (initial,.44**(n/2)-preparation_error,.56**(n/2)+preparation_error)
    def get_frame(herm):
        if not prepared_frame:
            return g.ProductFrame(herm,anchor=anchor)
        # In the final Hadamard basis the reference product density is diagonal.
        # W U W^-1 is a product of local conjugated gates. Its singular values
        # are bounded by multiplying the individual gate extrema.
        w=np.diag(np.array([.56,.44])**(-.25))
        h=np.array([[1.,1.],[1.,-1.]])/np.sqrt(2)
        hh=np.kron(h,h)
        gate=hh@np.diag([1.,1.,1.,np.exp(.35j)])@hh
        ww=np.kron(w,w)
        scales=np.linalg.svd(ww@gate@np.linalg.inv(ww),compute_uv=False)
        allowance=preparation_error*np.linalg.norm(w,2)**(2*n)
        return g.ProductFrame.from_anchor(herm,initial,[w]*n,
                       scales[-1]**(2*(n-1))-allowance,scales[0]**(2*(n-1))+allowance)
    density0=mpo.multiply(initial,mpo.adjoint(initial))
    density0,rounding=tn.compress(density0,16)
    density_error=2.**(n/2)*rounding['discarded_bound']
    def physical(t):
        density=copy.deepcopy(density0)
        scale=np.array([1.,np.exp(-t),np.exp(-t),1.])
        density.tensors=[a*scale[None,:,None] for a in density.tensors]
        return density
    counters={'max_sign_iterations':0,'max_force_bound':0.}
    def force(t,state):
        q=tn.add(state,mpo.adjoint(state),.5,.5)
        frame=get_frame(q)
        rho=g.exact_product(q,q,6_000_000)
        rho,rho_round=tn.compress(rho,2*chi)
        rhs,noise_round=tn.compress(mpo.onsite_dissipation(rho,np.full(n,.5),'dephasing'),2*chi)
        error=n*rho_round['discarded_bound']+noise_round['discarded_bound']
        f,info=g.adaptive_mpo_force(frame,rhs,error,.1*tolerance/total,
                                    sorted(set([chi,2*chi])))
        counters['max_sign_iterations']=max(counters['max_sign_iterations'],info['iterations'])
        counters['max_force_bound']=max(counters['max_force_bound'],info['force_error_bound'])
        return f,info
    def audit(t,state):
        herm=tn.add(state,mpo.adjoint(state),.5,.5)
        frame=get_frame(herm)
        report=frame.endpoint(physical(t),density_error)
        report['cp_bound']+=tn.distance(herm,state)
        return report
    state,info=g.integrate_guarded(initial,0.,total,total/intervals,chi,force,
        lambda a:tn.compress(a,chi,normalize=True),audit,tolerance,max_halvings=4)
    extra=dict(n=n,chi=chi,time=info['coordinate'],target=total,seconds=perf_counter()-started,**counters)
    if n<=8 and state is not None:
        density=tn.to_matrix(physical(info['coordinate']).dense(),n)
        v,u=np.linalg.eigh(density)
        exact=(u*np.sqrt(np.maximum(v,0)))@u.conj().T
        extra['cp_error']=np.linalg.norm(tn.to_matrix(state.dense(),n)-exact)
        if info['status']=='accepted':
            assert extra['cp_error']<=info['cp_bound']+1e-11
    frame_tag='_prep' if prepared_frame else ''
    return summarize(out,f'correlated_n{n}_chi{chi}_s{intervals}_tol{tolerance:g}{frame_tag}',state,info,extra)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('family',choices=['critical','correlated'])
    p.add_argument('--n',type=int,default=6)
    p.add_argument('--chi',type=int,default=8)
    p.add_argument('--intervals',type=int,default=8)
    p.add_argument('--tol',type=float,default=.005)
    p.add_argument('--time',type=float,default=.1)
    p.add_argument('--out',type=Path,default=Path(__file__).parent/'guarded_results')
    p.add_argument('--prepared-frame',action='store_true')
    p.add_argument('--trial-force-bond',type=int,default=0)
    args=p.parse_args()
    result=(critical if args.family=='critical' else correlated)(
        args.n,args.chi,args.intervals,args.tol,args.time,args.out,
        **({'prepared_frame':args.prepared_frame} if args.family=='correlated' else
           {'trial_force_bond':args.trial_force_bond}))
    raise SystemExit(0 if result['status']=='accepted' else 2)
