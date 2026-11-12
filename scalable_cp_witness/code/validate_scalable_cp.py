"""Independent dense checks and replay of the earlier rejected tensor proposals."""
import argparse
from pathlib import Path

import scalable_cp_witness as w
import guarded_cp_tdvp as guard
import cp_accuracy_refinement as old
import tdvp_core as tn
import mpo_cp_force as mpo
import numpy as np


def explicit_density(n,t,noise,p,angle):
    h=np.array([[1.,1.],[1.,-1.]])/np.sqrt(2)
    local=(mpo.I+p*mpo.X)/2
    rho=np.ones((1,1),complex)
    change=np.ones((1,1))
    for _ in range(n):
        rho=np.kron(rho,local)
        change=np.kron(change,h)
    labels=np.arange(2**n)
    bits=(labels[:,None]>>np.arange(n)) & 1
    phases=np.exp(1j*angle*np.sum(bits[:,1:]*bits[:,:-1],axis=1))
    rho=change@(phases[:,None]*rho*phases.conj()[None,:])@change.T
    survive=np.exp(-t)
    if noise=='dephasing':
        kraus=[np.sqrt((1+survive)/2)*mpo.I,np.sqrt((1-survive)/2)*mpo.Z]
    else:
        kraus=[np.diag([1.,np.sqrt(survive)]), np.sqrt(1-survive)*mpo.SM]
    for j in range(n):
        ops=[np.kron(np.eye(2**j),np.kron(k,np.eye(2**(n-1-j)))) for k in kraus]
        rho=sum(k@rho@k.conj().T for k in ops)
    return rho


def dense_checks(out):
    n,p,angle,t=5,.53,1.17,.21
    rng=np.random.default_rng(7325)
    rows=[]
    for noise in ('dephasing','loss'):
        engine=w.WitnessAudit(n,p,angle,residual_bond=4,noise=noise)
        density=explicit_density(n,t,noise,p,angle)
        reference=w.channel(engine.physical,t,noise)
        rho_error=np.linalg.norm(density-w.matrix(reference))
        assert rho_error<2e-14
        pop,vec=np.linalg.eigh(density)
        exact=(vec*np.sqrt(pop))@vec.conj().T
        for defect in ('accurate','perturbed','negative'):
            trial=exact.copy()
            if defect=='perturbed':
                z=rng.normal(size=trial.shape)+1j*rng.normal(size=trial.shape)
                trial+=.0001*(z+z.conj().T)
            if defect=='negative':
                trial=-trial
            root=old.root_mpo(trial,n)
            factor=old.root_mpo((vec*pop**.25)@vec.conj().T,n)
            eta=tn.distance(root,mpo.multiply(factor,mpo.adjoint(factor)))
            info,_=engine.endpoint(root,eta,reference,t)
            generic=w.certify_candidate(root,factor,reference,engine.metric,
                        engine.inverse_norm*(np.exp(n*t/2) if noise=='loss' else 1.))
            actual=np.linalg.norm(trial-exact)
            metric=w.matrix(engine.metric)
            slack=np.linalg.eigvalsh(metric-np.linalg.inv(density))[0]
            direct=np.linalg.norm(trial@trial-density)
            valid=actual<=min(info['bound'],generic)+2e-12 and slack>=-1e-8
            assert valid
            rows.append(dict(channel=noise,case=defect,density_error=rho_error,
                             square_residual=direct,actual=actual,bound=info['bound'],
                             min_metric_slack=slack))
        # A general complex residual detects index-transposition errors.
        e=rng.normal(size=trial.shape)+1j*rng.normal(size=trial.shape)
        a=old.root_mpo(e,n)
        expected=np.trace(e.conj().T@e@metric).real**.5
        assert abs(w.weighted_norm(a,engine.metric)/expected-1)<2e-13
    # Rank opening from a pure input. Use the maximally mixed auxiliary
    # density so the certificate needs no inverse of the true density.
    prob=.0004
    diag=np.zeros(2**n)
    diag[:2]=[1-prob,prob]
    rho=old.root_mpo(np.diag(diag),n)
    g0=np.zeros_like(diag)
    g0[0]=1.
    trial=old.root_mpo(np.diag(g0),n)
    d=2**n
    metric=old.root_mpo(d*np.eye(d),n)
    bound=w.certify_candidate(trial,trial,rho,metric,np.sqrt(d),gap_free=True)
    actual=np.linalg.norm(g0-np.sqrt(diag))
    assert actual <= bound and actual >= np.sqrt(prob)
    rows.append(dict(channel='rank opening',case='previous pure root',density_error=0.,
                     square_residual=np.linalg.norm(g0-diag),actual=actual,bound=bound,
                     min_metric_slack=''))
    w.save_table(out/'dense_checks.csv',rows)
    return rows


def replay(n, prior_dir, out):
    name=f'mpo_trajectory_n{n}_s8_chi12_tol1e-05_frame_last_accepted.npz'
    with np.load(prior_dir/name) as f:
        t0=float(f['time'])
        state=tn.MPS(n,max_bond=12)
        state.tensors=[f[f'a{j}'] for j in range(n)]
        state.center=n-1
    dt={48:.0025,64:.000625}[n]
    t1=t0+dt
    _,physical,_,frame_for=old.correlated_data(n)
    engine=w.WitnessAudit(n)
    results=[]
    def compare(stage,current,t,original):
        root=old.hermitian_round(current,current.bond_dimension())
        start=w.prepared_power(n,.25)
        _,eta=w.fit_witness(start,root,engine.factor_preconditioner,8,12,iterations=5)
        info,_=engine.endpoint(root,eta,w.channel(engine.physical,t),t)
        results.append(dict(n=n,stage=stage,time=t,old_frame=original,
                            new_bound=info['bound']+tn.distance(root,current),witness=eta,
                            passed=info['bound']+tn.distance(root,current)<=1e-5))
    b=max(6,state.bond_dimension())
    def force(t,current):
        try:
            frame=frame_for(current)
        except guard.Rejected as failure:
            compare('TDVP force input',current,t,failure.reason)
            raise
        square=tn.compress(guard.exact_product(frame.root,frame.root,frame.max_elements),2*b)[0]
        rhs=mpo.onsite_dissipation(square,np.full(n,.5),'dephasing')
        f=old.chebyshev_solve(frame.root,rhs,frame.lower,frame.upper,b,4,frame)
        return f,{}
    try:
        trial,_=tn.midpoint_step(state,t0,dt,force,b,
                    retract=lambda a:(old.hermitian_round(a,b,normalize=True),{}),mode='two')
        try:
            frame_for(trial)
        except guard.Rejected as failure:
            compare('TDVP endpoint',trial,t1,failure.reason)
    except guard.Rejected:
        pass
    previous=frame_for(state)
    for fb,wb,k in [(6,8,12),(6,12,24),(8,16,24),(12,20,32)]:
        if fb<b:
            continue
        x=old.mpo_trial(previous,physical(t1),fb,wb,k,True)
        status='verified'
        try:
            frame_for(x)
        except guard.Rejected as failure:
            status=failure.reason
        compare(f'endpoint correction {fb}/{wb}/{k}',x,t1,status)
    w.save_table(out/f'replay_n{n}.csv',results)
    print(results,flush=True)


def kraus_residual(defect, inverse_root, time, noise):
    """Second audit via a Kraus purification of the metric and a streaming QR norm."""
    survive=np.exp(-time)
    if noise=='dephasing':
        ops=np.stack([np.sqrt((1+survive)/2)*mpo.I,np.sqrt((1-survive)/2)*mpo.Z])
    else:
        ops=np.stack([np.diag([1.,np.sqrt(survive)]),np.sqrt(1-survive)*mpo.SM])
        ops=np.diag([1/(2-survive),1/survive])@ops
    carry=np.ones((1,1),complex)
    log_scale=0.
    for x,m in reversed(list(zip(defect.tensors,inverse_root.tensors))):
        x=x.reshape(x.shape[0],2,2,x.shape[-1])
        m=m.reshape(m.shape[0],2,2,m.shape[-1])
        local=np.einsum('eqj,cjkd->cqked',ops,m,optimize=True)
        right=carry.reshape(x.shape[-1],m.shape[-1],-1)
        moved=np.einsum('cqked,bdr->cqkebr',local,right,optimize=True)
        block=np.einsum('apqb,cqkebr->acpker',x,moved,optimize=True)
        block=block.reshape(x.shape[0]*m.shape[0],-1)
        carry=np.linalg.qr(block.T,mode='r').T
        scale=np.linalg.norm(carry)
        if scale<=0:
            raise ArithmeticError('Lost the weighted residual in the QR sweep')
        carry/=scale
        log_scale+=np.log(scale)
    return np.exp(log_scale)


def checkpoint_audit(path,out):
    with np.load(path) as data:
        n=sum(k.startswith('x') for k in data.files)
        t=float(data['time'])
        claimed=float(data['bound'])
        def load(prefix):
            tensors=[data[f'{prefix}{j}'] for j in range(n)]
            s=tn.MPS(n,max_bond=max(a.shape[-1] for a in tensors))
            s.tensors,s.center=tensors,n-1
            return s
        root,factor=load('x'),load('g')
    stem=path.stem
    noise='loss' if '_loss' in stem else 'dephasing'
    if 'parity' in stem:
        inv=w.parity_power(n,-.5)
        target=w.channel(w.parity_power(n,1.),t)
    else:
        p=float(stem.split('_p')[1].split('_')[0])
        angle=float(stem.split('_a')[1].split('_')[0])
        inv=w.prepared_power(n,-.5,p,angle)
        target=w.channel(w.prepared_power(n,1.,p,angle),t,noise)
    herm=old.hermitian_round(root,root.bond_dimension())
    branch=tn.distance(herm,mpo.multiply(factor,mpo.adjoint(factor)))
    nonherm=tn.distance(herm,root)
    defect=tn.add(mpo.multiply(herm,herm),target,beta=-1.)
    defect.move_center(0)
    defect.move_center(n-1)
    weighted=kraus_residual(defect,inv,t,noise)
    checked=nonherm+2*branch+weighted
    assert np.isfinite(checked) and checked<=claimed+max(1e-11,1e-5*claimed),(checked,claimed)
    row=dict(n=n,time=t,stored_bound=claimed,kraus_qr_bound=checked,
             branch_distance=branch,rehermitization=nonherm,weighted_residual=weighted)
    w.save_table(out/(stem.replace('_last_accepted','')+'_verification.csv'),[row])
    print(row)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay',type=int,choices=[48,64])
    parser.add_argument('--checkpoint',type=Path)
    here=Path(__file__).resolve().parent
    packed=here.name=='code'
    parser.add_argument('--prior',type=Path,default=here.parent/'inputs' if packed
                        else here.parent/'outputs'/'cp_accuracy_refinement')
    parser.add_argument('--out',type=Path,default=here.parent if packed
                        else here.parent/'outputs'/'scalable_cp_witness')
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    if args.checkpoint:
        checkpoint_audit(args.checkpoint,args.out)
    elif args.replay:
        replay(args.replay,args.prior,args.out)
    else:
        print(dense_checks(args.out))
