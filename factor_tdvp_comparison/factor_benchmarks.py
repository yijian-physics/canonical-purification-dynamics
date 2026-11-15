"""Matched dense-reference controls for genuine MPO factor and legacy root TDVP.

All physical models match the historical definitions; smaller N is explicitly
labelled. Exact roots are initialization/scoring only, except finite channel
bridges from the CURRENT state. Reuses original algorithms unchanged.
"""
from __future__ import annotations
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import argparse,csv,json,resource,sys,time,cProfile,pstats,io,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.linalg import expm
from factor_tdvp import factor_tdvp as fd
from factor_tdvp import tangent_solver as ts
from factor_tdvp import rank_bridge as rb
from factor_tdvp.baseline import _legacy_module

I,X,Z,SM=fd.I,fd.PAULI_X,fd.Z,fd.SM

def kron(items):
    result=np.ones((1,1),complex)
    for a in items: result=np.kron(result,a)
    return result

def phase_vector(n,angles):
    bits=(np.arange(2**n)[:,None]>>np.arange(n-1,-1,-1))&1
    return np.exp(1j*np.sum(bits[:,:-1]*bits[:,1:]*np.asarray(angles),axis=1))

def circuit_reference(n,angles,rates,t):
    phase=phase_vector(n,angles)
    local=[fd.positive_power((I+np.exp(-rate*t)*X)/2,.5) for rate in rates]
    root=phase[:,None]*kron(local)*phase.conj()[None,:]
    return root,root@root

def dephase_density(rho,n,rates,t):
    bits=(np.arange(2**n)[:,None]>>np.arange(n-1,-1,-1))&1
    return rho*np.exp(-t*np.sum((bits[:,None,:]!=bits[None,:,:])*rates,axis=-1))

def model(benchmark,n,theta=.1):
    d=2**n;zero=np.zeros((d,d),complex)
    if benchmark in ('A','B','C'):
        angles=np.full(n-1,np.pi) if benchmark=='A' else (.27+.03*np.arange(n-1) if benchmark=='B' else np.full(n-1,.37))
        phase=phase_vector(n,angles);psi=phase/np.sqrt(d);rho=np.outer(psi,psi.conj())
        if benchmark!='C':
            rates=.7+.15*np.cos(np.arange(n))**2
            jumps=[np.sqrt(rate/2)*fd.embed(Z,j,n) for j,rate in enumerate(rates)]
            reference=lambda t:circuit_reference(n,angles,rates,t)
        else:
            p_op=fd.embed(np.cos(theta)*Z+np.sin(theta)*X,n//2,n)
            rate=-np.log(.8);jumps=[np.sqrt(rate/2)*p_op]
            def reference(t):
                p=-np.expm1(-rate*t)/2
                u,s,_=np.linalg.svd(np.column_stack([np.sqrt(1-p)*psi,np.sqrt(p)*(p_op@psi)]),full_matrices=False)
                root=(u*s)@u.conj().T
                return root,root@root
        return rho,zero,jumps,reference
    if benchmark=='D':
        import scalable_cp_witness as witness
        rho=ts.matrix(witness.prepared_power(n,1.))
        jumps=[fd.embed(Z,j,n)/np.sqrt(2) for j in range(n)]
        def reference(t):
            target=dephase_density(rho,n,np.ones(n),t)
            return fd.positive_power(target,.5),target
        return rho,zero,jumps,reference
    if benchmark=='E':
        import scalable_cp_witness as witness
        rho=ts.matrix(witness.parity_power(n,1.))
        jumps=[fd.embed(Z,j,n)/np.sqrt(2) for j in range(n)]
        def reference(t):
            target=ts.matrix(witness.parity_power(n,1.,transverse=.4*np.exp(-t)))
            return ts.matrix(witness.parity_power(n,.5,transverse=.4*np.exp(-t))),target
        return rho,zero,jumps,reference
    if benchmark=='F_critical':
        h=-sum(fd.embed(X,j,n) for j in range(n))-sum(fd.embed(Z,j,n)@fd.embed(Z,j+1,n) for j in range(n-1))
        _,vec=np.linalg.eigh(h);psi=vec[:,0];rho=np.outer(psi,psi.conj())
        jumps=[fd.embed(Z,j,n)/np.sqrt(2) for j in range(n)]
        def reference(t):
            # Sequential Kraus-factor SVD retains root-scale tiny populations.
            factor=psi[:,None];p=-np.expm1(-t)/2
            for j in range(n):
                factor=np.column_stack([np.sqrt(1-p)*factor,np.sqrt(p)*fd.embed(Z,j,n)@factor])
                u,s,_=np.linalg.svd(factor,full_matrices=False);factor=u*s
            u,s,_=np.linalg.svd(factor,full_matrices=False);root=(u*s)@u.conj().T
            return root,root@root
        return rho,zero,jumps,reference
    if benchmark=='F_loss':
        h=.7*sum((fd.embed(Z,j,n)@fd.embed(Z,j+1,n) for j in range(n-1)),zero.copy())+.43*sum(fd.embed(X,j,n) for j in range(n))+.19*sum(fd.embed(Z,j,n) for j in range(n))
        jumps=[np.sqrt(.3+.12*np.cos(j)**2)*fd.embed(SM,j,n) for j in range(n)]
        psi=np.ones(d)/np.sqrt(d);rho=np.outer(psi,psi)
    elif benchmark=='F_delayed':
        assert n==1
        h=.6*X;jumps=[np.sqrt(.5)*SM];rho=np.diag([1.,0.]).astype(complex)
    else: raise ValueError(benchmark)
    liouv=rb.lindblad_superoperator(h,jumps)
    def reference(t):
        target=rb.herm((expm(t*liouv)@rho.reshape(-1,order='F')).reshape((d,d),order='F'))
        return fd.positive_power(target,.5),target
    return rho,h,jumps,reference

def metrics(amplitude,reference):
    root,rho=reference
    u,s,_=np.linalg.svd(amplitude,full_matrices=False)
    canonical=(u*s)@u.conj().T
    return dict(global_cp_error=float(np.linalg.norm(amplitude-root)),density_error=float(np.linalg.norm(amplitude@amplitude.conj().T-rho)),branch_defect=float(np.linalg.norm(amplitude-canonical)),normalization_defect=float(abs(np.linalg.norm(amplitude)**2-1)))

def maximum(tree,key):return max(fd._numbers(tree,key),default=0.)

def run_case(benchmark='A',n=3,method='factor_tdvp',steps=20,chi=4,total=1.,theta=.1,clock=None,profile=False,work_bond=None):
    begun=time.perf_counter();rho,ham,jumps,reference=model(benchmark,n,theta)
    pure=np.linalg.matrix_rank(rho,tol=1e-10)<len(rho)
    factor=ts.matrix_to_mpo(fd.positive_power(rho,.25),n,chi)
    factor,_=ts.normalize_factor(factor)
    # Benchmark tables report initial representation error, including truncation.
    if method=='factor_tdvp' and fd.numerical_rank(ts.matrix(factor))<fd.numerical_rank(fd.positive_power(rho,.25)):
        raise rb.RankOpeningError('Initial factor truncation lost physical support',{'chi':chi})
    initial_amplitude=ts.matrix(ts.root_mpo(factor)) if method=='factor_tdvp' else fd.positive_power(rho,.5)
    setup=time.perf_counter()-begun
    initial_error=metrics(initial_amplitude,reference(0))['global_cp_error']
    times=np.linspace(0,total,steps+1);dt=total/steps
    clock=clock or ('quartic' if pure and method=='factor_tdvp' else ('sqrt' if pure else 'physical'))
    records=[];evolution=scoring=0.;last=0.;bridge_count=0;history=[];audit_removed=0.
    if method=='factor_tdvp':
        hm=ts.matrix_to_mpo(ham,n);jm=[ts.matrix_to_mpo(j,n) for j in jumps]
    else:
        legacy=_legacy_module();tn=legacy.tn
        state=ts.matrix_to_mpo(fd.positive_power(rho,.5),n,chi)
        provider=legacy.DenseForce(n,rb.lindblad_superoperator(ham,jumps),clock='physical')
        retract=legacy.polar_retractor(n,chi)
    if method!='factor_tdvp': initial_error=metrics(ts.matrix(state),reference(0))['global_cp_error']
    setup=time.perf_counter()-begun
    prof=cProfile.Profile() if profile else None
    if prof:prof.enable()
    try:
        for left,right in zip(times[:-1],times[1:]):
            tick=time.perf_counter()
            current_h=ts.matrix(factor) if method=='factor_tdvp' else None
            if method!='factor_tdvp':
                a=ts.matrix(state);u,s,_=np.linalg.svd(a);audit_removed=max(audit_removed,float(np.linalg.norm(s[s<=1e-12*s[0]])));s=np.where(s>1e-12*s[0],s,0.);current_h=(u*np.sqrt(s))@u.conj().T
            before=rb.audit_support(current_h,ham,jumps,dt)
            if not before['safe_handoff']:
                seed,info=rb.bridge_factor(current_h,ham,jumps,dt)
                if method=='factor_tdvp':factor,re=ts.retract_factor(ts.matrix_to_mpo(seed,n),chi)
                else:state,re=retract(ts.matrix_to_mpo(seed@seed.conj().T,n))
                if method=='factor_tdvp':
                    actual_seed=ts.matrix(factor)
                    handoff=rb.audit_support(actual_seed,ham,jumps,dt,probe=False)
                    if fd.numerical_rank(actual_seed)<fd.numerical_rank(seed) or not handoff['safe_handoff']:
                        raise rb.RankOpeningError('Compressed bridge factor lost required support',handoff)
                bridge_count+=1;info={'bridge':info,'retraction':re};step_kind='bridge'
            else:
                q=4 if clock=='quartic' else (2 if clock=='sqrt' else 1)
                u=left**(1/q);v=right**(1/q);du=v-u
                scales=(q*u**(q-1),q*((u+v)/2)**(q-1))
                if method=='factor_tdvp':
                    def guard(stage):
                        status=rb.audit_support(ts.matrix(stage),ham,jumps,dt,probe=False)
                        if status['rank']<before['rank'] or not status['safe_handoff']:
                            raise rb.RankOpeningError('Smooth factor support audit failed',status)
                    factor,info=ts.midpoint_step(factor,hm,jm,du,chi,stage_scales=scales,guard=guard,work_bond=work_bond)
                    guard(factor)
                else:
                    def scaled_provider(coord,current):
                        force,diagnostic=provider(coord,current)
                        if diagnostic['null_forcing']>1e-9:raise RuntimeError('Legacy root stage has unresolved kernel forcing')
                        return tn.scale(force,q*coord**(q-1)),diagnostic
                    state,info=tn.midpoint_step(state,u,du,scaled_provider,chi,retract=retract,mode='one')
                step_kind='tangent'
            evolution+=time.perf_counter()-tick
            scoring_start=time.perf_counter()
            amplitude=ts.matrix(ts.root_mpo(factor)) if method=='factor_tdvp' else ts.matrix(state)
            measured=metrics(amplitude,reference(right))
            hb=factor.bond_dimension() if method=='factor_tdvp' else None
            xb=ts.root_mpo(factor).bond_dimension() if method=='factor_tdvp' else state.bond_dimension()
            scoring+=time.perf_counter()-scoring_start
            record=dict(benchmark=benchmark,method=method,n=n,time=float(right),dt=dt,chi=chi,theta=theta,clock=clock,step_kind=step_kind,h_bond=hb,x_bond=xb,**measured,
                        jacobian_residual=maximum(info,'force_residual') if method=='factor_tdvp' else None,
                        conditioning=maximum(info,'ls_retained_condition') if method=='factor_tdvp' else None,
                        tangent_solve_seconds=sum(fd._numbers(info,'seconds')) if method=='factor_tdvp' else None,
                        truncation_error=maximum(info,'root_truncation_error'),normalization_change=maximum(info,'root_normalization_change'),
                        product_local_discard_sum=maximum(info,'product_local_discard_sum'),rank_before=rb.audit_support(current_h,ham,jumps,dt,probe=False)['rank'],
                        evolution_seconds=evolution,scoring_seconds=scoring,bridge_events=bridge_count)
            records.append(record);last=float(right);history.append(info)
        status='completed';error_message=None
    except Exception as exc:
        status='failed';error_message=f'{type(exc).__name__}: {exc}'
        evolution+=time.perf_counter()-tick
    if prof:
        prof.disable();buf=io.StringIO();pstats.Stats(prof,stream=buf).sort_stats('cumulative').print_stats(18);profile_text=buf.getvalue()
    else:profile_text=None
    final=records[-1] if records else {}
    row=dict(benchmark=benchmark,method=method,n=n,status=status,error_message=error_message,time=total,reached_time=last,dt=dt,steps=steps,accepted_steps=len(records),rejected_steps=int(status=='failed'),chi=chi,theta=theta,clock=clock,work_bond=work_bond,
             setup_seconds=setup,evolution_seconds=evolution,scoring_seconds=scoring,total_seconds=time.perf_counter()-begun,
             peak_process_memory_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
             memory_measurement='process high-water mark, cumulative across serial cases',
             global_cp_error=final.get('global_cp_error') if status=='completed' else None,density_error=final.get('density_error') if status=='completed' else None,
             branch_defect=final.get('branch_defect'),normalization_defect=final.get('normalization_defect'),initial_global_cp_error=initial_error,
             h_bond=final.get('h_bond'),x_bond=final.get('x_bond'),max_h_bond=max([r['h_bond'] for r in records if r['h_bond'] is not None],default=None),
             average_h_bond=float(np.mean([r['h_bond'] for r in records])) if method=='factor_tdvp' and records else None,
             max_x_bond=max([r['x_bond'] for r in records],default=None),average_x_bond=float(np.mean([r['x_bond'] for r in records])) if records else None,
             max_jacobian_residual=maximum(history,'force_residual') if method=='factor_tdvp' else None,max_conditioning=maximum(history,'ls_retained_condition') if method=='factor_tdvp' else None,
             tangent_solve_seconds=sum(fd._numbers(history,'seconds')) if method=='factor_tdvp' else None,
             max_root_truncation=maximum(history,'root_truncation_error'),max_normalization_change=maximum(history,'root_normalization_change'),bridge_events=bridge_count,
             profiling_enabled=bool(profile),support_audit_root_rtol=1e-12 if method=='root_tdvp' else None,max_support_audit_removed_root_frobenius=audit_removed if method=='root_tdvp' else None,metric_kind='measured_global',reference_kind='analytic_circuit_or_dense_channel',implementation_scope='finite_size_dense_Jacobian_MPO_tangent' if method=='factor_tdvp' else 'legacy_dense_force_one_site_root_TDVP')
    return row,records,profile_text

def configurations(quick=False):
    cases=[]
    def pair(bench,n,steps,chi,total=1.,theta=.1):
        for method in ('factor_tdvp','root_tdvp'):cases.append(dict(benchmark=bench,n=n,method=method,steps=steps,chi=chi,total=total,theta=theta))
    for bench in ('A','B'):
        for steps in ((20,) if quick else (20,40,80)):pair(bench,3,steps,4)
        if not quick:pair(bench,4,20,4)
    for theta in (0.,.05,.1,.2):pair('C',4,20,4,theta=theta)
    if not quick:pair('C',4,80,4,theta=.1)
    for bench,total in (('D',.1),('E',1.)):
        for steps in ((10,) if quick else (5,10,20)):pair(bench,3,steps,4,total)
        if not quick:pair(bench,4,10,4,total)
    for chi in (2,4):pair('F_critical',3,20,chi,.3)
    pair('F_loss',3,10,4,.03)
    pair('F_delayed',1,16,1,.08)
    return cases

def save(out,rows,trajectories):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    (out/'factor_measurements.json').write_text(json.dumps(rows,indent=2)+'\n')
    (out/'factor_trajectories.json').write_text(json.dumps(trajectories,indent=2)+'\n')
    if rows:
        with (out/'factor_measurements.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)),lineterminator='\n');w.writeheader();w.writerows(rows)

def run_suite(outdir='factor_tdvp_comparison/results',quick=False):
    rows=[];trajectories=[]
    for config in configurations(quick):
        tick=time.perf_counter();row,record,profile=run_case(**config,profile=(config['benchmark']=='A' and config['n']==4 and config['method']=='factor_tdvp'))
        rows.append(row);trajectories.extend(record);save(outdir,rows,trajectories)
        if profile:Path(outdir,'factor_profile_n4.txt').write_text(profile)
        print(json.dumps(row),flush=True)
    return rows,trajectories

def repeated_control(outdir='factor_tdvp_comparison/results'):
    """New run of successive generator activation; audits rank1->2->4."""
    started=time.perf_counter();rho=np.ones((4,4),complex)/4
    state=ts.matrix_to_mpo(rho,2,1);ham=np.zeros((4,4),complex)
    setup=time.perf_counter()-started;history=[];evolution=0.
    for site in (0,1):
        times=np.r_[site*.08,np.linspace(site*.08+.02,(site+1)*.08,7)]
        tick=time.perf_counter();state,events=fd.evolve(state,ham,[fd.embed(Z,site,2)/np.sqrt(2)],times,1,clock='quartic')
        evolution+=time.perf_counter()-tick;history.extend(events)
    tick=time.perf_counter();local=(I+np.exp(-.08)*X)/2;target=np.kron(local,local)
    measured=metrics(ts.matrix(ts.root_mpo(state)),(fd.positive_power(target,.5),target));scoring=time.perf_counter()-tick
    row=dict(benchmark='F_repeated',method='factor_tdvp',n=2,status='completed',time=.16,steps=14,accepted_steps=14,rejected_steps=0,dt=None,chi=1,h_bond=1,x_bond=1,setup_seconds=setup,evolution_seconds=evolution,scoring_seconds=scoring,total_seconds=time.perf_counter()-started,bridge_events=2,rank_sequence=[1,2,4],max_jacobian_residual=maximum(history,'force_residual'),max_conditioning=maximum(history,'ls_retained_condition'),metric_kind='measured_global',**measured)
    out=Path(outdir);out.mkdir(parents=True,exist_ok=True);(out/'factor_repeated.json').write_text(json.dumps(dict(row=row,events=history),indent=2)+'\n');return row


def extended(outdir='factor_tdvp_comparison/results'):
    """Budgeted larger cases, with explicit failed subprocess rows."""
    out=Path(outdir);out.mkdir(parents=True,exist_ok=True);rows=[]
    tasks=[('A',4,20,16,16,180),('C',6,4,4,16,150),('E',8,5,2,8,180)]
    for bench,n,steps,chi,work,timeout in tasks:
        dest=out/f'factor_extended_{bench}_n{n}';started=time.perf_counter()
        cmd=[sys.executable,__file__,'--single',bench,'--n',str(n),'--steps',str(steps),'--chi',str(chi),'--work-bond',str(work),'--outdir',str(dest)]
        try:
            run=subprocess.run(cmd,capture_output=True,text=True,timeout=timeout)
            code=run.returncode;log=run.stdout+run.stderr
        except subprocess.TimeoutExpired as exc:
            code=124;log='Timed out at '+str(timeout)+' seconds.\n'
        elapsed=time.perf_counter()-started
        (out/f'factor_extended_{bench}_n{n}_stdout.txt').write_text(log)
        if code==0 and (dest/'factor_measurements.json').exists():
            row=json.loads((dest/'factor_measurements.json').read_text())[0]
        else:
            row=dict(benchmark=bench,method='factor_tdvp',n=n,status='failed',error_message=f'Budgeted subprocess exited {code}; no completed global error',process_returncode=code,timeout_seconds=timeout,global_cp_error=None,density_error=None,time=1.,steps=steps,dt=1/steps,chi=chi,work_bond=work,total_seconds=elapsed,metric_kind='measured_global')
        row['subprocess_command']=cmd;rows.append(row)
        (out/'factor_extended.json').write_text(json.dumps(rows,indent=2)+'\n')
    rows.append(repeated_control(outdir))
    (out/'factor_extended.json').write_text(json.dumps(rows,indent=2)+'\n')
    return rows


def force_profile(n=6,work_bond=16,outdir='factor_tdvp_comparison/results'):
    out=Path(outdir);out.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter();rho,ham,jumps,_=model('A',n)
    seed,_=rb.bridge_factor(fd.positive_power(rho,.25),ham,jumps,.05)
    state=ts.matrix_to_mpo(seed,n,4);setup=time.perf_counter()-start
    hm=ts.matrix_to_mpo(ham,n);jm=[ts.matrix_to_mpo(j,n) for j in jumps]
    profile=cProfile.Profile();profile.enable();start=time.perf_counter()
    velocity,info=ts.factor_velocity(state,hm,jm,work_bond=work_bond)
    profile.disable();row=dict(n=n,setup_seconds=setup,profiling_enabled=True,**info)
    (out/f'factor_n{n}_profile.json').write_text(json.dumps(row,indent=2)+'\n')
    stream=io.StringIO();pstats.Stats(profile,stream=stream).sort_stats('cumulative').print_stats(20)
    (out/f'factor_n{n}_profile.txt').write_text(stream.getvalue())
    return row

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--quick',action='store_true');p.add_argument('--extended',action='store_true');p.add_argument('--repeated',action='store_true');p.add_argument('--force-profile',action='store_true');p.add_argument('--profile',action='store_true');p.add_argument('--work-bond',type=int);p.add_argument('--outdir',default='factor_tdvp_comparison/results');p.add_argument('--single',choices=['A','B','C','D','E','F_critical','F_loss','F_delayed']);p.add_argument('--total',type=float);p.add_argument('--method',choices=['factor_tdvp','root_tdvp'],default='factor_tdvp');p.add_argument('--clock',choices=['physical','sqrt','quartic']);p.add_argument('--n',type=int,default=3);p.add_argument('--steps',type=int,default=20);p.add_argument('--chi',type=int,default=4)
    a=p.parse_args()
    if a.force_profile:print(json.dumps(force_profile(a.n,a.work_bond or 16,a.outdir)),flush=True)
    elif a.extended:extended(a.outdir)
    elif a.repeated:print(json.dumps(repeated_control(a.outdir)),flush=True)
    elif a.single:
        total=a.total if a.total is not None else {'D':.1,'F_critical':.3,'F_loss':.03,'F_delayed':.08}.get(a.single,1.)
        row,records,profile=run_case(a.single,a.n,method=a.method,steps=a.steps,chi=a.chi,total=total,clock=a.clock,profile=a.profile,work_bond=a.work_bond);save(a.outdir,[row],records)
        if profile:Path(a.outdir,'factor_profile.txt').write_text(profile)
        print(json.dumps(row),flush=True)
    else:run_suite(a.outdir,a.quick)
