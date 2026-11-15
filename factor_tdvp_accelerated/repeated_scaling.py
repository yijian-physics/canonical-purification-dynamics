"""Repeated, resource-bounded four-method benchmarks with exclusive timing.

Algorithms are unchanged: instrumentation wraps existing functions/callbacks.
The finite factor/root adapter retains dense support audits and reference
scoring. Unsupported rows are resource-policy decisions, never timings.
"""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import contextlib
import functools
import hashlib
import inspect
import json
import platform
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from factor_tdvp_accelerated import run_benchmarks as base
import numpy as np
import scipy

class Ledger:
    """Nested wall-time accounting; exclusive components cannot double count."""
    def __init__(self, clock=time.perf_counter):
        self.clock=clock; self.stack=[]; self.inclusive={}; self.exclusive={}; self.calls={}; self.patches=[]; self.guarded_failures=[]
    @contextlib.contextmanager
    def measure(self,name):
        frame=[self.clock(),0.];self.stack.append(frame)
        try:yield
        finally:
            elapsed=self.clock()-frame[0];self.stack.pop()
            self.inclusive[name]=self.inclusive.get(name,0.)+elapsed
            self.exclusive[name]=self.exclusive.get(name,0.)+max(0.,elapsed-frame[1])
            self.calls[name]=self.calls.get(name,0)+1
            if self.stack:self.stack[-1][1]+=elapsed
    def wrap(self,fn,name):
        @functools.wraps(fn)
        def measured(*a,**kw):
            try:
                with self.measure(name):return fn(*a,**kw)
            except Exception as exc:
                if hasattr(exc,'diagnostics'):
                    self.guarded_failures.append(dict(category=name,error=type(exc).__name__,diagnostics=exc.diagnostics))
                raise
        return measured
    def patch(self,obj,name,category):
        original=inspect.getattr_static(obj,name);fn=getattr(obj,name)
        self.patches.append((obj,name,original));setattr(obj,name,self.wrap(fn,category))
    def restore(self):
        for obj,name,original in reversed(self.patches):setattr(obj,name,original)

@contextlib.contextmanager
def instrument(ledger):
    import scalable_cp_witness as native
    from factor_tdvp_accelerated import matrix_free_solver as mf
    ts,rb=base.ts,base.fd.rb
    for obj,name,category in (
        (native,'fit_witness','witness_fit'),
        (native.WitnessAudit,'correction','correction_direction'),
        (native.WitnessAudit,'endpoint','certification'),
        (native.WitnessAudit,'__init__','reference_metric_initialization'),
        (native.WitnessAudit,'parity_chain','reference_metric_initialization'),
        (native,'channel','physical_or_inverse_metric_propagation'),
        (native,'tdvp_prediction','prediction_other'),
        (native,'factor_to_root','factor_root_construction_and_normalization'),
        (native,'hermitian_round','hermitian_retraction'),
        (mf,'factor_velocity','factor_tangent_force'),
        (ts,'retract_factor','factor_retraction'),
        (rb,'audit_support','dense_support_audit'),
        (rb,'bridge_factor','finite_channel_bridge')):
        ledger.patch(obj,name,category)
    old_midpoint=ts.tn.midpoint_step
    @functools.wraps(old_midpoint)
    def midpoint(state,u,h,force_provider,chi,retract=None,mode='one'):
        force=ledger.wrap(force_provider,'root_predictor_force')
        retraction=ledger.wrap(retract,'root_retraction') if retract else None
        return old_midpoint(state,u,h,force,chi,retract=retraction,mode=mode)
    ledger.patches.append((ts.tn,'midpoint_step',old_midpoint));ts.tn.midpoint_step=midpoint
    old_model=base.previous.model
    def model(*a,**kw):
        with ledger.measure('dense_model_initialization'):
            rho,ham,jumps,reference=old_model(*a,**kw)
        return rho,ham,jumps,ledger.wrap(reference,'independent_reference_scoring')
    ledger.patches.append((base.previous,'model',old_model));base.previous.model=model
    try:yield
    finally:ledger.restore()

def native_witness(config,out):
    import scalable_cp_witness as native
    n=config['n'];parity=config['family']=='E'
    times=np.linspace(0.,config['total'],config['steps']+1)[1:].tolist()
    levels=[(4,6,8),(6,8,12)] if parity else [(4,6,10),(6,8,12),(8,12,16)]
    success=native.run(n,times,levels,config['tolerance'],out,8 if parity else 4,predict=config['method']=='witness',parity=parity)
    paths=[p for p in out.glob('*.csv') if not p.name.endswith('_trials.csv')]
    import csv
    with paths[0].open() as stream:endpoints=list(csv.DictReader(stream))
    last=endpoints[-1]
    trials=[]
    for path in out.glob('*_trials.csv'):
        with path.open() as stream:trials.extend(csv.DictReader(stream))
    actual=float(last['actual']) if last['actual'] else None
    return dict(status='completed' if success else 'failed_certificate',
                global_cp_error=actual if success else None,global_cp_bound=float(last['bound']),
                reached_time=float(last['last_accepted']),root_bond=int(last['root_bond']),factor_bond=int(last['factor_bond']),
                positivity_witness_distance=float(last['witness']),accepted_steps=sum(r['accepted']=='True' and float(r['time'])>0 for r in endpoints),
                correction_trial_count=len(trials),accepted_correction_trials=sum(r['improved']=='True' for r in trials),
                used_predictions=sum(r['used_prediction']=='True' for r in endpoints),
                max_residual_rounding=max(float(r['residual_rounding']) for r in endpoints),
                accuracy_kind='measured_global_and_bound' if actual is not None else 'global_bound_only',
                auxiliary_information='independently propagated density and inverse-bound metric included in total',
                time_grid=times)

def start_memory_watchdog(out,limit_mib,interval=.25):
    """Child high-water RSS polling works inside the app sandbox.

    It is a sampled guard, not a hard address-space quota: overshoot between
    samples is possible. The event file survives an immediate guarded exit.
    """
    import threading
    stop=threading.Event()
    def watch():
        while not stop.is_set():
            peak=base.memory_mib()
            if peak>limit_mib:
                base.save(out/'memory_limit_event.json',dict(status='memory_limit',peak_process_memory_mib=peak,limit_mib=limit_mib,interval_seconds=interval))
                os._exit(86)
            stop.wait(interval)
    thread=threading.Thread(target=watch,daemon=True);thread.start()
    return stop

def worker(config,out,memory_limit_mib=2048.):
    out.mkdir(parents=True,exist_ok=True)
    worker_source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    watchdog_stop=start_memory_watchdog(out,memory_limit_mib)
    ledger=Ledger();started=time.perf_counter()
    try:
        with instrument(ledger):
            if config['method'] in ('witness','correction_only'):row=native_witness(config,out)
            else:
                raw=base.trajectory_case(config,out)
                base.save(out/'legacy_summary.json',raw)
                row=dict(raw,root_bond=raw.get('x_bond'),factor_bond=raw.get('h_bond'),accuracy_kind='measured_global',
                         auxiliary_information='dense independent reference scoring; no endpoint correction')
    except Exception as exc:
        (out/'exception.txt').write_text(traceback.format_exc())
        row=dict(status='failed',error_message=f'{type(exc).__name__}: {exc}',global_cp_error=None,global_cp_bound=None)
    total=time.perf_counter()-started
    watchdog_stop.set()
    row.update(config)
    row['memory_enforcement']='child getrusage peak-RSS watchdog, sampled every 250ms; transient overshoot possible'
    row['worker_rss_limit_mib']=memory_limit_mib
    row['worker_instrumentation_sha256']=worker_source_hash
    row.update(total_seconds=total,peak_process_memory_mib=base.memory_mib(),
               guarded_solver_failures=ledger.guarded_failures,
               timing_exclusive_seconds=dict(ledger.exclusive,other_worker_seconds=max(0.,total-sum(ledger.exclusive.values()))),
               timing_inclusive_seconds=ledger.inclusive,timing_calls=ledger.calls,
               timing_semantics='exclusive categories partition worker wall time; inclusive contextual categories overlap; other includes scoring, correction application, I/O and uninstrumented setup',
               timing_partition_error_seconds=total-sum(ledger.exclusive.values())-max(0.,total-sum(ledger.exclusive.values())))
    if (out/'solves.json').exists():
        solves=json.loads((out/'solves.json').read_text())
        row['krylov_iterations_total']=sum(s['iterations'] for s in solves)
        row['force_solve_count']=len(solves)
        row['max_relative_force_residual']=max(s['relative_force_residual'] for s in solves)
    base.save(out/'summary.json',row)

def configurations(suite):
    def finite(family,n,steps,method,repeat_count=3,**kw):
        cap=16 if n==4 and family=='D' else 4
        if family=='E' and n>=5:cap=2
        return dict(kind='trajectory',family=family,n=n,method=method,steps=steps,chi=cap,
                    work_bond=4**(n//2) if n<=6 else 16,total=.1 if family=='D' else 1.,
                    solver_tolerance=1e-8,requested_repeats=repeat_count,**kw)
    if suite == 'supplemental':
        for steps in (50,150):
            for method in ('root','matrix_free'):
                yield finite('E',3,steps,method)
        for n in (256,512):
            yield dict(kind='trajectory',family='D',n=n,method='witness',steps=5,
                       total=.1,tolerance=1e-5,requested_repeats=1,timeout_seconds=120.)
        config=finite('E',8,5,'matrix_free',repeat_count=1)
        config.update(work_bond=64,timeout_seconds=45.)
        yield config
    if suite == 'parity_eight':
        config=finite('E',8,50,'matrix_free')
        config.update(work_bond=64,timeout_seconds=45.)
        yield config
    if suite in ('primary','all'):
        for family in ('D','E'):
            for n in (3,4):
                for steps in (5,10,20):
                    for method in ('root','matrix_free'):yield finite(family,n,steps,method)
                for tolerance in (1e-5,1e-6,1e-7):
                    for method in ('witness','correction_only'):
                        yield dict(kind='trajectory',family=family,n=n,method=method,steps=5,total=.1 if family=='D' else 1.,tolerance=tolerance,requested_repeats=3)
    if suite in ('extension','all'):
        for family in ('D','E'):
            for n in (5,6):
                for method in ('root','matrix_free'):yield finite(family,n,5,method)
                for method in ('witness','correction_only'):
                    for tol in (1e-5,1e-7):yield dict(kind='trajectory',family=family,n=n,method=method,steps=5,total=.1 if family=='D' else 1.,tolerance=tol,requested_repeats=3)
            for method in ('matrix_free',):yield finite(family,8,5,method,repeat_count=1)
        for method in ('root','matrix_free'):
            config=finite('F_loss',3,10,method);config['total']=.03;yield config
    if suite in ('large','all'):
        for family in ('D','E'):
            for n in (8,12,16,32,64,128,256,512):
                for method in ('witness','correction_only'):
                    yield dict(kind='trajectory',family=family,n=n,method=method,steps=5,total=.1 if family=='D' else 1.,tolerance=1e-5,requested_repeats=3 if n<=16 else 1)
    if suite in ('unsupported','all'):
        for family in ('D','E'):
            for n in (8,12,16,32,64,128,256,512):
                for method in ('root','matrix_free'):
                    if n==8 and method=='matrix_free':continue
                    yield dict(kind='trajectory',family=family,n=n,method=method,steps=5,total=.1 if family=='D' else 1.,requested_repeats=0,
                               unsupported_reason='Conventional root adapter explicitly constructs a (4**N)^2 dense superoperator, exceeding the memory cap.' if method=='root' else 'Finite factor driver requires dense factor/density support eigendecompositions; this run caps dense-supported trajectories at N=8. The matrix-free force itself has no such dense-output restriction.')

def identifier(config):
    digest=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()[:8]
    return f"{config['family']}{config['n']:03d}_{config['method']}_s{config['steps']}_{digest}"

def execute(config,repeat,out,timeout,memory_mib):
    cid=identifier(config);folder=out/f'{cid}_r{repeat}';folder.mkdir()
    cmd=[sys.executable,str(Path(__file__).resolve()),'--worker',json.dumps(config),'--out',str(folder),'--memory-mib',str(memory_mib)]
    started=time.perf_counter();peak=0.;stop=None
    with (folder/'stdout.txt').open('w') as stream:
        process=subprocess.Popen(cmd,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        while process.poll() is None:
            elapsed=time.perf_counter()-started
            try:
                sample=subprocess.run(['ps','-o','rss=','-p',str(process.pid)],capture_output=True,text=True,check=False,timeout=2)
                rss=float(sample.stdout.strip() or 0)/1024
                peak=max(peak,rss)
            except (ValueError,OSError,subprocess.TimeoutExpired):pass
            if elapsed>timeout:stop='timeout'
            elif peak>memory_mib:stop='memory_limit'
            if stop:
                try:os.killpg(process.pid,signal.SIGTERM)
                except ProcessLookupError:pass
                try:process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    try:os.killpg(process.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                    process.wait()
                break
            time.sleep(.25)
    wall=time.perf_counter()-started
    if stop:row=dict(config,status=stop,global_cp_error=None,global_cp_bound=None,total_seconds=None,error_message=f'watchdog {stop}',budget_elapsed_seconds=wall)
    elif (folder/'memory_limit_event.json').exists():
        row=dict(config,**json.loads((folder/'memory_limit_event.json').read_text()),global_cp_error=None,global_cp_bound=None,total_seconds=None)
    elif (folder/'summary.json').exists():row=json.loads((folder/'summary.json').read_text())
    else:row=dict(config,status='failed',global_cp_error=None,global_cp_bound=None,total_seconds=None,error_message=f'worker exit {process.returncode} without summary')
    row.update(config_id=cid,repeat=repeat,directory=folder.name,wall_seconds=wall,
               peak_observed_rss_mib=peak,timeout_seconds=timeout,rss_limit_mib=memory_mib,
               parent_rss_monitor_available=peak>0,memory_limit_semantics='child peak-RSS watchdog every 250ms; optional parent ps monitor; transient overshoot possible',process_returncode=process.returncode)
    base.save(folder/'measurement.json',row);return row

def repeated_frontier(rows,target):
    """Cheapest observed median with complete, valid, same-endpoint evidence."""
    def finite_number(value, positive=False):
        try:
            number=float(value)
        except (TypeError,ValueError,OverflowError):
            return False
        return bool(np.isfinite(number) and (number>0 if positive else number>=0))
    def comparison_key(row):
        return row['family'],row['n'],row['method'],row.get('total')
    groups={}
    for r in rows:
        if r.get('requested_repeats',0)<3:continue
        groups.setdefault(r['config_id'],[]).append(r)
    choices=[]
    for cid,group in groups.items():
        expected=group[0]['requested_repeats']
        if (len(group)!=expected or any(r.get('requested_repeats')!=expected for r in group)
            or {r.get('repeat') for r in group}!=set(range(1,expected+1))
            or any(r['status']!='completed' for r in group)
            or len({comparison_key(r) for r in group})!=1):
            continue
        # A supplied actual error takes precedence even if invalid: never
        # hide NaN, infinity or a failing measurement behind a favorable bound.
        errs=[r.get('global_cp_error') if r.get('global_cp_error') is not None else r.get('global_cp_bound') for r in group]
        if any(not finite_number(e) or float(e)>target for e in errs):continue
        values=[r.get('total_seconds') for r in group]
        if any(not finite_number(v,positive=True) for v in values):continue
        values=list(map(float,values));errs=list(map(float,errs))
        kinds={'measured' if r.get('global_cp_error') is not None else 'bound' for r in group}
        evidence=next(iter(kinds)) if len(kinds)==1 else 'mixed_measured_and_bound'
        choices.append(dict(family=group[0]['family'],n=group[0]['n'],method=group[0]['method'],total=group[0].get('total'),target=target,attained=True,config_id=cid,repeats=len(group),median_seconds=float(np.median(values)),min_seconds=min(values),max_seconds=max(values),max_error=max(errs),evidence=evidence,steps=group[0]['steps']))
    result=[]
    keys={comparison_key(r) for r in rows}
    for key in sorted(keys,key=lambda k:(k[:3],float('-inf') if k[3] is None else k[3])):
        matching=[r for r in choices if comparison_key(r)==key]
        if matching:result.append(min(matching,key=lambda r:r['median_seconds']))
        else:result.append(dict(family=key[0],n=key[1],method=key[2],total=key[3],target=target,attained=False,reason='No complete configuration with at least three distinct repeats met target in every repeat'))
    return result

def run(args):
    args.out.mkdir(parents=True,exist_ok=False)
    try:
        from threadpoolctl import threadpool_info
        blas=threadpool_info()
    except ImportError:blas=[]
    hardware={}
    for key in ('hw.model','hw.memsize','machdep.cpu.brand_string'):
        try:
            p=subprocess.run(['sysctl','-n',key],capture_output=True,text=True,check=False);hardware[key]=p.stdout.strip() or 'unavailable'
        except FileNotFoundError:hardware[key]='unavailable on this platform'
    base.save(args.out/'environment.json',dict(command=sys.argv,python=sys.version,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform(),hardware=hardware,blas=blas,blas_threads=1,timeout_seconds=args.timeout,rss_limit_mib=args.memory_mib,source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__).resolve(),ROOT/'factor_tdvp_accelerated/matrix_free_solver.py']}))
    rows=[];start=time.perf_counter()
    for config in configurations(args.suite):
        if 'unsupported_reason' in config:
            rows.append(dict(config,config_id=identifier(config),repeat=0,status='unsupported',total_seconds=None,wall_seconds=None,global_cp_error=None,global_cp_bound=None))
        else:
            for rep in range(1,config['requested_repeats']+1):
                if time.perf_counter()-start>args.budget:
                    row=dict(config,config_id=identifier(config),repeat=rep,status='not_run_budget',total_seconds=None,wall_seconds=None,global_cp_error=None,global_cp_bound=None)
                else:row=execute(config,rep,args.out,config.get('timeout_seconds',args.timeout),args.memory_mib)
                rows.append(row)
                print(row.get('directory',row['config_id']),row['status'],row.get('global_cp_error'),row.get('global_cp_bound'),flush=True)
        base.save(args.out/'measurements.json',rows);base.table(args.out/'measurements.csv',rows)
        for target in (1e-5,1e-6,1e-7):
            data=repeated_frontier(rows,target);base.save(args.out/f'frontier_{target:g}.json',data);base.table(args.out/f'frontier_{target:g}.csv',data)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--suite',choices=('primary','extension','large','unsupported','supplemental','parity_eight','all'),default='all');parser.add_argument('--out',type=Path,required=True);parser.add_argument('--timeout',type=float,default=60.);parser.add_argument('--memory-mib',type=float,default=2048.);parser.add_argument('--budget',type=float,default=900.);parser.add_argument('--worker')
    a=parser.parse_args()
    if a.worker:worker(json.loads(a.worker),a.out,a.memory_mib)
    else:run(a)
