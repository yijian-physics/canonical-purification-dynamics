"""Reproducible, isolated finite-reference experiments for matrix-free factor TDVP.

Every worker runs with one BLAS thread. The production force is matrix-free;
these small-system experiments deliberately use independent dense references
and the unchanged dense rank-event audit. Timings include those costs and are
not a claim of scalable end-to-end evolution. A fresh output directory is
required: earlier numerical runs are never overwritten by the suite.
"""
from __future__ import annotations
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_key] = '1'
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/tdvp-matplotlib')
import argparse
import contextlib
import cProfile
import csv
import io
import json
import platform
import pstats
import resource
import subprocess
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import scipy
from factor_tdvp import tangent_solver as ts
from factor_tdvp import factor_tdvp as fd
from factor_tdvp_comparison import factor_benchmarks as previous


def memory_mib():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024**2 if sys.platform == 'darwin' else 1024)


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def table(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, keys)
        writer.writeheader(); writer.writerows(rows)


def profile_summary(profiler, path):
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats('cumulative')
    stats.print_stats(50)
    Path(path).write_text(stream.getvalue())
    # These cumulative categories overlap; never sum them into a total.
    names = ('tangent_basis', 'multiply', 'compress', 'dense', 'lstsq',
             'normalize_factor', 'retract_factor', 'audit_support', 'apply', 'adjoint')
    return {name: sum(v[3] for k, v in stats.stats.items() if k[2] == name) for name in names}


def force_case(config, out):
    started = time.perf_counter()
    family, n, chi = config.get('family', 'D'), config['n'], config.get('chi', 4)
    rho, ham, jumps, _ = previous.model(family, n)
    state = ts.matrix_to_mpo(fd.positive_power(rho, .25), n, chi)
    state, _ = ts.normalize_factor(state)
    hm = ts.matrix_to_mpo(ham, n)
    jm = [ts.matrix_to_mpo(j, n) for j in jumps]
    setup = time.perf_counter()-started
    solver = ts.factor_velocity
    if config['method'] == 'matrix_free':
        from factor_tdvp_accelerated.matrix_free_solver import factor_velocity
        solver = factor_velocity
    profiler = cProfile.Profile()
    tick = time.perf_counter(); profiler.enable()
    velocity, info = solver(state, hm, jm, work_bond=config['work_bond'])
    elapsed = time.perf_counter()-tick; profiler.disable()
    categories = profile_summary(profiler, out/'profile.txt')
    np.savez_compressed(out/'force.npz', factor=ts.matrix(state), velocity=ts.matrix(velocity))
    row = dict(config)
    row.update(status='completed', setup_seconds=setup,
                solve_seconds=elapsed, total_seconds=time.perf_counter()-started,
                peak_process_memory_mib=memory_mib(), profile_cumulative_seconds=categories,
                profile_categories_overlap=True, profiling_enabled=True)
    row.update(info)
    row['force_total_seconds'] = elapsed
    row['input_setup_seconds'] = setup
    return row


@contextlib.contextmanager
def accelerated_backend(tolerance=1e-8):
    """Worker-local injection only; the historical algorithms are unchanged."""
    from factor_tdvp_accelerated import matrix_free_solver as mf
    original = ts.factor_velocity
    diagnostics = []
    def solve(*args, **kwargs):
        result, info = mf.factor_velocity(*args, tolerance=tolerance, **kwargs)
        diagnostics.append(info)
        # The historical recursive aggregator must not treat an intermediate
        # LSQR residual as the final residual of an accepted force solve.
        clean = dict(info)
        clean['history'] = [{('iteration_force_residual' if k == 'force_residual' else k): v for k, v in item.items()} for item in info.get('history', [])]
        return result, clean
    ts.factor_velocity = solve
    try:
        yield diagnostics
    finally:
        ts.factor_velocity = original


def trajectory_case(config, out):
    method = config['method']
    arguments = dict(benchmark=config['family'], n=config['n'], steps=config['steps'],
                     chi=config['chi'], total=config['total'], work_bond=config['work_bond'], profile=False)
    if method == 'matrix_free':
        with accelerated_backend(config.get('solver_tolerance',1e-8)) as solves:
            row, records, profile = previous.run_case(method='factor_tdvp', **arguments)
    else:
        solves = []
        row, records, profile = previous.run_case(method='root_tdvp' if method == 'root' else 'factor_tdvp', **arguments)
    row.update(kind='trajectory', method=method, solver_tolerance=config.get('solver_tolerance',1e-8) if method=='matrix_free' else None, peak_process_memory_mib=memory_mib(),
               memory_measurement='isolated worker high-water RSS; platform units corrected',
               implementation_scope='MPO force with dense support audits and independent dense scoring' if method == 'matrix_free' else row['implementation_scope'])
    if solves:
        save(out/'solves.json', solves)
        row['max_krylov_iterations'] = max(s.get('iterations', s.get('krylov_iterations', 0)) for s in solves)
        row['max_stationarity_relative'] = max(s.get('ls_normal_relative', 0.) for s in solves)
        row['force_seconds'] = sum(s['seconds'] for s in solves)
        row['max_jacobian_residual'] = max(s['force_residual'] for s in solves)
        row['max_conditioning'] = max(s['condition_estimate'] for s in solves)
        row['conditioning_kind'] = 'LSQR bidiagonal estimate, not full Jacobian condition'
    save(out/'trajectory.json', records)
    if profile:
        (out/'profile.txt').write_text(profile)
    # Endpoint scoring is separate; the previous driver already times all
    # physical-reference generation and support audits in its own stages.
    row['certificate_kind'] = 'independent_exact_global_error_not_online_acceptance_certificate'
    return row


def witness_case(config, out):
    import scalable_cp_witness as native
    family, n, tolerance = config['family'], config['n'], config['tolerance']
    parity = family == 'E'
    times = [.02, .1, .3, .6, 1.] if parity else [.005, .02, .05, .1]
    if 'steps' in config:
        times = np.linspace(0., times[-1], config['steps']+1)[1:].tolist()
    levels = [(4,6,8),(6,8,12)] if parity else [(4,6,10),(6,8,12),(8,12,16)]
    predict = config['method'] == 'witness'
    # Measure independent-reference, certification, predictor and correction
    # components without changing native acceptance or correction arithmetic.
    timings = {'reference_seconds': 0., 'certificate_seconds': 0.,
               'prediction_seconds': 0., 'correction_seconds': 0.}
    originals = []
    def instrument(obj, attr, category):
        original = getattr(obj, attr)
        def wrapped(*args, **kwargs):
            tick = time.perf_counter()
            try: return original(*args, **kwargs)
            finally: timings[category] += time.perf_counter()-tick
        originals.append((obj, attr, original)); setattr(obj, attr, wrapped)
    instrument(native, 'channel', 'reference_seconds')
    instrument(native.WitnessAudit, 'endpoint', 'certificate_seconds')
    instrument(native, 'tdvp_prediction', 'prediction_seconds')
    instrument(native.WitnessAudit, 'correction', 'correction_seconds')
    started = time.perf_counter()
    try:
        success = native.run(n,times,levels,tolerance,out,8 if parity else 4,
                             predict=predict,parity=parity)
    finally:
        for obj, attr, original in originals: setattr(obj, attr, original)
    native_seconds = time.perf_counter()-started
    endpoint_files = [p for p in out.glob('*.csv') if not p.name.endswith('_trials.csv')]
    endpoints = list(csv.DictReader(endpoint_files[0].open()))
    endpoint = endpoints[-1]
    actual = float(endpoint['actual']) if endpoint['actual'] else None
    checkpoint = next(out.glob('*_last_accepted.npz'), None)
    density_error = normalization = None
    extra_started = time.perf_counter()
    if checkpoint and success:
        with np.load(checkpoint) as data:
            state = ts.tn.MPS(n,max_bond=int(endpoint['root_bond']))
            state.tensors = [data[f'x{j}'] for j in range(n)]; state.center=n-1
        root = ts.matrix(state)
        _, _, _, reference = previous.model(family,n)
        target = reference(times[-1])[1]
        density_error = float(np.linalg.norm(root@root.conj().T-target))
        normalization = float(abs(np.linalg.norm(root)**2-1))
    row = dict(config)
    row.update(status='completed' if success else 'failed_certificate',
                time=times[-1], reached_time=float(endpoint['time']),
                global_cp_error=actual if success else None,
                global_cp_bound=float(endpoint['bound']), density_error=density_error,
                normalization_defect=normalization, h_bond=int(endpoint['factor_bond']),
                x_bond=int(endpoint['root_bond']), total_seconds=time.perf_counter()-started,
                native_combined_seconds=native_seconds, scoring_seconds=time.perf_counter()-extra_started,
                peak_process_memory_mib=memory_mib(), time_grid=times,
                used_predictions=sum(r['used_prediction']=='True' for r in endpoints),
                certificate_kind='independent_density_inverse_metric_bound_float_evaluation',
                timing_categories_overlap=True,
                timing_note='reference calls may be nested in predictor/certificate; native time includes initialization, I/O and internal exact scoring',
                **timings)
    return row


def worker(config, out):
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        if config['kind']=='force': row=force_case(config,out)
        elif config['method'] in ('witness','correction_only'): row=witness_case(config,out)
        else: row=trajectory_case(config,out)
    except Exception as exc:
        import traceback
        (out/'exception.txt').write_text(traceback.format_exc())
        row=dict(**config,status='failed',error_message=f'{type(exc).__name__}: {exc}',
                 total_seconds=time.perf_counter()-started,peak_process_memory_mib=memory_mib(),global_cp_error=None)
    save(out/'summary.json',row)
    print(json.dumps(row),flush=True)


def configurations(suite):
    if suite == 'stress_ablation':
        yield dict(kind='trajectory',family='F_loss',n=3,chi=4,work_bond=4,method='explicit',steps=10,total=.03)
        for tolerance in (1e-8,1e-10,1e-12):
            yield dict(kind='trajectory',family='F_loss',n=3,chi=4,work_bond=4,method='matrix_free',steps=10,total=.03,solver_tolerance=tolerance)
    if suite in ('force','all'):
        for n in (3,4,6):
            for method in ('explicit','matrix_free'):
                yield dict(kind='force',family='D',n=n,chi=4,work_bond=4**(n//2),method=method)
    if suite in ('trajectories','all'):
        for family,total in (('A',1.),('B',1.),('C',1.),('E',1.),('F_loss',.03),('F_delayed',.08)):
            n=1 if family=='F_delayed' else 3
            for method in ('matrix_free','root'):
                yield dict(kind='trajectory',family=family,n=n,chi=1 if n==1 else 4,
                           work_bond=1 if n==1 else 4,method=method,steps=10,total=total)
        for n in (3,4):
            for steps in (5,10,20):
                for method in ('matrix_free','root'):
                    yield dict(kind='trajectory',family='D',n=n,chi=4 if n==3 else 16,
                               work_bond=4**(n//2),method=method,steps=steps,total=.1)
    if suite in ('matched_witness','all'):
        for steps in (5,10,20):
            for tolerance in (1e-3,1e-5,1e-7):
                for method in ('witness','correction_only'):
                    yield dict(kind='trajectory',family='D',n=3,method=method,tolerance=tolerance,steps=steps)
    if suite == 'witness':
        for family in ('D','E'):
            for tolerance in (1e-3,1e-5,1e-7):
                for method in ('witness','correction_only'):
                    yield dict(kind='trajectory',family=family,n=3,method=method,tolerance=tolerance)


def frontier(rows):
    result=[]
    for family,total in (('D',.1),('E',1.)):
        for method in ('root','matrix_free','witness','correction_only'):
            for target in (1e-3,1e-5,1e-7):
                candidates=[r for r in rows if r.get('kind')=='trajectory' and r.get('family',r.get('benchmark'))==family
                            and r['method']==method and r['n']==3 and r['status']=='completed'
                            and abs(r.get('time',r.get('total',-1))-total)<1e-12
                            and r.get('global_cp_error') is not None and r['global_cp_error']<=target]
                best=min(candidates,key=lambda r:r['total_seconds']) if candidates else None
                result.append(dict(family=family,n=3,time=total,method=method,target=target,
                                   attained=best is not None,total_seconds=best['total_seconds'] if best else None,
                                   global_cp_error=best['global_cp_error'] if best else None,
                                   run=best['run'] if best else None,
                                   selection='minimum observed cost; finite grid, unequal correction information'))
    return result


def run(suite,out,timeout):
    out.mkdir(parents=True,exist_ok=False)
    save(out/'environment.json',dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,
                                    platform=platform.platform(),blas_threads=1,
                                    solver_sha256=__import__('hashlib').sha256((ROOT/'factor_tdvp_accelerated/matrix_free_solver.py').read_bytes()).hexdigest(),
                                    command=sys.argv,timeout_per_worker_seconds=timeout))
    rows=[]
    for index,config in enumerate(configurations(suite)):
        name=f"{index:03d}_{config['kind']}_{config['family']}_n{config['n']}_{config['method']}"
        target=out/name;target.mkdir()
        command=[sys.executable,str(Path(__file__).resolve()),'--worker',json.dumps(config),'--out',str(target)]
        begun=time.perf_counter()
        with (target/'stdout.txt').open('w') as log:
            try:
                process=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
                if not (target/'summary.json').exists():
                    raise RuntimeError(f'worker exited {process.returncode} without result')
                row=json.loads((target/'summary.json').read_text())
            except (subprocess.TimeoutExpired,RuntimeError) as exc:
                row=dict(**config,status='timeout' if isinstance(exc,subprocess.TimeoutExpired) else 'failed',
                         error_message=str(exc),global_cp_error=None,timeout_seconds=timeout)
                save(target/'summary.json',row)
        row['subprocess_seconds']=time.perf_counter()-begun;row['run']=name
        rows.append(row);save(out/'measurements.json',rows);table(out/'measurements.csv',rows)
        save(out/'matched_accuracy.json',frontier(rows));table(out/'matched_accuracy.csv',frontier(rows))
        print(name,row['status'],row.get('global_cp_error'),flush=True)
    # Independent small-state velocity comparison; no inference from timings.
    checks=[]
    for n in (3,4,6):
        pair={r['method']:r for r in rows if r['kind']=='force' and r['n']==n and r['status']=='completed'}
        if len(pair)!=2:continue
        with np.load(out/pair['explicit']['run']/'force.npz') as data: a=data['velocity'].copy();h=data['factor'].copy()
        with np.load(out/pair['matrix_free']['run']/'force.npz') as data: b=data['velocity'].copy()
        checks.append(dict(n=n,relative_velocity_difference=float(np.linalg.norm(a-b)/max(np.linalg.norm(a),1e-300)),
                           relative_jacobian_image_difference=float(np.linalg.norm(ts.jacobian_dense(h,a-b))/max(np.linalg.norm(ts.jacobian_dense(h,a)),1e-300)),
                           force_speedup=pair['explicit']['force_total_seconds']/pair['matrix_free']['force_total_seconds']))
    save(out/'velocity_comparison.json',checks)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite',choices=['force','trajectories','witness','matched_witness','stress_ablation','all'],default='all')
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--timeout',type=float,default=120.)
    parser.add_argument('--worker',help='internal JSON worker configuration')
    args=parser.parse_args()
    if args.worker: worker(json.loads(args.worker),args.out)
    else: run(args.suite,args.out,args.timeout)
