"""Reproduce native positive-witness and corrected critical-root benchmarks.

The existing algorithms are called unchanged in isolated Python subprocesses.
Native timers combine setup, prediction, certificate/correction and scoring;
they are not mislabeled as evolution-only timings. Scoring added here is timed
separately. All checkpoints and CSVs are written under the requested output.
"""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / 'scalable_cp_witness' / 'code'


def _save(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_suffix('.json').write_text(json.dumps(rows, indent=2)+'\n')
    if rows:
        with path.with_suffix('.csv').open('w', newline='') as f:
            keys = list(dict.fromkeys(k for row in rows for k in row))
            w = csv.DictWriter(f, fieldnames=keys, lineterminator='\n')
            w.writeheader(); w.writerows(rows)


def _worker(family, n, out, tolerance=1e-5, bond=64):
    sys.path.insert(0, str(CODE))
    import numpy as np
    import tdvp_core as tn
    import mpo_cp_force as mpo
    import scalable_cp_witness as native
    out = Path(out).resolve(); out.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    if family == 'F':
        import cp_accuracy_refinement as critical
        critical.OUT = out
        critical.trajectory(n, bond, 8, True)
        native_seconds = perf_counter()-started
        endpoint = list(csv.DictReader((out/f'trajectory_n{n}_chi{bond}_s8_correct1.csv').open()))[-1]
        row = dict(benchmark='F', method='corrected_root_tdvp', n=n,
                   status='success', global_cp_error=float(endpoint['cp_error']),
                   global_cp_bound=float(endpoint['cp_bound']), error_kind='measured_global_and_analytic_bound',
                   density_error=None, time=.3, dt=None, seed_time=.01, steps=8,
                   h_bond=None, x_bond=bond, x_bond_limit=bond,
                   accepted_steps=8, rejected_steps=0, rank_events=1,
                   startup_mode='fixed_positive_channel_seed_not_automatic_detection',
                   setup_seconds=None, evolution_seconds=None, scoring_seconds=None,
                   native_combined_seconds=native_seconds, native_includes_scoring=True,
                   timing_scope='setup_evolution_correction_and_scoring_combined',
                   command=f'python scalable_cp_witness/code/cp_accuracy_refinement.py trajectory --n {n} --bond {bond} --intervals 8 --out {out.relative_to(ROOT)}')
    else:
        parity = family == 'E'
        times = [.02,.1,.3,.6,1.] if parity else [.005,.02,.05,.1]
        levels = [(4,6,8),(6,8,12)] if parity else [(4,6,10),(6,8,12),(8,12,16)]
        passes = 8 if parity else 4
        success = native.run(n,times,levels,tolerance,out,passes,predict=True,parity=parity)
        native_seconds = perf_counter()-started
        files = [p for p in out.glob('*.csv') if not p.name.endswith('_trials.csv')]
        endpoints = list(csv.DictReader(files[0].open()))
        attempted_endpoint = endpoints[-1]
        checkpoint = next(out.glob('*_last_accepted.npz'))
        scoring_started = perf_counter()
        with np.load(checkpoint) as data:
            tensors = [data[f'x{j}'] for j in range(n)]
            factor_tensors = [data[f'g{j}'] for j in range(n)]
            elapsed = float(data['time'])
        endpoint = next(row for row in reversed(endpoints)
                        if row['accepted']=='True' and abs(float(row['time'])-elapsed)<1e-12)
        state = tn.MPS(n,max_bond=max(a.shape[-1] for a in tensors))
        state.tensors=tensors; state.center=n-1
        factor = tn.MPS(n,max_bond=max(a.shape[-1] for a in factor_tensors))
        factor.tensors=factor_tensors; factor.center=n-1
        target_density = native.parity_power(n,1.,.6,.4*np.exp(-elapsed)) if parity else native.channel(native.prepared_power(n,1.),elapsed)
        produced_density = mpo.multiply(state,mpo.adjoint(state))
        density_error = tn.distance(produced_density,target_density)
        normalization_error = abs(tn.norm(state)**2-1)
        branch_defect = None
        if n <= 8:
            a=native.matrix(state)
            u,s,_=np.linalg.svd(a,full_matrices=False)
            branch_defect=float(np.linalg.norm(a-(u*s)@u.conj().T))
        cp_error = None if endpoint['actual']=='' else float(endpoint['actual'])
        scoring_seconds = perf_counter()-scoring_started
        row = dict(benchmark=family, method='positive_witness_tdvp', n=n,
                   status='success' if success else 'failed_certificate',
                   attempted_final_bound=float(attempted_endpoint['bound']),
                   attempted_final_time=float(attempted_endpoint['time']),
                   global_cp_error=cp_error, global_cp_bound=float(endpoint['bound']),
                   error_kind='measured_global_and_analytic_bound' if cp_error is not None else 'analytic_global_bound_float_evaluation',
                   density_error=float(density_error), normalization_defect=float(normalization_error),
                   canonical_branch_defect=branch_defect,
                   positivity_witness_distance=float(endpoint['witness']),
                   time=elapsed, requested_time=times[-1], dt=None, time_grid=times,
                   steps=len(times), accepted_steps=sum(x['accepted']=='True' and float(x['time'])>0 for x in endpoints),
                   rejected_steps=sum(x['accepted']!='True' for x in endpoints),rank_events=0,
                   h_bond=factor.bond_dimension(),
                   factor_role='separately_fitted_witness_not_evolving_factor',
                   x_bond=state.bond_dimension(),
                   x_bond_max=max(int(x['root_bond']) for x in endpoints),
                   x_bond_average=float(np.mean([int(x['root_bond']) for x in endpoints])),
                   witness_bond_max=max(int(x['factor_bond']) for x in endpoints),
                   setup_seconds=None, evolution_seconds=None, scoring_seconds=scoring_seconds,
                   native_combined_seconds=native_seconds, native_includes_scoring=True,
                   timing_scope='native_setup_evolution_certification_scoring_plus_extra_scoring',
                   endpoint_tolerance=tolerance,
                   command=f'python scalable_cp_witness/code/scalable_cp_witness.py --n {n} --tdvp '+ ('--parity ' if parity else '')+f'--times '+ ' '.join(map(str,times))+f' --tol {tolerance} --levels '+':'.join(','.join(map(str,l)) for l in levels)+f' --passes {passes} --out {out.relative_to(ROOT)}')
    row['worker_total_seconds']=perf_counter()-started
    row['peak_rss_mb']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    # Raw numerical arrays are startup/checkpoint intermediates, not deliverables.
    for checkpoint in out.glob('*.npz'):
        checkpoint.unlink()
    (out/'summary.json').write_text(json.dumps(row,indent=2)+'\n')
    print(json.dumps(row),flush=True)
    return row


def run(outdir=None, timeout=240., small=False, matched=False):
    out = Path(outdir or ROOT/'factor_tdvp_comparison'/'results').resolve()
    out.mkdir(parents=True, exist_ok=True)
    configs = [('D',8,1e-5,64),('E',8,1e-6,64)]
    if not small:
        configs += [('D',64,1e-5,64),('D',512,1e-5,64),('E',128,1e-6,64),('E',512,1e-6,64),('F',8,1e-5,64),('F',8,1e-5,128)]
    if matched:
        configs = [(family,3,tol,64) for family in ('D','E') for tol in (1e-3,1e-5,1e-7)]
    rows=[]
    for family,n,tol,bond in configs:
        tag=f'{family}_n{n}_tol{tol:g}_bond{bond}'
        raw=out/'witness_raw'/tag; raw.mkdir(parents=True,exist_ok=True)
        command=[sys.executable,str(Path(__file__).resolve()),'--worker',family,'--n',str(n),
                 '--tolerance',str(tol),'--bond',str(bond),'--out',str(raw)]
        started=perf_counter()
        environment=dict(os.environ,OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='1')
        with (raw/'stdout.txt').open('w') as log:
            try:
                process=subprocess.run(command,cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
                elapsed=perf_counter()-started
                if process.returncode != 0 or not (raw/'summary.json').exists():
                    raise RuntimeError(f'worker exited with code {process.returncode}')
                row=json.loads((raw/'summary.json').read_text())
            except (subprocess.TimeoutExpired,RuntimeError) as exc:
                elapsed=perf_counter()-started
                row=dict(benchmark=family,method='corrected_root_tdvp' if family=='F' else 'positive_witness_tdvp',
                         n=n,status='timeout' if isinstance(exc,subprocess.TimeoutExpired) else 'failed',
                         reason=str(exc),x_bond_limit=bond,global_cp_error=None,global_cp_bound=None)
        row['total_seconds']=elapsed
        row['log']=str((raw/'stdout.txt').relative_to(ROOT))
        rows.append(row)
        _save(out/('witness_matched_measurements' if matched else 'witness_measurements'),rows)
        print(json.dumps(row),flush=True)
    if not matched:
        validate_reproductions(out)
    return rows


def validate_reproductions(out):
    """Compare actual reruns with exact rows in the untouched historical CSVs."""
    out=Path(out)
    rows=json.loads((out/'witness_measurements.json').read_text())
    comparisons=[
        ('D',8,6,'n8_t0.3_s5_p0.12_a0.35_jensen_tdvp.csv',.1,'actual','global_cp_error'),
        ('D',512,8,'n512_t0.1_s4_p0.12_a0.35_jensen_tdvp.csv',.1,'bound','global_cp_bound'),
        ('E',512,2,'n512_t1_s5_parity0.6_x0.4_jensen_tdvp.csv',1.,'actual','global_cp_error'),
        ('F',8,64,'trajectory_n8_chi64_s8_correct1.csv',.3,'cp_error','global_cp_error'),
        ('F',8,128,'trajectory_n8_chi128_s8_correct1.csv',.3,'cp_error','global_cp_error')]
    checks=[]
    for family,n,bond,filename,time,field,key in comparisons:
        rerun=next((r for r in rows if r['benchmark']==family and r['n']==n
                    and r.get('x_bond')==bond and r['status']=='success'),None)
        if rerun is None:
            continue
        source=ROOT/'scalable_cp_witness'/filename
        historical=next(r for r in csv.DictReader(source.open()) if abs(float(r['time'])-time)<1e-12)
        reference=float(historical[field]); actual=rerun[key]
        difference=abs(actual-reference)
        checks.append(dict(benchmark=family,n=n,bond=bond,reference=reference,
                           reference_file=str(source.relative_to(ROOT)),reference_field=field,
                           rerun=actual,absolute_deviation=difference,tolerance=1e-10,
                           passed=difference<1e-10))
    (out/'witness_validation.json').write_text(json.dumps(checks,indent=2)+'\n')
    return checks


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=None)
    parser.add_argument('--small',action='store_true')
    parser.add_argument('--matched',action='store_true',help='N=3 tolerance sweeps matching the small factor controls')
    parser.add_argument('--timeout',type=float,default=240.)
    parser.add_argument('--worker',choices=['D','E','F'])
    parser.add_argument('--n',type=int,default=8)
    parser.add_argument('--bond',type=int,default=64)
    parser.add_argument('--tolerance',type=float,default=1e-5)
    args=parser.parse_args()
    if args.worker:
        _worker(args.worker,args.n,args.out,args.tolerance,args.bond)
    else:
        run(args.out,args.timeout,args.small,args.matched)

if __name__=='__main__':
    main()
