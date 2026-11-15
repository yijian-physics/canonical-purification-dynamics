"""Run isolated benchmark workers and summarize their actual measurements.

Run from the repository root. No historical result file is overwritten.
Each worker is a separate process to isolate legacy module-name collisions.
"""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
HERE=Path(__file__).resolve().parent


def read_rows(path):
    data=json.loads(path.read_text())
    return data if isinstance(data,list) else data.get('rows',data.get('results',[]))


def write_csv(path,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n')
        writer.writeheader();writer.writerows(rows)


def observed_cost_frontier(rows,cost_scope='setup_plus_evolution'):
    """Best OBSERVED successful cost at each accuracy, within identical inputs.

    This is a finite grid search; it is not an optimized algorithmic complexity
    or interpolated runtime prediction. Oracle fits and upper bounds are excluded.
    Setup+evolution excludes independent validation/scoring work.
    """
    out=[]
    groups={}
    for row in rows:
        error=row.get('global_cp_error')
        if row.get('status')!='ok' or error is None or 'oracle' in row.get('method',''):
            continue
        if row.get('error_kind') not in ('measured_global','measured_global_and_bound','measured_global_and_analytic_bound',None):
            continue
        key=tuple(row.get(k) for k in ('benchmark','n','theta','time','method'))
        groups.setdefault(key,[]).append(row)
    for key,group in groups.items():
        for tolerance in (1e-3,1e-5,1e-7):
            candidates=[]
            for row in group:
                if row['global_cp_error']>tolerance: continue
                if cost_scope=='end_to_end':
                    cost=row.get('total_seconds')
                else:
                    setup,evolution=row.get('setup_seconds'),row.get('evolution_seconds')
                    cost=None if setup is None or evolution is None else setup+evolution
                if cost is None: continue
                candidates.append((cost,row))
            winner=min(candidates,key=lambda x:x[0]) if candidates else None
            achieved=any(r['global_cp_error']<=tolerance for r in group)
            result=dict(zip(('benchmark','n','theta','time','method'),key))
            result.update(target_cp_error=tolerance,status=('attained' if winner else ('attained_timing_breakdown_unavailable' if achieved else 'not_attained_in_measured_grid')),
                          cost_scope=cost_scope,**{cost_scope+'_seconds':winner[0] if winner else None},
                          global_cp_error=winner[1]['global_cp_error'] if winner else None,
                          steps=winner[1].get('steps') if winner else None,
                          window=winner[1].get('window') if winner else None,
                          h_bond=winner[1].get('h_bond') if winner else None,
                          x_bond=winner[1].get('x_bond') if winner else None)
            out.append(result)
    return out


def figures(rows,history,outdir,frontier):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    outdir.mkdir(parents=True,exist_ok=True)
    colors={'factor_tdvp':'#c44735','root_tdvp':'#2b6cb0','half_petz':'#16836b',
            'local_unitary':'#9655ad','local_unitary_regularized':'#9655ad',
            'local_unitary_unregularized_adapter':'#bd72a7','corrected_root_tdvp':'#a06d20',
            'positive_witness_tdvp':'#b87b17','unitary_oracle':'#555555'}
    valid=[r for r in rows if r.get('status')=='ok']
    def value(r,key):
        v=r.get(key)
        return v if isinstance(v,(int,float)) and np.isfinite(v) else None
    def points(ax,subset,x,y,group=('method',),logx=False,logy=True):
        groups={}
        for r in subset:
            a,b=value(r,x),value(r,y)
            if a is None or b is None or (logx and a<=0) or (logy and b<=0): continue
            key=tuple(r.get(k) for k in group);groups.setdefault(key,[]).append((a,b))
        for key,pairs in groups.items():
            pairs=sorted(pairs);a,b=zip(*pairs)
            ax.plot(a,b,'o-',ms=4,lw=1,label=' / '.join(map(str,key)),color=colors.get(key[0]))
        if logx: ax.set_xscale('log')
        if logy: ax.set_yscale('log')
        ax.grid(alpha=.2);ax.set_xlabel(x.replace('_',' '));ax.set_ylabel(y.replace('_',' '))
        if groups: ax.legend(fontsize=6)
        else: ax.text(.5,.5,'No comparable measurements',ha='center',transform=ax.transAxes)
    fig,axes=plt.subplots(4,2,figsize=(13,17),layout='constrained')
    ab=[r for r in valid if r.get('benchmark')=='A' and r.get('dt')==.05
        and r.get('chi_x_limit',4)==4 and r.get('chi',4)==4 and not r.get('resource_control')]
    points(axes[0,0],ab,'n','global_cp_error',logx=True)
    axes[0,0].set_title('A: cluster, T=1, h=0.05 (actual global errors)')
    traj=[r for r in history if r.get('benchmark')=='A' and r.get('n')==3 and abs(r.get('dt',.05)-.05)<1e-10]
    points(axes[0,1],traj,'time','global_cp_error');axes[0,1].set_title('A: matched three-spin time histories')
    bonds=[dict(r,reported_bond=r.get('h_bond') if r.get('method')=='factor_tdvp' else r.get('x_bond'))
           for r in valid if r.get('benchmark') in ('F_critical','F') and r.get('n') in (3,8)]
    points(axes[1,0],bonds,'reported_bond','global_cp_error',group=('method','benchmark','n'))
    axes[1,0].set_title('Critical Ising: H and X budgets are distinct')
    steps=[r for r in valid if r.get('benchmark')=='A' and r.get('n') in (3,4,6)
           and r.get('method') in ('factor_tdvp','root_tdvp','half_petz') and r.get('chi_x_limit',4)==4]
    points(axes[1,1],steps,'dt','global_cp_error',group=('method','n'),logx=True)
    axes[1,1].set_title('A: time refinement (different N shown separately)')
    points(axes[2,0],ab,'n','evolution_seconds',logx=True);axes[2,0].set_title('A: evolution time; excludes scoring')
    costs=[r for r in frontier if r['status']=='attained' and r['benchmark']=='A' and r['n']==3]
    points(axes[2,1],costs,'target_cp_error','setup_plus_evolution_seconds',logx=True)
    axes[2,1].set_title('A, N=3: best observed cost at target error')
    factors=[r for r in valid if r.get('method')=='factor_tdvp']
    # Workers retain their native residual naming; make the plot explicit.
    for r in factors:
        r['jacobian_residual_plot']=r.get('max_force_residual',r.get('jacobian_residual',r.get('force_residual')))
    points(axes[3,0],factors,'jacobian_residual_plot','global_cp_error',group=('benchmark',),logx=True)
    axes[3,0].set_title('Factor residual versus endpoint error; not a certificate')
    diagnostics=[dict(r,conditioning_plot=r.get('max_condition',r.get('max_ls_retained_condition',r.get('condition_number')))) for r in factors]
    points(axes[3,1],diagnostics,'conditioning_plot','global_cp_error',group=('benchmark',),logx=True)
    axes[3,1].set_title('Conditioning versus endpoint error')
    for ax in axes[3]:
        for line in ax.lines: line.set_linestyle('None')
    fig.savefig(outdir/'comparison.png',dpi=160);fig.savefig(outdir/'comparison.svg');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4.8),layout='constrained')
    for ax,n in zip(axes,(4,6,8)):
        c=[r for r in valid if r.get('benchmark')=='C' and r.get('n')==n and r.get('window',3)==3
           and (r.get('method') not in ('factor_tdvp','root_tdvp') or r.get('steps') in (4,20))]
        points(ax,c,'theta','global_cp_error',group=('method',))
        ax.set_title(f'C, N={n}: width-three lifts'+('\nTDVP 20 steps' if n==4 else ('\nFactor 4 steps' if n==6 else '\nHistorical control reproduction')))
    fig.savefig(outdir/'tilted.png',dpi=160);fig.savefig(outdir/'tilted.svg');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for ax,benchmark,tolerance in zip(axes,('D','E'),(1e-5,1e-6)):
        subset=[r for r in valid if r.get('benchmark')==benchmark and r.get('n',0)>=8
                and r.get('method')=='positive_witness_tdvp' and r.get('endpoint_tolerance')==tolerance]
        measured=[r for r in subset if r.get('global_cp_error') is not None]
        bound=[r for r in subset if r.get('global_cp_bound') is not None]
        if measured: ax.loglog([r['n'] for r in measured],[r['global_cp_error'] for r in measured],'o-',label='Measured global error')
        if bound: ax.loglog([r['n'] for r in bound],[r['global_cp_bound'] for r in bound],'s--',label='Analytic bound, float evaluation')
        ax.set_title(f'{benchmark}: positive-witness baseline');ax.set_xlabel('N');ax.set_ylabel('Global CP error / bound');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(outdir/'witness_scaling.png',dpi=160);fig.savefig(outdir/'witness_scaling.svg');plt.close(fig)


def summarize(outdir):
    import platform
    import numpy
    import scipy
    import matplotlib
    rows=[];history=[]
    for stem in ('factor_measurements','factor_extended','lifts','witness_measurements','witness_matched_measurements'):
        path=outdir/(stem+'.json')
        if path.exists(): rows+=read_rows(path)
    for stem in ('factor','lifts'):
        for suffix in ('_history','_trajectories'):
            path=outdir/(stem+suffix+'.json')
            if path.exists():
                data=read_rows(path)
                history+=data
    if not rows: raise RuntimeError('No worker measurement files found')
    for row in rows:
        row['native_status']=row.get('status')
        if row.get('status') in ('success','completed'): row['status']='ok'
        if 'error_kind' not in row: row['error_kind']=row.get('metric_kind')
        if row.get('benchmark')!='C': row['theta']=None
        if row.get('method')=='factor_tdvp':
            row['max_force_residual']=row.get('max_jacobian_residual',row.get('max_force_residual'))
            row['max_ls_retained_condition']=row.get('max_conditioning',row.get('max_ls_retained_condition'))
    (outdir/'comparison.json').write_text(json.dumps(rows,indent=2)+'\n')
    write_csv(outdir/'comparison.csv',rows)
    environment=dict(python=platform.python_version(),numpy=numpy.__version__,scipy=scipy.__version__,matplotlib=matplotlib.__version__,
                     base_commit='4af4c9f9d9f5dbc03e896508592563abb2a74839',
                     recovered_factor_commit='a93549e2cc7a962ca686ac8b377dd815c92e102d',
                     repository='xiaoyanghuangphys-create/canonical-purification-dynamics',
                     timing_caveat='Single samples; some workers ran concurrently. Profiling adds overhead. These are observed wall times, not isolated-machine speed guarantees.')
    (outdir/'environment.json').write_text(json.dumps(environment,indent=2)+'\n')
    requested={'A':(6,16,64,128,256),'B':(6,16,64,128,256),
               'C':(6,8,32,64,256),'D':(8,48,64,128,256,512),
               'E':(8,128,512),'F':(8,), 'G':(64,128)}
    coverage=[]
    for benchmark,sizes in requested.items():
        for n in sizes:
            observed=[r for r in rows if r.get('benchmark')==benchmark and r.get('n')==n]
            factor=[r for r in observed if r.get('method')=='factor_tdvp']
            coverage.append(dict(benchmark=benchmark,n=n,
                factor_status=('measured' if any(r.get('status')=='ok' for r in factor) else ('attempt_failed' if factor else ('force_profile_only' if benchmark=='A' and n==6 and (outdir/'factor_n6_profile.json').exists() else 'not_run'))),
                other_methods_measured=','.join(sorted({r.get('method','') for r in observed})),
                reason=None if factor else ('Optional local-observable controls deferred; historical local errors are not global.' if benchmark=='G' else 'Explicit dense Jacobian and dense support bridge exceed practical run budget; see measured small-N profiles.'),
                estimated_real_jacobian_rows=str(2*4**n),
                estimated_chi4_real_columns=max(8,96*n-160),
                estimated_chi4_design_bytes=str(8*(2*4**n)*max(8,96*n-160)),
                estimate_note='Algebraic storage estimate at uniform actual H bond4, not measured memory or runtime.'))
    write_csv(outdir/'requested_size_coverage.csv',coverage)
    frontier=observed_cost_frontier(rows)
    write_csv(outdir/'matched_accuracy.csv',frontier)
    write_csv(outdir/'matched_accuracy_end_to_end.csv',observed_cost_frontier(rows,'end_to_end'))
    figures(rows,history,outdir.parent/'figures',frontier)
    # Normalize generated text formatting without changing numerical values.
    for path in list(outdir.rglob('*.csv'))+list((outdir.parent/'figures').glob('*.svg')):
        path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    print(f'Summarized {len(rows)} recorded rows, {len(history)} trajectory points, {len(frontier)} target comparisons.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--component',choices=('all','factor','lifts','witness','summarize'),default='all')
    parser.add_argument('--outdir',type=Path,default=HERE/'results')
    args=parser.parse_args();args.outdir.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    components=('factor','lifts','witness') if args.component=='all' else (args.component,)
    for component in components:
        if component=='summarize': continue
        script={'factor':'factor_benchmarks.py','lifts':'lift_benchmarks.py','witness':'witness_benchmarks.py'}[component]
        with (args.outdir/(component+'_stdout.txt')).open('w') as log:
            completed=subprocess.run([sys.executable,str(HERE/script),'--outdir',str(args.outdir)],
                                      cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        if completed.returncode:
            print(f'{component} exited {completed.returncode}; inspect its log; keeping other results.',file=sys.stderr)
        if component=='witness':
            with (args.outdir/'witness_matched_stdout.txt').open('w') as log:
                subprocess.run([sys.executable,str(HERE/script),'--matched','--outdir',str(args.outdir)],
                               cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    summarize(args.outdir)


if __name__=='__main__': main()
