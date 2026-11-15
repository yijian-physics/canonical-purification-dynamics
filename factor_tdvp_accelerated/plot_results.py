"""Combine completed benchmark runs and plot measured costs, no extrapolation."""
from __future__ import annotations
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/tdvp-matplotlib')
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from factor_tdvp_accelerated.run_benchmarks import frontier, save, table


def create(result_dirs, output):
    output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for folder in result_dirs:
        for row in json.loads((folder/'measurements.json').read_text()):
            row['run']=str(folder/row['run'])
            rows.append(row)
    save(output/'combined_measurements.json',rows)
    table(output/'combined_measurements.csv',rows)
    matched=frontier(rows)
    save(output/'matched_accuracy.json',matched);table(output/'matched_accuracy.csv',matched)
    colors={'root':'#315a9c','matrix_free':'#c14953','witness':'#478c5c','correction_only':'#8b5cab','explicit':'#65717e'}
    names={'root':'Conventional root','matrix_free':'Factor LSQR','witness':'Root + witness','correction_only':'Correction only','explicit':'Explicit factor'}
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':130})
    fig,ax=plt.subplots(figsize=(6.4,4.4))
    for method in ('root','matrix_free','witness','correction_only'):
        selected=[r for r in rows if r.get('kind')=='trajectory' and r.get('family',r.get('benchmark'))=='D' and r['n']==3 and r['method']==method and r['status']=='completed']
        ax.scatter([r['global_cp_error'] for r in selected],[r['total_seconds'] for r in selected],label=names[method],color=colors[method],s=33)
    ax.set(xscale='log',yscale='log',xlabel='Measured global canonical-root error',ylabel='End-to-end worker time (s)',title='D, 3 spins, t = 0.1; measured parameter grid')
    ax.legend(fontsize=8);ax.grid(alpha=.2,which='both');fig.tight_layout();fig.savefig(output/'accuracy_runtime.png');plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(9,3.8))
    for method in ('explicit','matrix_free'):
        selected=sorted([r for r in rows if r['kind']=='force' and r['method']==method and r['status']=='completed'],key=lambda r:r['n'])
        axs[0].plot([r['n'] for r in selected],[r.get('force_total_seconds',r['seconds']) for r in selected],'o-',label=names[method],color=colors[method])
        axs[1].plot([r['n'] for r in selected],[r['peak_process_memory_mib'] for r in selected],'o-',label=names[method],color=colors[method])
    axs[0].set(yscale='log',ylabel='Profiled complete force time (s)',xlabel='Spins N');axs[1].set(ylabel='Worker peak RSS (MiB)',xlabel='Spins N')
    for ax in axs:ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('D initial factor, bond 4; exact finite-chain work caps')
    fig.tight_layout();fig.savefig(output/'force_scaling.png');plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(9,3.8))
    selected=[r for r in rows if r['kind']=='force' and r['method']=='matrix_free' and r['status']=='completed']
    for r in selected:
        hist=r.get('history',[])
        axs[0].semilogy([h['iteration'] for h in hist],[max(h.get('normal_relative',h.get('normal_relative_estimate',1.)),1e-18) for h in hist],label=f"N={r['n']}")
        axs[1].plot([h['iteration'] for h in hist],[h['condition_estimate'] for h in hist],label=f"N={r['n']}")
    axs[0].set(xlabel='LSQR iteration',ylabel='Relative stationarity (estimated between checks)');axs[1].set(xlabel='LSQR iteration',ylabel='Bidiagonal condition estimate')
    for ax in axs:ax.grid(alpha=.2);ax.legend()
    fig.suptitle('Solver diagnostics; estimates are not full Jacobian condition numbers')
    fig.tight_layout();fig.savefig(output/'conditioning.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.4,4.4))
    for n in (3,4):
        for method in ('root','matrix_free'):
            selected=sorted([r for r in rows if r.get('kind')=='trajectory' and r.get('family',r.get('benchmark'))=='D' and r['n']==n and r['method']==method and r['status']=='completed'],key=lambda r:r['steps'])
            ax.loglog([r['dt'] for r in selected],[r['global_cp_error'] for r in selected],'o-' if n==3 else 's--',label=f"{names[method]}, N={n}",color=colors[method])
    ax.set(xlabel='Time step',ylabel='Global canonical-root error',title='D smooth full-rank refinement, t = 0.1')
    ax.grid(alpha=.2,which='both');ax.legend(fontsize=8);fig.tight_layout();fig.savefig(output/'step_convergence.png');plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('results',nargs='+',type=Path);p.add_argument('--out',required=True,type=Path);a=p.parse_args();create(a.results,a.out)
