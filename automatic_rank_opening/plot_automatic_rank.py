from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse


def read(path):
    with path.open() as f:return list(csv.DictReader(f))


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,default=Path(__file__).parent/'automatic_rank_results');p.add_argument('--output',type=Path,default=Path(__file__).parent/'summary.png');args=p.parse_args()
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(14,4.3),layout='constrained')
    rows=read(args.directory/'opening_scaling.csv')
    for case,label,color in [('pure_plus_loss','Ordinary opening: slope 1/2','#2765A6'),('Hamiltonian_activated_loss','Delayed loss: slope 5/2','#CF6631')]:
        data=sorted([r for r in rows if r['case']==case],key=lambda r:float(r['step']))
        axes[0].loglog([float(r['step']) for r in data],[float(r['finite_probe_tail']) for r in data],'o-',label=label,color=color)
    axes[0].set(xlabel='Probe step',ylabel='New singular-value tail',title='Detecting different opening orders');axes[0].legend(frameon=False,fontsize=9)
    for budget,color in [(1e-3,'#2765A6'),(1e-4,'#CF6631')]:
        data=[]
        for path in args.directory.glob('n*_product_budget*.csv'):
            row=read(path)[0]
            if float(row['cp_budget'])==budget:data.append(row)
        data=sorted(data,key=lambda r:int(r['n']))
        axes[1].loglog([int(r['n']) for r in data],[float(r['step']) for r in data],'o-',label=f'CP budget {budget:g}',color=color)
    axes[1].set(xlabel='Spin count',ylabel='Automatically chosen startup step',title='Finite startup through 64 spins');axes[1].set_xticks([4,16,32,64],labels=['4','16','32','64']);axes[1].legend(frameon=False,fontsize=9)
    rows=read(args.directory/'evolution.csv')
    for case,label,color in [('pure_plus_loss','4-spin interacting loss','#2765A6'),('Hamiltonian_activated_loss','Hamiltonian-activated loss','#CF6631'),('near_boundary_full_rank','Near-boundary full-rank input','#29836D')]:
        data=sorted([r for r in rows if r['case']==case],key=lambda r:float(r['tolerance']))
        axes[2].loglog([float(r['tolerance']) for r in data],[float(r['cp_error']) for r in data],'o-',color=color,label=label)
    axes[2].set(xlabel='Requested integration tolerance',ylabel='Measured final global CP error',title='Automatic bridge and continuation');axes[2].legend(frameon=False,fontsize=8)
    for ax in axes:ax.grid(True,alpha=.2)
    fig.suptitle('Rank opening: detect support leakage, take a positive step, check the handoff',fontsize=13)
    fig.savefig(args.output,dpi=170);fig.savefig(args.output.with_suffix('.pdf'));plt.close(fig)


if __name__=='__main__':main()
