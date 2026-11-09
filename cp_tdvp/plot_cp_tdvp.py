from pathlib import Path
import argparse
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator,NullLocator,FormatStrFormatter
import numpy as np


def read(path):
    with path.open(newline='',encoding='utf-8') as stream:return list(csv.DictReader(stream))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--directory',type=Path,default=Path(__file__).parent)
    root=parser.parse_args().directory;data=root/'results'
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(11.2,8),layout='constrained')
    dense=read(data/'dense.csv')
    for model,label in [('ising_loss','Interacting Ising + loss, N=4'),('pair','Pair dephasing, N=4')]:
        selected=[r for r in dense if r['model']==model and r['n']=='4' and r['chi']=='16' and r['branch']=='polar']
        axes[0,0].loglog([float(r['du']) for r in selected],[float(r['cp_error']) for r in selected],'o-',label=label)
    selected=[r for r in read(data/'sector.csv') if r['n']=='32']
    axes[0,0].loglog([float(r['du']) for r in selected],[float(r['cp_error']) for r in selected],'s-',label='One-excitation transport, N=32')
    axes[0,0].set(title='A. Integrated TDVP convergence',xlabel=r'Step in $u=\sqrt{t}$',ylabel='Global CP error at t=1')
    axes[0,0].legend(fontsize=8.5)
    clock=read(data/'clock.csv')
    for mode,label in [('physical','Uniform physical-time grid'),('sqrt','Uniform square-root-time grid')]:
        selected=[r for r in clock if r['clock']==mode]
        axes[0,1].loglog([1/int(r['steps']) for r in selected],[float(r['cp_error']) for r in selected],'o-',label=label)
    axes[0,1].set(title='B. Resolving the initial square root',xlabel='Inverse step count, 1/M',ylabel='CP error: uniform loss, N=16')
    axes[0,1].legend(fontsize=9)
    for chi in ('2','4','8','16'):
        selected=[r for r in dense if r['model']=='ising_loss' and r['n']=='4' and r['chi']==chi and r['branch']=='polar']
        axes[1,0].loglog([float(r['du']) for r in selected],[float(r['cp_error']) for r in selected],'o-',label=f'Bond cap {chi}')
    axes[1,0].set(title='C. Time accuracy and bond accuracy differ',xlabel=r'Step in $u=\sqrt{t}$',ylabel='CP error: interacting Ising + loss')
    axes[1,0].legend(fontsize=9)
    adaptive=read(data/'adaptive_time.csv');locations=np.arange(len(adaptive))
    for shift,key,label in [(-.25,'cp_error','Measured CP error'),(0.,'cp_step_defect_bound','Channel-defect bound'),(.25,'cp_error_bound','Density-residual bound')]:
        axes[1,1].bar(locations+shift,[float(r[key]) for r in adaptive],width=.24,label=label)
    axes[1,1].set_yscale('log');axes[1,1].set_xticks(locations,[r['tolerance'] for r in adaptive])
    axes[1,1].set(title='D. Adaptive integration and global bounds',xlabel='Requested integration tolerance',ylabel='CP error or upper bound')
    axes[1,1].legend(fontsize=8.5)
    for ax in [axes[0,0],axes[0,1],axes[1,0]]:
        ax.xaxis.set_major_locator(FixedLocator([.0125,.025,.05,.1,.2]));ax.xaxis.set_major_formatter(FormatStrFormatter('%g'));ax.xaxis.set_minor_locator(NullLocator())
    for ax in axes.flat:ax.grid(alpha=.15);ax.set_axisbelow(True)
    fig.savefig(root/'summary.png',dpi=180);fig.savefig(root/'summary.svg')


if __name__=='__main__':main()
