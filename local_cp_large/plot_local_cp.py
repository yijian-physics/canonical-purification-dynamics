"""Regenerate the summary figure from the bundled CSV results."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
import numpy as np

root=Path(__file__).resolve().parent
def read(name):
    with (root/'results'/name).open(newline='',encoding='utf-8') as f:
        return list(csv.DictReader(f))

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                     'axes.titleweight':'bold','savefig.facecolor':'white'})
fig,axes=plt.subplots(1,3,figsize=(14.6,4.7),layout='constrained')
colors=['#3569a8','#d17825','#258574']
scaling=read('scaling.csv')
for radius,color in zip((1,2),colors):
    rows=[r for r in scaling if int(r['radius'])==radius]
    n=np.array([int(r['n']) for r in rows])
    elapsed=np.array([float(r['update_seconds']) for r in rows])
    axes[0].plot(n,elapsed,'o-',color=color,label=f'Guard {radius}; active width {1+2*radius}')
axes[0].plot(n,elapsed[-1]*n/n[-1],':',color='#777777',label='Linear reference')
axes[0].set(xscale='log',yscale='log',xlabel='Chain length N',ylabel='Two updates (seconds)',
            title='A  Local evolution scales with N')
axes[0].set_xticks([32,64,128,256],labels=['32','64','128','256'])
axes[0].xaxis.set_minor_locator(NullLocator())
axes[0].legend(frameon=False,fontsize=8.5,loc='upper left')

rows=read('classical.csv')
row=next(r for r in rows if r['n']=='64' and r['radius']=='2' and r['buffer']=='1' and r['chi']=='16' and r['sweep']=='2')
b=np.array([2,4,6,8,10])
bound=np.minimum(1,[float(row[f'certificate_b{x}']) for x in b])
axes[1].semilogy(b,bound,'o-',color=colors[1],label='Local CP certificate')
axes[1].axhline(float(row['cp_local_error']),color=colors[0],linestyle='--',label='24-site reference discrepancy')
axes[1].annotate(r'$5.38\times10^{-4}$',xy=(10,bound[-1]),xytext=(7.2,1.6e-3),fontsize=10)
axes[1].set(xlabel='Certification buffer on each side',ylabel='Two-site doubled trace distance',
            title='B  A bound uniform in chain length',xticks=b,ylim=(5e-6,2))
axes[1].legend(frameon=False,fontsize=8.5,loc='upper right')

quantum=read('precise_quantum.csv')
for n,disorder,color,marker,label in [(128,'False',colors[0],'o','Uniform, N=128'),(64,'True',colors[2],'s','Disordered, N=64')]:
    rows=[r for r in quantum if int(r['n'])==n and r['disorder']==disorder]
    t=[.1*int(r['sweep']) for r in rows]
    axes[2].plot(t,[float(r['local_cp_residual_b3']) for r in rows],marker+'-',color=color,label=label+'; CP residual')
    axes[2].plot(t,[float(r['physical_error']) for r in rows],marker+'--',color=color,label=label+'; physical error')
axes[2].set(xlabel='Dephasing time',ylabel='Two-site trace distance',yscale='log',
            title='C  Mixed-field quantum Ising',ylim=(2e-6,1.7e-5),xticks=[.2,.5,1.])
axes[2].legend(frameon=False,fontsize=8,loc='lower right')
for ax in axes:
    ax.grid(True,which='major',alpha=.17)
fig.suptitle('Canonical purification from local physical marginals',fontsize=15,fontweight='bold')
fig.get_layout_engine().set(rect=(0,.08,1,.93))
fig.text(.5,.025,'A: classical bit flips, bond cap 16.  B: classical t=0.2; analytic exterior bound.  C: quantum CP residual uses an 8-site reference; no certified exterior tail.',
         ha='center',fontsize=9,color='#444444')
fig.savefig(root/'summary.png',dpi=190)
fig.savefig(root/'summary.svg')
