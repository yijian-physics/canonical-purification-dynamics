"""Recreate the global error and long-range benchmark figure."""
import csv
from pathlib import Path
import matplotlib.pyplot as plt

base=Path(__file__).resolve().parents[1]
def read(path):
    with path.open() as file:
        return list(csv.DictReader(file))

sizes=[8,48,64,128,256,512]
rows=[]
for n in sizes:
    file=next(base.glob(f'n{n}_*p0.12_a0.35_jensen_tdvp.csv'))
    rows.append(next(r for r in read(file) if float(r['time'])==.1))

fig,(ax,bx)=plt.subplots(1,2,figsize=(9.4,3.6),constrained_layout=True)
ax.semilogy(sizes,[float(r['scalar_bound'])+2*float(r['witness']) for r in rows],
            'o--',color='#9d6148',label='Scalar spectral bound')
ax.semilogy(sizes,[float(r['bound']) for r in rows],'o-',color='#1c6d69',label='Weighted bound')
ax.axhline(1e-5,color='#596474',ls=':',label='Requested tolerance')
ax.set(xlabel='Spins',ylabel='Upper bound on global CP error',title='Correlated mixed chain at t = 0.1')
ax.set_xscale('log',base=2)
ax.set_xticks([8,32,128,512],[8,32,128,512])
ax.legend(frameon=False,fontsize=8)
for n,marker,color in [(8,'o','#405f89'),(128,'s','#b6782c'),(512,'x','#175d50')]:
    values=read(next(base.glob(f'n{n}_t1_s5_parity0.6_x0.4_jensen_tdvp.csv')))[1:]
    bx.semilogy([float(r['time']) for r in values],[float(r['actual']) for r in values],
                marker=marker,color=color,label=f'{n} spins',markerfacecolor='none',
                lw=1,ms=8 if n==8 else 5)
bx.semilogy([float(r['time']) for r in values],[float(r['bound']) for r in values],
            'k--',label='512-spin upper bound',lw=1)
bx.set(xlabel='Physical time',ylabel='Global CP error',title='Noncommuting parity example')
bx.legend(frameon=False,fontsize=8)
for a in (ax,bx):
    a.spines[['top','right']].set_visible(False)
fig.savefig(base/'scaling.png',dpi=180)
