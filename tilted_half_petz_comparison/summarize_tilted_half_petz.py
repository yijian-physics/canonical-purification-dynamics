from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    data = Path(__file__).parent/'tilted_half_petz_results'
    with (data/'summary.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    small = [r for r in rows if r['n']=='8' and r['width']=='3' and float(r['theta'])>0]
    theta = np.array([float(r['theta']) for r in small])
    for label in ('half_cp_error','unitary_cp_error','local_unitary_cp_error'):
        errors = [float(r[label]) for r in small]
        print(label,'log-log slope',np.polyfit(np.log(theta),np.log(errors),1)[0])
    fig,axes = plt.subplots(1,2,figsize=(11.5,4.5),layout='constrained')
    for label,title,marker in [('half_cp_error','Half-Petz, three sites','o'),
                               ('local_unitary_cp_error','Local-data unitary, three sites','s'),
                               ('unitary_cp_error','Exact-target unitary optimum','x')]:
        axes[0].loglog(theta,[float(r[label]) for r in small],marker=marker,label=title)
    axes[0].set(xlabel='Dephasing-axis tilt θ (radians)',ylabel='Global CP vector error',
                title='Eight spins, fixed three-site support')
    at_angle = [r for r in rows if float(r['theta'])==.1]
    for width,label,title,marker in [(3,'half_cp_error','Half-Petz, three sites','o'),
                                     (3,'local_unitary_cp_error','Local-data unitary, three sites','s'),
                                     (5,'half_cp_error','Half-Petz, five sites','^')]:
        selected = [r for r in at_angle if int(r['width'])==width]
        axes[1].loglog([int(r['n']) for r in selected],
                       [float(r[label]) for r in selected],marker=marker,label=title)
    axes[1].set(xlabel='Chain length N',ylabel='Global CP vector error',
                title='Tilt θ = 0.1, noise probability p = 0.1')
    for ax in axes:
        ax.grid(alpha=.2,which='both')
        ax.legend(fontsize=8)
    fig.savefig(data/'comparison.png',dpi=180)
    fig.savefig(data/'comparison.pdf')
    baseline = next(r for r in at_angle if r['n']=='8' and r['width']=='3')
    for label in ('half_cp_error','local_unitary_cp_error','unitary_cp_error'):
        spread = max(abs(float(r[label])-float(baseline[label]))
                     for r in at_angle if r['width']=='3')
        print(label,'maximum size difference',spread)
    print('Maximum dense reference error',max(float(r.get('reference_dense_error') or 0) for r in rows))
    print('Maximum half-Petz norm defect',max(abs(float(r['half_norm'])-1) for r in rows))


if __name__=='__main__':
    main()
